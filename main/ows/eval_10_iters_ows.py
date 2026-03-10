"""
Version: 3.0.0
Abbr: EVAL-OWS
Base: EVAL-SEMA-MBATCH v2.0.1

CHANGELOG v3.0.0:
- Updated imports to load SequentialAMFVI from OWS_vis.py (v3.0.0) via importlib
- Updated model loading to read OWS-EM saved keys: max_iter, tol, eps, em_converged_iter, safety_net_triggered
- Removed sEMA metadata: tau, beta, M
- Updated CSV columns: replaced tau/beta/M with max_iter/tol/eps/em_converged_iter/safety_net_triggered
- Updated CSV filename to comprehensive_evaluation_OWS_{n}_iterations.csv
- All metric computation, dataset list, iteration logic unchanged
"""

import torch
import numpy as np
import os
import sys
import pickle
import csv
import importlib
from statistics import mean, stdev
import traceback
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── robust import of OWS_vis (lives in same directory: main/ows/) ────────────
# eval_10_iters_ows.py and OWS_vis.py are both in main/ows/
def _import_ows():
    """Import SequentialAMFVI and train_sequential_amf_vi from OWS_vis.py"""
    try:
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _vis_path = os.path.join(_this_dir, "ows", 'OWS_vis.py')
        if not os.path.exists(_vis_path):
            raise FileNotFoundError(f"OWS_vis.py not found at {_vis_path}")
        spec = importlib.util.spec_from_file_location("ows_vis", _vis_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SequentialAMFVI, mod.train_sequential_amf_vi
    except Exception as e:
        logger.error(f"Failed to import OWS_vis.py: {e}")
        traceback.print_exc()
        raise

import importlib.util
SequentialAMFVI, train_sequential_amf_vi = _import_ows()
# ── end OWS_vis import ───────────────────────────────────────────────────────

# Add project root for other imports
if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_cache import get_test_data

# Import metric functions
try:
    from .evaluate_threeflows_amf_vi_weights_log import (
        compute_cross_entropy_surrogate,
        compute_kl_divergence_metric,
    )
    from .evaluate_threeflows_amf_vi_wasserstein import (
        compute_sliced_wasserstein_distance,
        compute_full_wasserstein_distance,
    )
    from .evaluate_threeflows_amf_vi_mmd import (
        compute_mmd_comparison,
        compute_polynomial_mmd_comparison,
    )
except ImportError:
    from evaluate_threeflows_amf_vi_weights_log import (
        compute_cross_entropy_surrogate,
        compute_kl_divergence_metric,
    )
    from evaluate_threeflows_amf_vi_wasserstein import (
        compute_sliced_wasserstein_distance,
        compute_full_wasserstein_distance,
    )
    from evaluate_threeflows_amf_vi_mmd import (
        compute_mmd_comparison,
        compute_polynomial_mmd_comparison,
    )

random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
np.random.seed(2025)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers (unchanged from v1.0.0)
# ──────────────────────────────────────────────────────────────────────────────

def compute_single_iteration_metrics(target_samples, flow_model, dataset_name):
    """Compute all metrics for a single iteration."""
    metrics = {}

    try:
        with torch.no_grad():
            target_sample_count = target_samples.shape[0]
            generated_samples = flow_model.sample(target_sample_count)
            print(f"        Target: {target_sample_count} samples, "
                  f"Generated: {generated_samples.shape[0]} samples")

        # 1. NLL
        try:
            nll = compute_cross_entropy_surrogate(target_samples, flow_model)
            metrics['nll'] = nll
        except Exception as e:
            logger.error(f"Error computing NLL: {e}")
            traceback.print_exc()
            metrics['nll'] = None

        # 2. KL Divergence
        try:
            kl_div = compute_kl_divergence_metric(target_samples, flow_model, dataset_name)
            metrics['kl_divergence'] = kl_div
        except Exception as e:
            logger.error(f"Error computing KL divergence: {e}")
            traceback.print_exc()
            metrics['kl_divergence'] = None

        # 3. Full Wasserstein Distance
        try:
            full_wd = compute_full_wasserstein_distance(target_samples, generated_samples)
            metrics['full_wasserstein'] = full_wd
        except Exception as e:
            logger.error(f"Error computing Full Wasserstein: {e}")
            traceback.print_exc()
            metrics['full_wasserstein'] = None

        # 4. Gaussian MMD (unbiased + biased)
        try:
            gaussian_mmd = compute_mmd_comparison(target_samples, generated_samples, sigma=1.0)
            metrics['gaussian_mmd_unbiased'] = gaussian_mmd['mmd_unbiased']
            metrics['gaussian_mmd_biased']   = gaussian_mmd['mmd_biased']
        except Exception as e:
            logger.error(f"Error computing Gaussian MMD: {e}")
            traceback.print_exc()
            metrics['gaussian_mmd_unbiased'] = None
            metrics['gaussian_mmd_biased']   = None

    except Exception as e:
        logger.error(f"Critical error in compute_single_iteration_metrics: {e}")
        traceback.print_exc()
        return None

    return metrics


def compute_metrics_over_iterations(target_samples, flow_model, dataset_name, n_iterations=10):
    """Compute metrics over multiple iterations and return mean/std."""
    all_metrics = {
        'nll': [],
        'kl_divergence': [],
        'full_wasserstein': [],
        'gaussian_mmd_unbiased': [],
        'gaussian_mmd_biased': [],
    }

    print(f"    Computing metrics over {n_iterations} iterations...")

    for iteration in range(n_iterations):
        print(f"      Iteration {iteration + 1}/{n_iterations}")
        try:
            metrics = compute_single_iteration_metrics(target_samples, flow_model, dataset_name)
            if metrics is not None:
                for key in all_metrics:
                    if metrics.get(key) is not None:
                        all_metrics[key].append(metrics[key])
        except Exception as e:
            logger.error(f"Error in iteration {iteration + 1}: {e}")
            traceback.print_exc()
            continue

    summary_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            summary_metrics[f'{metric_name}_mean']  = mean(values)
            summary_metrics[f'{metric_name}_std']   = stdev(values) if len(values) > 1 else 0.0
            summary_metrics[f'{metric_name}_count'] = len(values)
        else:
            summary_metrics[f'{metric_name}_mean']  = None
            summary_metrics[f'{metric_name}_std']   = None
            summary_metrics[f'{metric_name}_count'] = 0

    return summary_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Per-dataset evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single_dataset_comprehensive(dataset_name, n_iterations=10, n_samples=2000):
    """Evaluate a single dataset with all metrics over multiple iterations."""

    print(f"\n{'='*60}")
    print(f"Comprehensive Evaluation [OWS-EM]: {dataset_name.upper()} ({n_iterations} iterations)")
    print(f"{'='*60}")

    try:
        # Load cached test split
        test_data = get_test_data(dataset_name, n_samples=200_000)

        # Ensure exact sample count
        if test_data.shape[0] != n_samples:
            print(f"Warning: test data has {test_data.shape[0]} samples, expected {n_samples}")
            if test_data.shape[0] > n_samples:
                test_data = test_data[:n_samples]
                print(f"Truncated to {n_samples} samples")
            elif test_data.shape[0] >= n_samples - 100:
                print(f"Using {test_data.shape[0]} samples (close enough)")
            else:
                logger.error(f"Insufficient samples ({test_data.shape[0]} < {n_samples})")
                print(f"Error: insufficient samples ({test_data.shape[0]} < {n_samples})")
                return None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_data = test_data.to(device)
        print(f"Final test data shape: {test_data.shape}")

        # ── Load OWS-EM model ─────────────────────────────────────────────────
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')

        if not os.path.exists(model_path):
            print(f"  ⚠️  Model not found for {dataset_name}. Skipping...")
            return None

        print(f"  Loading OWS-EM model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)

        model = saved_data['model']

        # ── Log OWS-EM metadata from saved file ──────────────────────────────
        model_version        = saved_data.get('version', 'unknown')
        model_abbr           = saved_data.get('abbr',    'unknown')
        max_iter             = saved_data.get('max_iter', 'N/A')
        tol                  = saved_data.get('tol',      'N/A')
        eps                  = saved_data.get('eps',      'N/A')
        em_converged_iter    = saved_data.get('em_converged_iter', 'N/A')
        safety_net_triggered = saved_data.get('safety_net_triggered', 'N/A')
        print(f"  Model: {model_abbr} v{model_version} | max_iter={max_iter} | tol={tol} | eps={eps}")
        print(f"  EM converged at iter: {em_converged_iter} | Safety net: {safety_net_triggered}")

        model = model.to(device)
        model.eval()

        # ── Flow names ────────────────────────────────────────────────────────
        flow_type_map = {
            'RealNVPFlow': 'realnvp', 'MAFFlow': 'maf', 'NAFFlowSimplified': 'naf',
            'NICEFlow': 'nice', 'IAFFlow': 'iaf', 'GaussianizationFlow': 'gaussianization',
            'GlowFlow': 'glow', 'TANFlow': 'tan', 'RBIGFlow': 'rbig',
        }
        flow_names = [
            flow_type_map.get(flow.__class__.__name__, flow.__class__.__name__.lower())
            for flow in model.flows
        ]

        # ── Weight extraction (OWS-EM uses model.weights directly, same as sEMA) ──
        if hasattr(model, 'weights_trained') and model.weights_trained:
            learned_weights = model.weights.detach().cpu().numpy()
            print(f"  Using trained simplex weights: {learned_weights}")
        else:
            learned_weights = np.ones(len(model.flows)) / len(model.flows)
            print(f"  ⚠️  weights_trained=False — using uniform weights for evaluation")

        # ── Evaluate mixture model ────────────────────────────────────────────
        print(f"  Evaluating mixture model...")
        mixture_metrics = compute_metrics_over_iterations(
            test_data, model, dataset_name, n_iterations
        )

        # ── Evaluate individual flows ─────────────────────────────────────────
        print(f"  Evaluating individual flows...")
        individual_metrics = {}
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            print(f"    Flow {i+1}/{len(flow_names)}: {name}")
            try:
                flow_metrics = compute_metrics_over_iterations(
                    test_data, flow, dataset_name, n_iterations
                )
                individual_metrics[name] = flow_metrics
            except Exception as e:
                logger.error(f"Error evaluating flow {name}: {e}")
                traceback.print_exc()
                individual_metrics[name] = None

        results = {
            'dataset':              dataset_name,
            'mixture_metrics':      mixture_metrics,
            'individual_metrics':   individual_metrics,
            'learned_weights':      learned_weights,
            'weights_trained':      getattr(model, 'weights_trained', False),
            'flow_names':           flow_names,
            'n_iterations':         n_iterations,
            # OWS-EM metadata
            'version':              model_version,
            'abbr':                 model_abbr,
            'max_iter':             max_iter,
            'tol':                  tol,
            'eps':                  eps,
            'em_converged_iter':    em_converged_iter,
            'safety_net_triggered': safety_net_triggered,
        }

        # ── Print summary ─────────────────────────────────────────────────────
        print(f"\n  Results Summary for {dataset_name} [{model_abbr} v{model_version}]:")
        print(f"    Mixture Model Metrics (mean ± std):")
        for metric in ['nll', 'kl_divergence', 'full_wasserstein', 'gaussian_mmd_unbiased']:
            mean_val  = mixture_metrics.get(f'{metric}_mean')
            std_val   = mixture_metrics.get(f'{metric}_std')
            count_val = mixture_metrics.get(f'{metric}_count', 0)
            if mean_val is not None:
                print(f"      {metric}: {mean_val:.6f} ± {std_val:.6f} (n={count_val})")
            else:
                print(f"      {metric}: FAILED")

        print(f"    Learned Weights: {learned_weights}")
        print(f"    Weights Trained: {results['weights_trained']}")
        print(f"    max_iter={max_iter} | tol={tol} | eps={eps}")
        print(f"    EM converged at iter: {em_converged_iter} | Safety net: {safety_net_triggered}")

        return results

    except Exception as e:
        logger.error(f"Critical error evaluating {dataset_name}: {e}")
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def comprehensive_evaluation(n_iterations=100):
    """Run comprehensive evaluation on all datasets using OWS-EM models."""

    datasets = [
        'banana',
        'x_shape',
        'bimodal_shared',
        'two_moons',
        'rings',
        "BLR",
        "BPR",
        "Weibull",
        "multimodal-5",
        "Real-GMM2",
        "Old-Faithful",
        "Iris-3Class",
    ]

    all_results = {}

    print(f"Starting Comprehensive Evaluation — OWS-EM ({n_iterations} iterations per metric)")
    print(f"Datasets: {datasets}")

    for dataset_name in datasets:
        try:
            results = evaluate_single_dataset_comprehensive(dataset_name, n_iterations)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            logger.error(f"Failed to evaluate {dataset_name}: {e}")
            traceback.print_exc()
            continue

    if not all_results:
        print("No datasets could be evaluated successfully.")
        return None

    # ── Build CSV ─────────────────────────────────────────────────────────────
    print(f"\nCreating OWS-EM results CSV...")
    summary_data = []

    for dataset_name, results in all_results.items():
        weights_status       = "Yes" if results['weights_trained'] else "No"
        mixture_metrics      = results['mixture_metrics']
        abbr                 = results.get('abbr', 'OWS')
        version              = results.get('version', 'unknown')
        max_iter             = results.get('max_iter', 'N/A')
        tol                  = results.get('tol', 'N/A')
        eps                  = results.get('eps', 'N/A')
        em_converged_iter    = results.get('em_converged_iter', 'N/A')
        safety_net_triggered = results.get('safety_net_triggered', 'N/A')

        # Mixture row
        mixture_row = [
            dataset_name,
            'MIXTURE',
            abbr,
            version,
            max_iter, tol, eps, em_converged_iter, safety_net_triggered,
            mixture_metrics.get('nll_mean'),
            mixture_metrics.get('nll_std'),
            mixture_metrics.get('kl_divergence_mean'),
            mixture_metrics.get('kl_divergence_std'),
            mixture_metrics.get('full_wasserstein_mean'),
            mixture_metrics.get('full_wasserstein_std'),
            mixture_metrics.get('gaussian_mmd_unbiased_mean'),
            mixture_metrics.get('gaussian_mmd_unbiased_std'),
            mixture_metrics.get('gaussian_mmd_biased_mean'),
            mixture_metrics.get('gaussian_mmd_biased_std'),
            'N/A',
            weights_status,
            n_iterations,
        ]
        summary_data.append(mixture_row)

        # Individual flow rows
        for i, flow_name in enumerate(results['flow_names']):
            individual_metrics = results['individual_metrics'].get(flow_name, {})
            flow_weight = results['learned_weights'][i]

            if individual_metrics is not None:
                flow_row = [
                    dataset_name,
                    flow_name.upper(),
                    abbr,
                    version,
                    max_iter, tol, eps, em_converged_iter, safety_net_triggered,
                    individual_metrics.get('nll_mean'),
                    individual_metrics.get('nll_std'),
                    individual_metrics.get('kl_divergence_mean'),
                    individual_metrics.get('kl_divergence_std'),
                    individual_metrics.get('full_wasserstein_mean'),
                    individual_metrics.get('full_wasserstein_std'),
                    individual_metrics.get('gaussian_mmd_unbiased_mean'),
                    individual_metrics.get('gaussian_mmd_unbiased_std'),
                    individual_metrics.get('gaussian_mmd_biased_mean'),
                    individual_metrics.get('gaussian_mmd_biased_std'),
                    flow_weight,
                    weights_status,
                    n_iterations,
                ]
            else:
                # Fill None for failed flow
                flow_row = (
                    [dataset_name, flow_name.upper(), abbr, version,
                     max_iter, tol, eps, em_converged_iter, safety_net_triggered]
                    + [None] * 11
                    + [flow_weight, weights_status, n_iterations]
                )

            summary_data.append(flow_row)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    csv_filename = f'comprehensive_evaluation_OWS_{n_iterations}_iterations.csv'
    csv_path     = os.path.join(results_dir, csv_filename)

    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'model', 'abbr', 'version',
                'max_iter', 'tol', 'eps', 'em_converged_iter', 'safety_net_triggered',
                'nll_mean', 'nll_std',
                'kl_divergence_mean', 'kl_divergence_std',
                'full_wasserstein_mean', 'full_wasserstein_std',
                'gaussian_mmd_unbiased_mean', 'gaussian_mmd_unbiased_std',
                'gaussian_mmd_biased_mean', 'gaussian_mmd_biased_std',
                'weight', 'weights_trained', 'n_iterations',
            ])
            writer.writerows(summary_data)

        print(f"✅ {csv_filename} saved to {csv_path}")

    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        traceback.print_exc()

    print(f"\nEvaluation completed! Processed {len(all_results)} datasets successfully.")
    return all_results


if __name__ == "__main__":
    print("Starting Comprehensive Evaluation — OWS-EM v3.0.0")
    print("=" * 80)

    try:
        results = comprehensive_evaluation(n_iterations=10)
        if results:
            print("\n🎉 Comprehensive evaluation completed successfully!")
        else:
            print("\n❌ Comprehensive evaluation failed — no results obtained.")
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}")
        print(f"\n💥 Critical error in main execution: {e}")
        traceback.print_exc()
