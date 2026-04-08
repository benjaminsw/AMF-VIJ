"""
File: eval_10_iters_mbatch.py | Version: 4.1.0 | Date: 2026-03-12
Abbr: EVAL-SEMA-MBATCH

CHANGELOG v4.1.0:
- Standardised __main__ dataset list to canonical 10-dataset set
- Removed 'multimodal-5' and 'Old-Faithful' (not in current benchmark scope)

CHANGELOG v4.0.0:
- Added bootstrap resampling per iteration: each of 10 iterations draws N_EVAL=5000 fresh samples
  from generated_pool (split_data['test']) and target_pool (generate_data n=25_000) independently
- Fixed near-zero std issue: previously same fixed tensors were reused across all iterations
- Removed compute_sliced_wasserstein_distance from both import blocks (deleted in wasserstein v3.0.0)
- compute_metrics_over_iterations: signature changed to accept generated_pool/target_pool + N_EVAL
- evaluate_single_dataset_comprehensive: builds pools once, passes to iteration loop for bootstrap

CHANGELOG v3.0.0:
- Replaced get_test_data(n_samples=200_000) with get_split_data() → split_data['test'] as generated_samples
- Added generate_data() call for fresh target_samples at n_eval = len(split_data['test'])
- Removed hardcoded 200_000 / 2000 sample sizes and truncation/validation block
- Updated compute_single_iteration_metrics to accept generated_samples param (no internal sampling)
- Updated compute_metrics_over_iterations to thread generated_samples through call chain
"""

import torch
import numpy as np
import os
import sys
import pickle
import csv
import logging
import importlib
from statistics import mean, stdev
import traceback
import random

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ── robust import of SEMA_MBATCH_vis (lives in main/unit_test/) ──────────────
def _import_sema_mbatch():
    """Import SequentialAMFVI and train_sequential_amf_vi from unit_test/SEMA_MBATCH_vis.py"""
    try:
        _this_dir = os.path.dirname(os.path.abspath(__file__))
        _vis_path = os.path.join(_this_dir, 'SEMA_MBATCH_vis.py')
        if not os.path.exists(_vis_path):
            raise FileNotFoundError(f"SEMA_MBATCH_vis.py not found at {_vis_path}")
        spec = importlib.util.spec_from_file_location("sema_mbatch_vis", _vis_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.SequentialAMFVI, mod.train_sequential_amf_vi
    except Exception as e:
        logging.error(f"Failed to import SEMA_MBATCH_vis.py: {e}")
        traceback.print_exc()
        raise

import importlib.util
SequentialAMFVI, train_sequential_amf_vi = _import_sema_mbatch()
# ── end SEMA-MBATCH-vis import ─────────────────────────────────────────────────

# Add project root for other imports
if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_cache import get_split_data
from data.data_generator import generate_data

# Import metric functions
from main.unit_test.evaluate_threeflows_amf_vi_weights_log import (
    compute_cross_entropy_surrogate,
    compute_kl_divergence_metric,
)
from main.unit_test.evaluate_threeflows_amf_vi_wasserstein import (
    compute_full_wasserstein_distance,
)
from main.unit_test.evaluate_threeflows_amf_vi_mmd import (
    compute_mmd_comparison,
    compute_polynomial_mmd_comparison,
)


random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
np.random.seed(2025)

# Bootstrap sample size per iteration — fixed for consistent metric estimation
N_EVAL = 5000
# Target pool size — must be >> N_EVAL to avoid correlated bootstrap draws
N_TARGET_POOL = 25_000


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_single_iteration_metrics(target_samples, generated_samples, flow_model, dataset_name):
    """Compute all metrics for a single iteration.

    Args:
        target_samples:    Fresh samples from generate_data() — used for NLL/KL target
        generated_samples: Cached split_data['test'] — used for W2/MMD comparison
        flow_model:        Trained flow model — used for log_prob (NLL)
        dataset_name:      Dataset identifier for logging
    """
    metrics = {}

    try:
        logging.info(f"        Target: {target_samples.shape[0]} samples, "
                     f"Generated: {generated_samples.shape[0]} samples")

        # 1. NLL
        try:
            nll = compute_cross_entropy_surrogate(target_samples, flow_model)
            metrics['nll'] = nll
        except Exception as e:
            logging.error(f"Error computing NLL for {dataset_name}: {e}")
            traceback.print_exc()
            metrics['nll'] = None

        # 2. KL Divergence
        try:
            kl_div = compute_kl_divergence_metric(target_samples, generated_samples, dataset_name)
            metrics['kl_divergence'] = kl_div
        except Exception as e:
            logging.error(f"Error computing KL divergence for {dataset_name}: {e}")
            traceback.print_exc()
            metrics['kl_divergence'] = None

        # 3. Full Wasserstein Distance
        try:
            full_wd = compute_full_wasserstein_distance(target_samples, generated_samples)
            metrics['full_wasserstein'] = full_wd
        except Exception as e:
            logging.error(f"Error computing Full Wasserstein for {dataset_name}: {e}")
            traceback.print_exc()
            metrics['full_wasserstein'] = None

        # 4. Gaussian MMD (unbiased + biased)
        try:
            gaussian_mmd = compute_mmd_comparison(target_samples, generated_samples, sigma=1.0)
            metrics['gaussian_mmd_unbiased'] = gaussian_mmd['mmd_unbiased']
            metrics['gaussian_mmd_biased']   = gaussian_mmd['mmd_biased']
        except Exception as e:
            logging.error(f"Error computing Gaussian MMD for {dataset_name}: {e}")
            traceback.print_exc()
            metrics['gaussian_mmd_unbiased'] = None
            metrics['gaussian_mmd_biased']   = None

    except Exception as e:
        logging.error(f"Critical error in compute_single_iteration_metrics for {dataset_name}: {e}")
        traceback.print_exc()
        return None

    return metrics


def compute_metrics_over_iterations(target_pool, generated_pool, flow_model, dataset_name, n_iterations=10):
    """Compute metrics over multiple iterations with bootstrap resampling.

    Each iteration independently resamples N_EVAL=5000 samples from target_pool and
    generated_pool, producing meaningful mean/std estimates across iterations.

    Args:
        target_pool:       Large pool of fresh target samples (N_TARGET_POOL=25_000)
        generated_pool:    Cached split_data['test'] pool
        flow_model:        Trained flow model
        dataset_name:      Dataset identifier
        n_iterations:      Number of bootstrap repetitions
    """
    all_metrics = {
        'nll': [],
        'kl_divergence': [],
        'full_wasserstein': [],
        'gaussian_mmd_unbiased': [],
        'gaussian_mmd_biased': [],
    }

    logging.info(f"    Computing metrics over {n_iterations} bootstrap iterations (N_EVAL={N_EVAL})...")

    for iteration in range(n_iterations):
        logging.info(f"      Iteration {iteration + 1}/{n_iterations}")
        try:
            # ── Bootstrap resample N_EVAL from each pool independently ─────────
            target_idx    = torch.randint(0, len(target_pool),    (N_EVAL,))
            generated_idx = torch.randint(0, len(generated_pool), (N_EVAL,))
            target_samples    = target_pool[target_idx]
            generated_samples = generated_pool[generated_idx]

            metrics = compute_single_iteration_metrics(
                target_samples, generated_samples, flow_model, dataset_name
            )
            if metrics is not None:
                for key in all_metrics:
                    if metrics.get(key) is not None:
                        all_metrics[key].append(metrics[key])
        except Exception as e:
            logging.error(f"Error in iteration {iteration + 1} for {dataset_name}: {e}")
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

def evaluate_single_dataset_comprehensive(dataset_name, n_iterations=10):
    """Evaluate a single dataset with all metrics over multiple iterations."""

    logging.info(f"\n{'='*60}")
    logging.info(f"Comprehensive Evaluation [SEMA-MBATCH]: {dataset_name.upper()} ({n_iterations} iterations)")
    logging.info(f"{'='*60}")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ── Load cached test split → generated_pool ──────────────────────────
        split_data      = get_split_data(dataset_name)
        generated_pool  = split_data['test'].to(device)
        logging.info(f"  {dataset_name}: generated_pool size={len(generated_pool)} from cached test split")

        # ── Generate fresh target_pool (>> N_EVAL for low-correlation bootstrap) ─
        target_pool = generate_data(dataset_name, n_samples=N_TARGET_POOL).to(device)
        logging.info(f"  {dataset_name}: target_pool generated fresh, shape={target_pool.shape} (N_TARGET_POOL={N_TARGET_POOL})")

        # ── Load SEMA-MBATCH model ────────────────────────────────────────────
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')

        if not os.path.exists(model_path):
            logging.error(f"Model not found for {dataset_name} at {model_path}. Skipping.")
            return None

        logging.info(f"  Loading SEMA-MBATCH model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)

        model = saved_data['model']

        # ── Log SEMA-MBATCH metadata from saved file ──────────────────────────
        model_version = saved_data.get('version', 'unknown')
        model_abbr    = saved_data.get('abbr',    'unknown')
        tau           = saved_data.get('tau',  'N/A')
        beta          = saved_data.get('beta', 'N/A')
        M             = saved_data.get('M',    'N/A')
        logging.info(f"  Model: {model_abbr} v{model_version} | τ={tau} | β={beta} | M={M}")

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

        # ── Weight extraction (SEMA-MBATCH uses model.weights directly) ───────
        if hasattr(model, 'weights_trained') and model.weights_trained:
            learned_weights = model.weights.detach().cpu().numpy()
            logging.info(f"  Using trained simplex weights: {learned_weights}")
        else:
            learned_weights = np.ones(len(model.flows)) / len(model.flows)
            logging.info(f"  weights_trained=False — using uniform weights for evaluation")

        # ── Evaluate mixture model ────────────────────────────────────────────
        logging.info(f"  Evaluating mixture model...")
        mixture_metrics = compute_metrics_over_iterations(
            target_pool, generated_pool, model, dataset_name, n_iterations
        )

        # ── Evaluate individual flows ─────────────────────────────────────────
        logging.info(f"  Evaluating individual flows...")
        individual_metrics = {}
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            logging.info(f"    Flow {i+1}/{len(flow_names)}: {name}")
            try:
                flow_metrics = compute_metrics_over_iterations(
                    target_pool, generated_pool, flow, dataset_name, n_iterations
                )
                individual_metrics[name] = flow_metrics
            except Exception as e:
                logging.error(f"Error evaluating flow {name} for {dataset_name}: {e}")
                traceback.print_exc()
                individual_metrics[name] = None

        results = {
            'dataset':            dataset_name,
            'mixture_metrics':    mixture_metrics,
            'individual_metrics': individual_metrics,
            'learned_weights':    learned_weights,
            'weights_trained':    getattr(model, 'weights_trained', False),
            'flow_names':         flow_names,
            'n_iterations':       n_iterations,
            # SEMA-MBATCH metadata
            'version': model_version,
            'abbr':    model_abbr,
            'tau':     tau,
            'beta':    beta,
            'M':       M,
        }

        # ── Print summary ─────────────────────────────────────────────────────
        logging.info(f"\n  Results Summary for {dataset_name} [{model_abbr} v{model_version}]:")
        logging.info(f"    Mixture Model Metrics (mean ± std):")
        for metric in ['nll', 'kl_divergence', 'full_wasserstein', 'gaussian_mmd_unbiased']:
            mean_val  = mixture_metrics.get(f'{metric}_mean')
            std_val   = mixture_metrics.get(f'{metric}_std')
            count_val = mixture_metrics.get(f'{metric}_count', 0)
            if mean_val is not None:
                logging.info(f"      {metric}: {mean_val:.6f} ± {std_val:.6f} (n={count_val})")
            else:
                logging.error(f"      {metric}: FAILED")

        logging.info(f"    Learned Weights: {learned_weights}")
        logging.info(f"    Weights Trained: {results['weights_trained']}")
        logging.info(f"    τ={tau} | β={beta} | M={M}")

        return results

    except Exception as e:
        logging.error(f"Critical error evaluating {dataset_name}: {e}")
        traceback.print_exc()
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def comprehensive_evaluation(n_iterations=100):
    """Run comprehensive evaluation on all datasets using SEMA-MBATCH models."""

    datasets = [
        'banana',
        'x_shape',
        'bimodal_shared',
        #'bimodal_different',
        #'multimodal',
        'two_moons',
        'rings',
        "multimodal-5",
        "BLR",
        "BPR",
        "Weibull",
        "Real-GMM2",
        #"Old-Faithful",
        #"Iris-3Class",
    ]

    all_results = {}

    logging.info(f"Starting Comprehensive Evaluation — SEMA-MBATCH ({n_iterations} iterations per metric)")
    logging.info(f"Datasets: {datasets}")

    for dataset_name in datasets:
        try:
            results = evaluate_single_dataset_comprehensive(dataset_name, n_iterations)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            logging.error(f"Failed to evaluate {dataset_name}: {e}")
            traceback.print_exc()
            continue

    if not all_results:
        logging.error("No datasets could be evaluated successfully.")
        return None

    # ── Build CSV ─────────────────────────────────────────────────────────────
    logging.info(f"\nCreating SEMA-MBATCH results CSV...")
    summary_data = []

    for dataset_name, results in all_results.items():
        weights_status  = "Yes" if results['weights_trained'] else "No"
        mixture_metrics = results['mixture_metrics']
        abbr            = results.get('abbr', 'SEMA-MBATCH')
        version         = results.get('version', 'unknown')
        tau             = results.get('tau', 'N/A')
        beta            = results.get('beta', 'N/A')
        M               = results.get('M', 'N/A')

        # Mixture row
        mixture_row = [
            dataset_name,
            'MIXTURE',
            abbr,
            version,
            tau, beta, M,
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
                    tau, beta, M,
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
                flow_row = (
                    [dataset_name, flow_name.upper(), abbr, version, tau, beta, M]
                    + [None] * 11
                    + [flow_weight, weights_status, n_iterations]
                )

            summary_data.append(flow_row)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    csv_filename = f'comprehensive_evaluation_SEMA-MBATCH_{n_iterations}_iterations.csv'
    csv_path     = os.path.join(results_dir, csv_filename)

    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'model', 'abbr', 'version', 'tau', 'beta', 'M',
                'nll_mean', 'nll_std',
                'kl_divergence_mean', 'kl_divergence_std',
                'full_wasserstein_mean', 'full_wasserstein_std',
                'gaussian_mmd_unbiased_mean', 'gaussian_mmd_unbiased_std',
                'gaussian_mmd_biased_mean', 'gaussian_mmd_biased_std',
                'weight', 'weights_trained', 'n_iterations',
            ])
            writer.writerows(summary_data)

        logging.info(f"{csv_filename} saved to {csv_path}")

    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
        traceback.print_exc()

    logging.info(f"\nEvaluation completed! Processed {len(all_results)} datasets successfully.")
    return all_results


if __name__ == "__main__":
    logging.info("Starting Comprehensive Evaluation — SEMA-MBATCH v3.0.0")
    logging.info("=" * 80)

    try:
        results = comprehensive_evaluation(n_iterations=10)
        if results:
            logging.info("\nComprehensive evaluation completed successfully!")
        else:
            logging.error("\nComprehensive evaluation failed — no results obtained.")
    except Exception as e:
        logging.error(f"\nCritical error in main execution: {e}")
        traceback.print_exc()
