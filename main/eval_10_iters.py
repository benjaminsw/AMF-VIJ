"""
File: eval_10_iters.py | Version: 2.0.0 | Date: 2026-03-10
Abbr: EVAL-OG-BOOTSTRAP

CHANGELOG v2.0.0:
- Added bootstrap resampling per iteration: each iteration draws N_EVAL=5000 samples
  from generated_pool (split_data['test']) and target_pool (generate_data n=25_000) independently
- Fixed seed block: added torch.manual_seed(2025) and np.random.seed(2025) (were missing)
- Replaced get_test_data(n_samples=200_000) with get_split_data()['test'] as generated_pool
- Added generate_data() for fresh target_pool; removed truncation/validation guard block
- compute_metrics_over_iterations: signature changed to (target_pool, generated_pool, ...)
- compute_single_iteration_metrics: now accepts generated_samples param (no internal sampling)
- Replaced all bare print() calls with logging.info/error; added logging.basicConfig
"""
import torch
import numpy as np
import os
import pickle
import csv
import sys
import logging
import traceback
import random
from statistics import mean, stdev

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- robust imports ---
if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from .threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
except Exception:
    try:
        from main.threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
    except Exception:
        from threeflows_amf_vi_weights_log import SequentialAMFVI, train_sequential_amf_vi
# --- end robust import header ---

from data.data_cache import get_split_data
from data.data_generator import generate_data

from .evaluate_threeflows_amf_vi_weights_log import (
    compute_cross_entropy_surrogate,
    compute_kl_divergence_metric,
)
from .evaluate_threeflows_amf_vi_wasserstein import (
    compute_full_wasserstein_distance,
)
from .evaluate_threeflows_amf_vi_mmd import (
    compute_mmd_comparison,
    compute_polynomial_mmd_comparison,
)

random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
np.random.seed(2025)

# Bootstrap sample size per iteration
N_EVAL = 5000
# Target pool size — must be >> N_EVAL
N_TARGET_POOL = 25_000

import random
random.seed(2025)


def compute_single_iteration_metrics(target_samples, generated_samples, flow_model, dataset_name):
    """Compute all metrics for a single bootstrap iteration.

    Args:
        target_samples:    N_EVAL samples resampled from target_pool (for NLL/KL)
        generated_samples: N_EVAL samples resampled from generated_pool (for W2/MMD)
        flow_model:        Trained flow model (for log_prob / NLL)
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
        
        # 5. Gaussian MMD (unbiased and biased)
        try:
            gaussian_mmd = compute_mmd_comparison(target_samples, generated_samples, sigma=1.0)
            metrics['gaussian_mmd_unbiased'] = gaussian_mmd['mmd_unbiased']
            metrics['gaussian_mmd_biased'] = gaussian_mmd['mmd_biased']
        except Exception as e:
            logging.error(f"Error computing Gaussian MMD for {dataset_name}: {e}")
            traceback.print_exc()
            metrics['gaussian_mmd_unbiased'] = None
            metrics['gaussian_mmd_biased'] = None

    except Exception as e:
        logging.error(f"Critical error in compute_single_iteration_metrics for {dataset_name}: {e}")
        traceback.print_exc()
        return None

    return metrics


def compute_metrics_over_iterations(target_pool, generated_pool, flow_model, dataset_name, n_iterations=10):
    """Compute metrics over multiple bootstrap iterations.

    Each iteration independently resamples N_EVAL=5000 from target_pool and generated_pool.

    Args:
        target_pool:    Fresh generate_data pool (N_TARGET_POOL=25_000)
        generated_pool: Cached split_data['test'] pool
        flow_model:     Trained flow model
        dataset_name:   Dataset identifier
        n_iterations:   Number of bootstrap repetitions
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
            # Bootstrap resample N_EVAL from each pool independently
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

def evaluate_single_dataset_comprehensive(dataset_name, n_iterations=10):
    """Evaluate a single dataset with all metrics over multiple bootstrap iterations."""

    logging.info(f"\n{'='*60}")
    logging.info(f"Comprehensive Evaluation [OG]: {dataset_name.upper()} ({n_iterations} iterations)")
    logging.info(f"{'='*60}")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build generated_pool from cached test split
        split_data     = get_split_data(dataset_name)
        generated_pool = split_data['test'].to(device)
        logging.info(f"  {dataset_name}: generated_pool size={len(generated_pool)} from cached test split")

        # Build fresh target_pool
        target_pool = generate_data(dataset_name, n_samples=N_TARGET_POOL).to(device)
        logging.info(f"  {dataset_name}: target_pool shape={target_pool.shape} (N_TARGET_POOL={N_TARGET_POOL})")

        # Load model
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        os.makedirs(results_dir, exist_ok=True)
        model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')

        if not os.path.exists(model_path):
            logging.error(f"  Model not found for {dataset_name} at {model_path}. Skipping.")
            return None

        logging.info(f"  Loading existing model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']

        model = model.to(device)
        model.eval()

        flow_type_map = {
            'RealNVPFlow': 'realnvp', 'MAFFlow': 'maf', 'NAFFlowSimplified': 'naf',
            'NICEFlow': 'nice', 'IAFFlow': 'iaf', 'GaussianizationFlow': 'gaussianization',
            'GlowFlow': 'glow', 'TANFlow': 'tan', 'RBIGFlow': 'rbig',
        }
        flow_names = [
            flow_type_map.get(flow.__class__.__name__, flow.__class__.__name__.lower())
            for flow in model.flows
        ]

        if hasattr(model, 'weights_trained') and model.weights_trained:
            if hasattr(model, 'log_weights'):
                learned_weights = torch.softmax(model.log_weights, dim=0).detach().cpu().numpy()
            else:
                learned_weights = model.weights.detach().cpu().numpy()
        else:
            learned_weights = np.ones(len(model.flows)) / len(model.flows)

        # Evaluate mixture model
        logging.info(f"  Evaluating mixture model...")
        mixture_metrics = compute_metrics_over_iterations(
            target_pool, generated_pool, model, dataset_name, n_iterations
        )

        # Evaluate individual flows
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
        }
        
        # Print summary
        logging.info(f"\n  Results Summary for {dataset_name}:")
        logging.info(f"    Mixture Model Metrics (mean ± std):")
        for metric in ['nll', 'kl_divergence', 'full_wasserstein', 'gaussian_mmd_unbiased']:
            mean_val = mixture_metrics.get(f'{metric}_mean')
            std_val = mixture_metrics.get(f'{metric}_std')
            count_val = mixture_metrics.get(f'{metric}_count', 0)
            if mean_val is not None:
                logging.info(f"      {metric}: {mean_val:.6f} ± {std_val:.6f} (n={count_val})")
            else:
                logging.error(f"      {metric}: FAILED")

        logging.info(f"    Learned Weights: {learned_weights}")
        logging.info(f"    Weights Trained: {results['weights_trained']}")

        return results

    except Exception as e:
        logging.error(f"Critical error evaluating {dataset_name}: {e}")
        traceback.print_exc()
        return None

def comprehensive_evaluation(n_iterations=100):
    """Run comprehensive evaluation on all datasets"""
    
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
        #"multimodal5_drop0",
        #"multimodal5_drop1",
        #"multimodal5_drop2",
        "Real-GMM2",
        "Old-Faithful",
        "Iris-3Class",
    ]
    # datasets = ['multimodal']
    all_results = {}
    
    logging.info(f"Starting Comprehensive Evaluation [OG] ({n_iterations} iterations per metric)")
    logging.info(f"Datasets: {datasets}")
    
    # Evaluate each dataset
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

    logging.info(f"\nCreating comprehensive results CSV...")
    summary_data = []
    
    for dataset_name, results in all_results.items():
        weights_status = "Yes" if results['weights_trained'] else "No"
        mixture_metrics = results['mixture_metrics']
        
        # Add mixture model row
        mixture_row = [
            dataset_name,
            'MIXTURE',
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
            'N/A',  # weight (not applicable for mixture)
            weights_status,
            n_iterations
        ]
        summary_data.append(mixture_row)
        
        # Add individual flow rows
        for i, flow_name in enumerate(results['flow_names']):
            individual_metrics = results['individual_metrics'].get(flow_name, {})
            flow_weight = results['learned_weights'][i]
            
            if individual_metrics is not None:
                flow_row = [
                    dataset_name,
                    flow_name.upper(),
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
                    n_iterations
                ]
            else:
                # Fill with None values if flow evaluation failed
                flow_row = [dataset_name, flow_name.upper()] + [None] * 11 + [flow_weight, weights_status, n_iterations]
            
            summary_data.append(flow_row)
    
    # Save comprehensive CSV
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_filename = f'comprehensive_evaluation_{n_iterations}_iterations.csv'
    csv_path = os.path.join(results_dir, csv_filename)
    
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'model', 
                'nll_mean', 'nll_std',
                'kl_divergence_mean', 'kl_divergence_std',
                'full_wasserstein_mean', 'full_wasserstein_std',
                'gaussian_mmd_unbiased_mean', 'gaussian_mmd_unbiased_std',
                'gaussian_mmd_biased_mean', 'gaussian_mmd_biased_std',
                'weight', 'weights_trained', 'n_iterations'
            ])
            writer.writerows(summary_data)
        
        logging.info(f"✅ {csv_filename} successfully created at {csv_path}")
        
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")
        traceback.print_exc()
    
    logging.info(f"\nEvaluation completed! Processed {len(all_results)} datasets successfully.")
    return all_results

if __name__ == "__main__":
    # Run comprehensive evaluation with 10 iterations
    logging.info("Starting Comprehensive Evaluation Script [OG v2.0.0]")
    logging.info("=" * 80)
    
    try:
        results = comprehensive_evaluation(n_iterations=10)
        if results:
            logging.info("\n🎉 Comprehensive evaluation completed successfully!")
        else:
            logging.error("\n❌ Comprehensive evaluation failed - no results obtained.")
    except Exception as e:
        logging.error(f"\n💥 Critical error in main execution: {e}")
        traceback.print_exc()
