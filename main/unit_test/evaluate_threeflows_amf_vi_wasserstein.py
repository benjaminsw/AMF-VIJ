"""
File: evaluate_threeflows_amf_vi_wasserstein.py | Version: 3.0.0 | Date: 2026-03-07

CHANGELOG v3.0.0:
- Removed compute_sliced_wasserstein_distance and all sliced W2 calls (mixture + per flow + CSV)
- ot.emd2: added numItermax=500_000 to resolve premature optimality warning
- compute_wasserstein_distance: default metric_type changed from 'sliced' to 'full'; sliced branch removed
- Standalone fallback: get_test_data n_samples changed from 200_000 to 5000; model.sample kept as len(target_samples)
- evaluate_individual_flows_wasserstein: removed sliced_wd call and sliced key from metrics dict
"""

import torch
import numpy as np
import logging
import ot
from .SEMA_MBATCH_vis import SequentialAMFVI, train_sequential_amf_vi
from data.data_cache import get_test_data
import os
import pickle
import csv

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)


def compute_full_wasserstein_distance(target_samples, generated_samples):
    """Compute full 2-Wasserstein distance using optimal transport."""
    target_np    = target_samples.detach().cpu().numpy()
    generated_np = generated_samples.detach().cpu().numpy()

    a = np.ones(len(target_np))    / len(target_np)
    b = np.ones(len(generated_np)) / len(generated_np)

    cost_matrix    = ot.dist(target_np, generated_np, metric='sqeuclidean')
    wasserstein_dist = ot.emd2(a, b, cost_matrix, numItermax=500_000)

    return np.sqrt(wasserstein_dist)


def compute_wasserstein_distance(target_samples, generated_samples, metric_type='full'):
    """Compute full W2 distance between target and pre-computed generated samples.

    Args:
        target_samples:    Ground-truth samples from p_data
        generated_samples: Pre-computed samples (split_data['test'] from orchestrator)
        metric_type:       'full' only
    """
    if metric_type == 'full':
        return compute_full_wasserstein_distance(target_samples, generated_samples)
    else:
        raise ValueError("metric_type must be 'full'")


def evaluate_individual_flows_wasserstein(model, target_samples, generated_samples, flow_names):
    """Evaluate each individual flow using both Wasserstein distances.

    Args:
        model:             Mixture model (for iterating flows)
        target_samples:    Ground-truth samples
        generated_samples: Pre-computed samples (split_data['test'])
        flow_names:        List of flow name strings
    """
    individual_metrics = {}

    with torch.no_grad():
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            full_wd = compute_wasserstein_distance(target_samples, generated_samples, 'full')

            individual_metrics[name] = {
                'full_wasserstein': full_wd,
            }

    return individual_metrics


def evaluate_single_sequential_dataset_wasserstein(dataset_name,
                                                    target_samples=None,
                                                    generated_samples=None):
    """Evaluate a single Sequential model using Wasserstein distances.

    When called standalone (target_samples / generated_samples are None), falls back to
    get_test_data(n_samples=200_000) for target_samples and flow_model.sample() for
    generated_samples. When called from eval_10_iters_mbatch, caller must pass both tensors.
    """

    logging.info(f"\n{'='*50}")
    logging.info(f"Evaluating Wasserstein for {dataset_name.upper()} dataset")
    logging.info(f"{'='*50}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load model ────────────────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')

    if os.path.exists(model_path):
        logging.info(f"Loading existing model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
    else:
        logging.info(f"Training new model for {dataset_name}")
        model, _, _ = train_sequential_amf_vi(dataset_name, show_plots=False, save_plots=False)
        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'dataset': dataset_name}, f)

    model = model.to(device)
    model.eval()

    # ── Standalone fallbacks ──────────────────────────────────────────────────
    if target_samples is None:
        target_samples = get_test_data(dataset_name, n_samples=5000).to(device)
        logging.info(f"  Standalone mode: loaded {len(target_samples)} target samples via get_test_data()")
    else:
        target_samples = target_samples.to(device)

    if generated_samples is None:
        with torch.no_grad():
            generated_samples = model.sample(len(target_samples))
        logging.info(f"  Standalone mode: sampled {len(generated_samples)} generated_samples from mixture")
    else:
        generated_samples = generated_samples.to(device)

    # ── Flow names ────────────────────────────────────────────────────────────
    flow_type_map = {
        'RealNVPFlow': 'realnvp', 'MAFFlow': 'maf', 'NAFFlowSimplified': 'naf',
        'NICEFlow': 'nice', 'IAFFlow': 'iaf', 'GaussianizationFlow': 'gaussianization',
        'GlowFlow': 'glow', 'TANFlow': 'tan', 'RBIGFlow': 'rbig',
    }
    flow_names = [
        flow_type_map.get(flow.__class__.__name__, flow.__class__.__name__.lower())
        for flow in model.flows
    ]

    # ── Metrics ───────────────────────────────────────────────────────────────
    mixture_full_wd = compute_wasserstein_distance(target_samples, generated_samples, 'full')

    individual_flow_metrics = evaluate_individual_flows_wasserstein(
        model, target_samples, generated_samples, flow_names
    )

    # ── Learned weights ───────────────────────────────────────────────────────
    if hasattr(model, 'weights_trained') and model.weights_trained:
        if hasattr(model, 'log_weights'):
            learned_weights = torch.softmax(model.log_weights, dim=0).detach().cpu().numpy()
        else:
            learned_weights = model.weights.detach().cpu().numpy()
    else:
        learned_weights = np.ones(len(model.flows)) / len(model.flows)

    results = {
        'dataset':                  dataset_name,
        'mixture_full_wasserstein': mixture_full_wd,
        'individual_flow_metrics':  individual_flow_metrics,
        'learned_weights':          learned_weights,
        'weights_trained':          getattr(model, 'weights_trained', False),
        'flow_names':               flow_names,
    }

    logging.info(f"Wasserstein Results for {dataset_name}:")
    logging.info(f"  Mixture Full Wasserstein: {mixture_full_wd:.4f}")
    logging.info(f"  Learned Weights: {learned_weights}")

    return results


def comprehensive_wasserstein_evaluation():
    """Comprehensive Wasserstein evaluation of all datasets (standalone)."""

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

    for dataset_name in datasets:
        try:
            results = evaluate_single_sequential_dataset_wasserstein(dataset_name)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            logging.error(f"Failed to evaluate {dataset_name}: {e}")
            continue

    if not all_results:
        logging.error("No models could be evaluated.")
        return None

    summary_data = []
    for dataset_name, results in all_results.items():
        weights_status = "Yes" if results['weights_trained'] else "No"

        for i, flow_name in enumerate(results['flow_names']):
            individual_metrics = results['individual_flow_metrics'].get(flow_name, {})
            flow_full_wd = individual_metrics.get('full_wasserstein', 0.0)
            flow_weight  = results['learned_weights'][i]

            summary_data.append([
                dataset_name,
                results['mixture_full_wasserstein'],
                flow_name.upper(),
                flow_full_wd,
                flow_weight,
                weights_status,
            ])

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, 'wasserstein_comprehensive_metrics.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'mixture_full_wasserstein',
                'flow', 'flow_full_wasserstein',
                'flow_weight', 'weights_trained',
            ])
            writer.writerows(summary_data)
        logging.info('wasserstein_comprehensive_metrics.csv successfully created')
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")

    return all_results


if __name__ == "__main__":
    results = comprehensive_wasserstein_evaluation()
