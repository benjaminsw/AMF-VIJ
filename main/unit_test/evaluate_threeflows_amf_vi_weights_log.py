"""
File: evaluate_threeflows_amf_vi_weights_log.py | Version: 2.0.0 | Date: 2026-03-04

CHANGELOG v2.0.0:
- compute_kl_divergence_metric: replaced (flow_model) param with (generated_samples); removed internal flow_model.sample(2000) calls
- KDE fallback: now uses passed generated_samples instead of re-sampling; replaced print() with logging.error()
- Removed broken histogram fallback (compute_kl_divergence_histogram was commented out, fallback was dead code)
- Added import logging; all warnings/errors now routed through logging.error()
"""

import torch
import torch.nn.functional as F
import numpy as np
import logging
from .SEMA_MBATCH_vis import SequentialAMFVI, train_sequential_amf_vi
from data.data_cache import get_test_data
from amf_vi.kde_kl_divergence import compute_kde_kl_divergence
import os
import pickle
import csv

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)


def compute_cross_entropy_surrogate(target_samples, flow_model):
    """Compute cross-entropy surrogate for KL divergence: -E_p[log q(x)]"""
    with torch.no_grad():
        log_q = flow_model.log_prob(target_samples)
        return -log_q.mean().item()


def compute_percentage_improvement(target_samples, mixture_model, baseline_flow):
    """Compute percentage improvement of mixture model over a single baseline flow."""
    mixture_cross_entropy  = compute_cross_entropy_surrogate(target_samples, mixture_model)
    baseline_cross_entropy = compute_cross_entropy_surrogate(target_samples, baseline_flow)

    if baseline_cross_entropy == 0:
        return 0.0

    improvement = ((baseline_cross_entropy - mixture_cross_entropy) / baseline_cross_entropy) * 100
    return improvement


def compute_kl_divergence_metric(target_samples, generated_samples, dataset_name):
    """Compute KL divergence using KDE-based approach.

    Args:
        target_samples:    Ground-truth samples from p_data (used as KDE reference)
        generated_samples: Pre-computed samples (split_data['test'] or flow samples)
        dataset_name:      Dataset identifier for error logging
    """
    with torch.no_grad():
        try:
            kl_divergence = compute_kde_kl_divergence(
                target_samples=target_samples,
                generated_samples=generated_samples,
                grid_resolution=100,
                bandwidth_method='scott',
                epsilon=1e-10
            )
            return kl_divergence

        except Exception as e:
            logging.error(f"KDE-based KL divergence failed for {dataset_name}: {e}")
            raise


def evaluate_individual_flows(model, test_data, generated_samples, flow_names, dataset_name):
    """Evaluate each individual flow against test data.

    Args:
        model:             Full mixture model (for weight extraction only)
        test_data:         Ground-truth samples for CE computation
        generated_samples: Pre-computed samples for KL computation
        flow_names:        List of flow name strings
        dataset_name:      Dataset identifier for error logging
    """
    individual_metrics = {}

    with torch.no_grad():
        for i, (flow, name) in enumerate(zip(model.flows, flow_names)):
            kl_divergence = compute_kl_divergence_metric(test_data, generated_samples, dataset_name)
            cross_entropy = compute_cross_entropy_surrogate(test_data, flow)

            individual_metrics[name] = {
                'kl_divergence':          kl_divergence,
                'cross_entropy_surrogate': cross_entropy,
            }

    return individual_metrics


def evaluate_single_sequential_dataset(dataset_name, target_samples=None, generated_samples=None):
    """Evaluate or train+evaluate a single Sequential model.

    When called standalone (target_samples / generated_samples are None), falls back to
    get_test_data(n_samples=200_000) for target_samples and flow_model.sample() for
    generated_samples to preserve backward compatibility for direct script execution.
    When called from eval_10_iters_mbatch, caller must pass both pre-computed tensors.
    """

    logging.info(f"\n{'='*50}")
    logging.info(f"Evaluating Sequential {dataset_name.upper()} dataset")
    logging.info(f"{'='*50}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Load test data (standalone fallback only) ─────────────────────────────
    if target_samples is None:
        target_samples = get_test_data(dataset_name, n_samples=200_000).to(device)
        logging.info(f"  Standalone mode: loaded {len(target_samples)} target samples via get_test_data()")
    else:
        target_samples = target_samples.to(device)

    # ── Load or train model ───────────────────────────────────────────────────
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')

    if os.path.exists(model_path):
        logging.info(f"Loading existing Sequential model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
    else:
        logging.info(f"Training new Sequential model for {dataset_name}")
        model, _, _ = train_sequential_amf_vi(dataset_name, show_plots=False, save_plots=False)

        with open(model_path, 'wb') as f:
            pickle.dump({'model': model, 'dataset': dataset_name}, f)

    model = model.to(device)

    # ── Flow names ────────────────────────────────────────────────────────────
    flow_type_map = {
        'RealNVPFlow': 'realnvp', 'MAFFlow': 'maf', 'NAFFlowSimplified': 'naf',
        'NICEFlow': 'nice', 'IAFFlow': 'iaf', 'GaussianizationFlow': 'gaussianization',
        'GlowFlow': 'glow', 'TANFlow': 'tan', 'RBIGFlow': 'rbig',
    }
    flow_names = []
    for flow in model.flows:
        flow_name = flow_type_map.get(flow.__class__.__name__, flow.__class__.__name__.lower())
        flow_names.append(flow_name)

    # ── Standalone fallback: sample generated_samples from mixture ────────────
    model.eval()
    if generated_samples is None:
        with torch.no_grad():
            generated_samples = model.sample(len(target_samples))
        logging.info(f"  Standalone mode: sampled {len(generated_samples)} generated_samples from mixture")
    else:
        generated_samples = generated_samples.to(device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    kl_divergence = compute_kl_divergence_metric(target_samples, generated_samples, dataset_name)
    cross_entropy = compute_cross_entropy_surrogate(target_samples, model)

    percentage_improvements = {}
    for i, name in enumerate(flow_names):
        improvement = compute_percentage_improvement(target_samples, model, model.flows[i])
        percentage_improvements[f'vs_{name}'] = improvement

    individual_flow_metrics = evaluate_individual_flows(
        model, target_samples, generated_samples, flow_names, dataset_name
    )

    # ── Learned weights ───────────────────────────────────────────────────────
    if model.weights_trained:
        logging.info('learned weights extracted')
        if hasattr(model, 'log_weights'):
            learned_weights = F.softmax(model.log_weights, dim=0).detach().cpu().numpy()
        else:
            learned_weights = model.weights.detach().cpu().numpy()
    else:
        learned_weights = np.ones(len(model.flows)) / len(model.flows)

    results = {
        'dataset':               dataset_name,
        'kl_divergence':         kl_divergence,
        'cross_entropy_surrogate': cross_entropy,
        'percentage_improvements': percentage_improvements,
        'individual_flow_metrics': individual_flow_metrics,
        'learned_weights':       learned_weights,
        'weights_trained':       model.weights_trained,
        'flow_names':            flow_names,
    }

    logging.info(f"Overall Sequential Mixture Results for {dataset_name}:")
    logging.info(f"  KL Divergence: {kl_divergence:.3f}")
    logging.info(f"  Cross-Entropy Surrogate: {cross_entropy:.3f}")
    for name, improvement in percentage_improvements.items():
        logging.info(f"  % Improvement {name}: {improvement:.1f}%")
    logging.info(f"  Learned Weights: {learned_weights}")
    logging.info(f"  Weights Trained: {model.weights_trained}")

    return results


def comprehensive_sequential_evaluation():
    """Comprehensive evaluation of all Sequential AMF-VI models (standalone)."""

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
            results = evaluate_single_sequential_dataset(dataset_name)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            logging.error(f"Failed to evaluate {dataset_name}: {e}")
            continue

    if not all_results:
        logging.error("No Sequential models could be trained/evaluated.")
        return None

    summary_data = []
    for dataset_name, results in all_results.items():
        weights_status = "Yes" if results['weights_trained'] else "No"

        for i, flow_name in enumerate(results['flow_names']):
            improvement_key = f'vs_{flow_name}'
            improvement     = results['percentage_improvements'].get(improvement_key, 0.0)

            individual_metrics = results['individual_flow_metrics'].get(flow_name, {})
            flow_kl     = individual_metrics.get('kl_divergence', 0.0)
            flow_ce     = individual_metrics.get('cross_entropy_surrogate', 0.0)
            flow_weight = (results['learned_weights'][i]
                           if results['weights_trained']
                           else 1 / len(results['learned_weights']))

            summary_data.append([
                dataset_name,
                results['kl_divergence'],
                results['cross_entropy_surrogate'],
                flow_name.upper(),
                flow_kl,
                flow_ce,
                flow_weight,
                improvement,
                weights_status,
            ])

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, 'sequential_comprehensive_metrics.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'dataset', 'mixture_kl_divergence', 'mixture_cross_entropy_surrogate',
                'flow', 'flow_kl_divergence', 'flow_cross_entropy_surrogate',
                'flow_weight', 'percentage_improvement', 'weights_trained',
            ])
            writer.writerows(summary_data)
        logging.info('sequential_comprehensive_metrics.csv successfully created')
    except Exception as e:
        logging.error(f"Error saving CSV: {e}")

    return all_results


if __name__ == "__main__":
    results = comprehensive_sequential_evaluation()
