"""
Evaluation script for Baseline EM Mixture of Flows models.
Computes metrics over 10 iterations for each dataset.
"""

import torch
import numpy as np
import os
import pickle
import csv
from statistics import mean, stdev
import traceback
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import generate_data

# Import model classes from separate module
from em_mixture_models import MixtureOfFlows, NormalizingFlow, AffineCouplingLayer

# Import evaluation functions from main directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main'))
from evaluate_threeflows_amf_vi_weights_log import (
    compute_cross_entropy_surrogate,
    compute_kl_divergence_metric
)
from evaluate_threeflows_amf_vi_wasserstein import (
    compute_full_wasserstein_distance
)
from evaluate_threeflows_amf_vi_mmd import (
    compute_mmd_comparison
)

# Set seed for reproducibility
torch.manual_seed(2025)
np.random.seed(2025)


def compute_nll_with_gradients(target_samples, model):
    """Compute NLL with proper gradient handling for EM mixture model"""
    try:
        # Ensure input has gradients
        if not target_samples.requires_grad:
            target_samples = target_samples.requires_grad_(True)
        
        # Compute log probability directly
        log_prob = model.compute_log_likelihood(target_samples)
        
        # Return negative log likelihood (mean)
        nll = -log_prob.mean().item()
        return nll
    except Exception as e:
        print(f"        Error in NLL wrapper: {e}")
        return None


def compute_single_iteration_metrics(target_samples, model, dataset_name):
    """Compute all metrics for a single iteration"""
    metrics = {}
    
    try:
        # Generate samples - ensure matching size
        with torch.no_grad():
            target_sample_count = target_samples.shape[0]
            generated_samples = model.sample(target_sample_count)
            print(f"        Target: {target_sample_count} samples, Generated: {generated_samples.shape[0]} samples")
        
        # 1. NLL (Negative Log-Likelihood)
        try:
            nll = compute_nll_with_gradients(target_samples, model)
            metrics['nll'] = nll
        except Exception as e:
            print(f"        Error computing NLL: {e}")
            metrics['nll'] = None
        
        # 2. KL Divergence
        try:
            kl_div = compute_kl_divergence_metric(target_samples, model, dataset_name)
            metrics['kl_divergence'] = kl_div
        except Exception as e:
            print(f"        Error computing KL divergence: {e}")
            metrics['kl_divergence'] = None
        
        # 3. Full Wasserstein Distance
        try:
            full_wd = compute_full_wasserstein_distance(target_samples, generated_samples)
            metrics['full_wasserstein'] = full_wd
        except Exception as e:
            print(f"        Error computing Full Wasserstein: {e}")
            metrics['full_wasserstein'] = None
        
        # 4. Gaussian MMD (unbiased and biased)
        try:
            gaussian_mmd = compute_mmd_comparison(target_samples, generated_samples, sigma=1.0)
            metrics['gaussian_mmd_unbiased'] = gaussian_mmd['mmd_unbiased']
            metrics['gaussian_mmd_biased'] = gaussian_mmd['mmd_biased']
        except Exception as e:
            print(f"        Error computing Gaussian MMD: {e}")
            metrics['gaussian_mmd_unbiased'] = None
            metrics['gaussian_mmd_biased'] = None
        
    except Exception as e:
        print(f"        Critical error in compute_single_iteration_metrics: {e}")
        traceback.print_exc()
        return None
    
    return metrics


def compute_metrics_over_iterations(target_samples, model, dataset_name, n_iterations=10):
    """Compute metrics over multiple iterations and return mean/std"""
    all_metrics = {
        'nll': [],
        'kl_divergence': [],
        'full_wasserstein': [],
        'gaussian_mmd_unbiased': [],
        'gaussian_mmd_biased': []
    }
    
    print(f"      Computing metrics over {n_iterations} iterations...")
    
    for iteration in range(n_iterations):
        print(f"        Iteration {iteration + 1}/{n_iterations}")
        
        try:
            metrics = compute_single_iteration_metrics(target_samples, model, dataset_name)
            if metrics is not None:
                for key in all_metrics.keys():
                    value = metrics.get(key)
                    # FIX: Only append finite values
                    if value is not None and np.isfinite(value):
                        all_metrics[key].append(value)
        except Exception as e:
            print(f"        Error in iteration {iteration + 1}: {e}")
            traceback.print_exc()
            continue
    
    # Calculate mean and std for each metric
    summary_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            # FIX: Use numpy for robust statistics with proper handling
            summary_metrics[f'{metric_name}_mean'] = float(np.mean(values))
            summary_metrics[f'{metric_name}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
            summary_metrics[f'{metric_name}_count'] = len(values)
        else:
            summary_metrics[f'{metric_name}_mean'] = None
            summary_metrics[f'{metric_name}_std'] = None
            summary_metrics[f'{metric_name}_count'] = 0
    
    return summary_metrics


def evaluate_single_dataset(dataset_name, n_iterations=10):
    """Evaluate EM mixture model for a single dataset with extensive debugging"""
    
    print(f"\n{'='*60}")
    print(f"Evaluating EM Mixture: {dataset_name.upper()} ({n_iterations} iterations)")
    print(f"{'='*60}")
    
    try:
        # Generate test data
        test_data = generate_data(dataset_name, n_samples=2000)
        
        # Verify sample count
        if test_data.shape[0] != 2000:
            print(f"  Warning: generate_data returned {test_data.shape[0]} samples instead of 2000")
            if test_data.shape[0] > 2000:
                test_data = test_data[:2000]
                print(f"  Truncated to 2000 samples")
            elif test_data.shape[0] >= 1900:
                print(f"  Using {test_data.shape[0]} samples (close enough to 2000)")
            else:
                print(f"  Error: Not enough samples generated ({test_data.shape[0]} < 2000)")
                return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_data = test_data.to(device)
        print(f"  Test data shape: {test_data.shape}")
        print(f"  Test data stats: mean={test_data.mean(dim=0)}, std={test_data.std(dim=0)}")
        print(f"  Test data range: min={test_data.min().item():.4f}, max={test_data.max().item():.4f}")
        
        # Load the EM mixture model
        baseline_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(baseline_dir, 'baseline_results')
        model_path = os.path.join(results_dir, f'em_mixture_{dataset_name}.pkl')
        
        if not os.path.exists(model_path):
            print(f"  Error: Model not found at {model_path}")
            return None
        
        print(f"  Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
            
        # === BACKWARD COMPATIBILITY FIX ===
        # Old models don't have scale_clip attribute in AffineCouplingLayer
        # Add it with default value to make old models work with new code
        print(f"  Applying backward compatibility fixes...")
        compatibility_fixes_applied = 0
        for flow_idx, flow in enumerate(model.flows):
            for layer_idx, layer in enumerate(flow.layers):
                if isinstance(layer, AffineCouplingLayer):
                    # Check if scale_clip exists, if not add it
                    if not hasattr(layer, 'scale_clip'):
                        layer.scale_clip = 2.0  # Default value from training script
                        compatibility_fixes_applied += 1
        
        if compatibility_fixes_applied > 0:
            print(f"    ✓ Applied {compatibility_fixes_applied} compatibility fixes (added scale_clip attribute)")
        else:
            print(f"    ✓ No fixes needed - model is up to date")
            
        
        # Move model to device and set to eval mode
        for flow in model.flows:
            flow.to(device)
            flow.eval()
        
        # Get mixture weights
        mixture_weights = model.mixture_weights.cpu().numpy()
        
        print(f"\n  Model info:")
        print(f"    Components: {model.K}")
        print(f"    Mixture weights: {mixture_weights}")
        print(f"    Weights sum: {mixture_weights.sum():.6f}")
        
        # ============ DEBUGGING SECTION ============
        print(f"\n  === MODEL DIAGNOSTICS ===")
        
        # Check mixture weights for issues
        print(f"    Mixture weights contain NaN: {np.isnan(mixture_weights).any()}")
        print(f"    Mixture weights contain Inf: {np.isinf(mixture_weights).any()}")
        print(f"    Mixture weights all positive: {(mixture_weights > 0).all()}")
        print(f"    Mixture weights sum to 1: {abs(mixture_weights.sum() - 1.0) < 1e-6}")
        
        # Check flow parameters for NaN/Inf
        print(f"\n    Flow parameter diagnostics:")
        for k, flow in enumerate(model.flows):
            has_nan = False
            has_inf = False
            param_count = 0
            
            for name, param in flow.named_parameters():
                param_count += 1
                if torch.isnan(param).any():
                    has_nan = True
                    print(f"      Flow {k}, {name}: contains NaN")
                if torch.isinf(param).any():
                    has_inf = True
                    print(f"      Flow {k}, {name}: contains Inf")
            
            if not has_nan and not has_inf:
                print(f"      Flow {k}: OK ({param_count} parameters, all finite)")
            else:
                print(f"      Flow {k}: CORRUPTED (NaN={has_nan}, Inf={has_inf})")
        
        # Test log-likelihood computation on small batch
        print(f"\n    Testing log-likelihood computation:")
        try:
            test_batch = test_data[:10].clone().requires_grad_(True)
            log_lik = model.compute_log_likelihood(test_batch)
            
            print(f"      Sample log-lik shape: {log_lik.shape}")
            print(f"      Sample log-lik values: {log_lik}")
            print(f"      Mean log-lik: {log_lik.mean().item():.4f}")
            print(f"      Log-lik contains NaN: {torch.isnan(log_lik).any().item()}")
            print(f"      Log-lik contains Inf: {torch.isinf(log_lik).any().item()}")
            print(f"      Log-lik range: [{log_lik.min().item():.4f}, {log_lik.max().item():.4f}]")
            
            if torch.isnan(log_lik).any() or torch.isinf(log_lik).any():
                print(f"      WARNING: Log-likelihood contains invalid values!")
                
                # Debug individual flow contributions
                print(f"\n      Debugging individual flows:")
                for k, flow in enumerate(model.flows):
                    try:
                        flow_log_prob = flow.log_prob(test_batch)
                        print(f"        Flow {k} log_prob: mean={flow_log_prob.mean().item():.4f}, "
                              f"min={flow_log_prob.min().item():.4f}, "
                              f"max={flow_log_prob.max().item():.4f}, "
                              f"NaN={torch.isnan(flow_log_prob).any().item()}, "
                              f"Inf={torch.isinf(flow_log_prob).any().item()}")
                    except Exception as e:
                        print(f"        Flow {k} FAILED: {e}")
        
        except Exception as e:
            print(f"      FAILED to compute log-likelihood: {e}")
            traceback.print_exc()
            print(f"      This model cannot be evaluated - returning None")
            return None
        
        # Test sampling
        print(f"\n    Testing sample generation:")
        try:
            test_samples = model.sample(100)
            print(f"      Generated samples shape: {test_samples.shape}")
            print(f"      Sample mean: {test_samples.mean(dim=0)}")
            print(f"      Sample std: {test_samples.std(dim=0)}")
            print(f"      Samples contain NaN: {torch.isnan(test_samples).any().item()}")
            print(f"      Samples contain Inf: {torch.isinf(test_samples).any().item()}")
            print(f"      Sample range: [{test_samples.min().item():.4f}, {test_samples.max().item():.4f}]")
            
            if torch.isnan(test_samples).any() or torch.isinf(test_samples).any():
                print(f"      WARNING: Generated samples contain invalid values!")
        
        except Exception as e:
            print(f"      FAILED to generate samples: {e}")
            traceback.print_exc()
            print(f"      This model cannot generate valid samples - returning None")
            return None
        
        # Check if model is completely broken
        if torch.isnan(log_lik).any() or torch.isnan(test_samples).any():
            print(f"\n  MODEL IS BROKEN: Contains NaN values")
            print(f"  Skipping evaluation for {dataset_name}")
            return None
        
        print(f"\n  === DIAGNOSTICS PASSED - PROCEEDING WITH EVALUATION ===")
        
        # ============ END DEBUGGING SECTION ============
        
        # Enable gradients for Jacobian computation
        test_data = test_data.requires_grad_(True)
        
        # Evaluate the mixture model
        print(f"\n  Evaluating mixture model...")
        mixture_metrics = compute_metrics_over_iterations(test_data, model, dataset_name, n_iterations)
        
        results = {
            'dataset': dataset_name,
            'mixture_metrics': mixture_metrics,
            'mixture_weights': mixture_weights,
            'n_components': model.K,
            'n_iterations': n_iterations
        }
        
        # Print summary
        print(f"\n  Results Summary for {dataset_name}:")
        print(f"    Mixture Model Metrics (mean ± std):")
        for metric in ['nll', 'kl_divergence', 'full_wasserstein', 'gaussian_mmd_unbiased', 'gaussian_mmd_biased']:
            mean_val = mixture_metrics.get(f'{metric}_mean')
            std_val = mixture_metrics.get(f'{metric}_std')
            count_val = mixture_metrics.get(f'{metric}_count', 0)
            if mean_val is not None:
                print(f"      {metric}: {mean_val:.6f} ± {std_val:.6f} (n={count_val})")
            else:
                print(f"      {metric}: FAILED (n={count_val})")
        
        print(f"    Mixture Weights: {mixture_weights}")
        
        return results
        
    except Exception as e:
        print(f"  Critical error evaluating {dataset_name}: {e}")
        traceback.print_exc()
        return None


def comprehensive_evaluation(n_iterations=10):
    """Run comprehensive evaluation on all datasets"""
    
    # datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 
                # 'multimodal', 'two_moons', 'rings']
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 
            'multimodal', 'two_moons', 'rings', 'BLR', 'BPR', 'Weibull', 
            'multimodal-5']
    
    all_results = {}
    
    print(f"Starting EM Mixture Evaluation ({n_iterations} iterations per metric)")
    print(f"Datasets: {datasets}")
    print("=" * 80)
    
    # Evaluate each dataset
    for dataset_name in datasets:
        try:
            results = evaluate_single_dataset(dataset_name, n_iterations)
            if results is not None:
                all_results[dataset_name] = results
        except Exception as e:
            print(f"Failed to evaluate {dataset_name}: {e}")
            traceback.print_exc()
            continue
    
    if not all_results:
        print("No datasets could be evaluated successfully.")
        return None
    
    # Create CSV with results
    print(f"\nCreating results CSV...")
    summary_data = []
    
    for dataset_name, results in all_results.items():
        mixture_metrics = results['mixture_metrics']
        weights_str = ', '.join([f'{w:.4f}' for w in results['mixture_weights']])
        
        # Add row for this dataset
        row = [
            dataset_name,
            'EM_MIXTURE',
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
            results['n_components'],
            weights_str,
            n_iterations
        ]
        summary_data.append(row)
    
    # Save CSV
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(baseline_dir, 'baseline_results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_filename = f'baseline_em_evaluation_{n_iterations}_iterations.csv'
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
                'n_components', 'mixture_weights', 'n_iterations'
            ])
            writer.writerows(summary_data)
        
        print(f"\n{csv_filename} successfully created at {csv_path}")
        print(f"Evaluated {len(all_results)} datasets successfully")
        
    except Exception as e:
        print(f"\nError saving CSV: {e}")
        traceback.print_exc()
    
    print(f"\nEvaluation completed! Processed {len(all_results)} datasets successfully.")
    return all_results


if __name__ == "__main__":
    print("Starting Baseline EM Mixture Evaluation Script")
    print("=" * 80)
    
    try:
        results = comprehensive_evaluation(n_iterations=10)
        if results:
            print("\nEvaluation completed successfully!")
        else:
            print("\nEvaluation failed - no results obtained.")
    except Exception as e:
        print(f"\nCritical error in main execution: {e}")
        traceback.print_exc()