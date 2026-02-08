"""
Evaluation script for baseline NICE and Residual Flow models.
Computes metrics over 10 iterations and saves results to CSV.
"""

import torch
import numpy as np
import os
import sys
import pickle
import csv
from statistics import mean, stdev
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import generate_data

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


def compute_single_iteration_metrics(target_samples, flow_model, dataset_name):
    """Compute all metrics for a single iteration"""
    metrics = {}
    
    try:
        # Generate samples for this iteration
        with torch.no_grad():
            target_sample_count = target_samples.shape[0]
            generated_samples = flow_model.sample(target_sample_count)
            print(f"        Target: {target_sample_count} samples, Generated: {generated_samples.shape[0]} samples")
        
        # 1. NLL (Negative Log-Likelihood)
        try:
            #nll = compute_cross_entropy_surrogate(target_samples, flow_model)
            nll = compute_nll_with_gradients(target_samples, flow_model)
            metrics['nll'] = nll
        except Exception as e:
            print(f"        Error computing NLL: {e}")
            metrics['nll'] = None
        
        # 2. KL Divergence
        try:
            kl_div = compute_kl_divergence_metric(target_samples, flow_model, dataset_name)
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
        return None
    
    return metrics


def compute_metrics_over_iterations(target_samples, flow_model, dataset_name, n_iterations=10):
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
            metrics = compute_single_iteration_metrics(target_samples, flow_model, dataset_name)
            if metrics is not None:
                for key in all_metrics.keys():
                    if metrics.get(key) is not None:
                        all_metrics[key].append(metrics[key])
        except Exception as e:
            print(f"        Error in iteration {iteration + 1}: {e}")
            continue
    
    # Calculate mean and std for each metric
    summary_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            summary_metrics[f'{metric_name}_mean'] = mean(values)
            summary_metrics[f'{metric_name}_std'] = stdev(values) if len(values) > 1 else 0.0
            summary_metrics[f'{metric_name}_count'] = len(values)
        else:
            summary_metrics[f'{metric_name}_mean'] = None
            summary_metrics[f'{metric_name}_std'] = None
            summary_metrics[f'{metric_name}_count'] = 0
    
    return summary_metrics


def evaluate_baseline_model(dataset_name, model_type, n_iterations=10):
    """Evaluate a single baseline model"""
    
    print(f"\n  Evaluating {model_type.upper()} on {dataset_name}")
    
    try:
        # Load model
        baseline_dir = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.join(baseline_dir, 'baseline_results')
        model_path = os.path.join(results_dir, f'{model_type}_{dataset_name}.pkl')
        
        if not os.path.exists(model_path):
            print(f"    Model file not found: {model_path}")
            return None
        
        print(f"    Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
            model = saved_data['model']
        
        # Generate test data
        test_data = generate_data(dataset_name, n_samples=2000)
        
        # Verify sample count
        if test_data.shape[0] != 2000:
            print(f"    Warning: Expected 2000 samples, got {test_data.shape[0]}")
            if test_data.shape[0] > 2000:
                test_data = test_data[:2000]
            elif test_data.shape[0] < 1900:
                print(f"    Error: Not enough samples ({test_data.shape[0]} < 1900)")
                return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_data = test_data.to(device)
        model = model.to(device)
        model.eval()
        
        # Enable gradients for Jacobian computation
        test_data = test_data.requires_grad_(True)
        
        # Compute metrics over iterations
        metrics = compute_metrics_over_iterations(test_data, model, dataset_name, n_iterations)
        
        # Print summary
        print(f"    Results for {model_type.upper()}:")
        for metric in ['nll', 'kl_divergence', 'full_wasserstein', 'gaussian_mmd_unbiased']:
            mean_val = metrics.get(f'{metric}_mean')
            std_val = metrics.get(f'{metric}_std')
            count_val = metrics.get(f'{metric}_count', 0)
            if mean_val is not None:
                print(f"      {metric}: {mean_val:.6f} ± {std_val:.6f} (n={count_val})")
            else:
                print(f"      {metric}: FAILED")
        
        return metrics
        
    except Exception as e:
        print(f"    Error evaluating {model_type}: {e}")
        traceback.print_exc()
        return None


def evaluate_all_baseline_models(n_iterations=10):
    """Evaluate all baseline models and save results to CSV"""
    
    # datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 
                # 'multimodal', 'two_moons', 'rings']
    
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 
            'multimodal', 'two_moons', 'rings', 'BLR', 'BPR', 'Weibull', 
            'multimodal-5']
    
    
    model_types = ['nice', 'resflow']
    
    print(f"Starting Baseline Model Evaluation ({n_iterations} iterations per metric)")
    print(f"Datasets: {datasets}")
    print(f"Models: {model_types}")
    print("=" * 60)
    
    all_results = []
    
    # Evaluate each dataset and model
    for dataset_name in datasets:
        print(f"\nDataset: {dataset_name.upper()}")
        
        for model_type in model_types:
            try:
                metrics = evaluate_baseline_model(dataset_name, model_type, n_iterations)
                
                if metrics is not None:
                    result_row = {
                        'dataset': dataset_name,
                        'model': model_type.upper(),
                        'nll_mean': metrics.get('nll_mean'),
                        'nll_std': metrics.get('nll_std'),
                        'kl_divergence_mean': metrics.get('kl_divergence_mean'),
                        'kl_divergence_std': metrics.get('kl_divergence_std'),
                        'full_wasserstein_mean': metrics.get('full_wasserstein_mean'),
                        'full_wasserstein_std': metrics.get('full_wasserstein_std'),
                        'gaussian_mmd_unbiased_mean': metrics.get('gaussian_mmd_unbiased_mean'),
                        'gaussian_mmd_unbiased_std': metrics.get('gaussian_mmd_unbiased_std'),
                        'gaussian_mmd_biased_mean': metrics.get('gaussian_mmd_biased_mean'),
                        'gaussian_mmd_biased_std': metrics.get('gaussian_mmd_biased_std'),
                        'n_iterations': n_iterations
                    }
                    all_results.append(result_row)
                    
            except Exception as e:
                print(f"  Failed to evaluate {model_type} on {dataset_name}: {e}")
                continue
    
    if not all_results:
        print("\nNo models could be evaluated successfully.")
        return None
    
    # Save results to CSV
    baseline_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(baseline_dir, 'baseline_results')
    os.makedirs(results_dir, exist_ok=True)
    
    csv_filename = f'baseline_evaluation_{n_iterations}_iterations.csv'
    csv_path = os.path.join(results_dir, csv_filename)
    
    try:
        with open(csv_path, 'w', newline='') as f:
            fieldnames = [
                'dataset', 'model',
                'nll_mean', 'nll_std',
                'kl_divergence_mean', 'kl_divergence_std',
                'full_wasserstein_mean', 'full_wasserstein_std',
                'gaussian_mmd_unbiased_mean', 'gaussian_mmd_unbiased_std',
                'gaussian_mmd_biased_mean', 'gaussian_mmd_biased_std',
                'n_iterations'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n{csv_filename} successfully created at {csv_path}")
        print(f"Evaluated {len(all_results)} model-dataset combinations")
        
    except Exception as e:
        print(f"\nError saving CSV: {e}")
        traceback.print_exc()
    
    return all_results

def compute_nll_with_gradients(target_samples, flow_model):
    """Compute NLL with proper gradient handling for ResidualFlow"""
    try:
        # Ensure input has gradients
        if not target_samples.requires_grad:
            target_samples = target_samples.requires_grad_(True)
        
        # Compute log probability directly
        log_prob = flow_model.log_prob(target_samples)
        
        # Return negative log likelihood (mean)
        nll = -log_prob.mean().item()
        return nll
    except Exception as e:
        print(f"        Error in NLL wrapper: {e}")
        return None


if __name__ == "__main__":
    print("Starting Baseline Model Evaluation Script")
    print("=" * 80)
    
    try:
        results = evaluate_all_baseline_models(n_iterations=10)
        if results:
            print("\nBaseline evaluation completed successfully!")
        else:
            print("\nBaseline evaluation failed - no results obtained.")
    except Exception as e:
        print(f"\nCritical error in main execution: {e}")
        traceback.print_exc()
