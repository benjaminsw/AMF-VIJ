"""
EM-style training script for Mixture of Normalizing Flows.
Trains mixture models on various datasets and saves to pickle files.
Updated with k-means initialization, minimum weight constraint, and stability improvements.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import pickle
import sys
import matplotlib.pyplot as plt
import traceback
from sklearn.cluster import KMeans

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import generate_data

# Import model classes from separate module
from em_mixture_models import MixtureOfFlows

# Set seed for reproducibility
torch.manual_seed(2025)
np.random.seed(2025)


def visualize_results(data, em_samples, em_losses, dataset_name, save_path, em_model):
    """Create visualization for EM mixture model."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    data_np = data.detach().cpu().numpy()
    em_np = em_samples.cpu().numpy()
    
    axes[0, 0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, c='blue', s=20)
    axes[0, 0].set_title('Target Data')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(em_np[:, 0], em_np[:, 1], alpha=0.6, c='purple', s=20)
    axes[0, 1].set_title('EM Mixture Samples')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(em_losses, label='EM Mixture', color='purple', alpha=0.7)
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Negative Log-Likelihood')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    stats_text = f"Data Mean: [{data.mean(0)[0]:.2f}, {data.mean(0)[1]:.2f}]\n"
    stats_text += f"Data Std: [{data.std(0)[0]:.2f}, {data.std(0)[1]:.2f}]\n\n"
    stats_text += f"EM Mean: [{em_samples.mean(0)[0]:.2f}, {em_samples.mean(0)[1]:.2f}]\n"
    stats_text += f"EM Std: [{em_samples.std(0)[0]:.2f}, {em_samples.std(0)[1]:.2f}]\n\n"
    
    total_params = sum(sum(p.numel() for p in flow.parameters()) for flow in em_model.flows)
    stats_text += f"Total Parameters: {total_params:,}\n"
    stats_text += f"Components: {em_model.K}\n"
    stats_text += f"Final Weights: {em_model.mixture_weights.cpu().numpy()}\n"
    stats_text += f"Final Loss: {em_losses[-1]:.4f}"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistics & Model Info')
    
    plt.suptitle(f'EM Mixture of Flows - {dataset_name.title()}', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Visualization saved to {save_path}")
    plt.close()


def initialize_components_with_kmeans(data, n_components, min_weight):
    """
    Initialize mixture components using k-means with stability checks.
    Ensures balanced initialization to prevent unstable regions.
    """
    data_np = data.cpu().numpy()
    n_samples = len(data_np)
    
    # Run k-means multiple times to find best initialization
    best_kmeans = None
    best_inertia = float('inf')
    
    print("\n  Running k-means with multiple initializations...")
    for trial in range(5):
        kmeans = KMeans(n_clusters=n_components, random_state=2025+trial, n_init=10)
        kmeans.fit(data_np)
        
        # Check if any cluster is too small
        cluster_sizes = np.bincount(kmeans.labels_, minlength=n_components)
        min_cluster_size = cluster_sizes.min()
        min_cluster_pct = min_cluster_size / n_samples
        
        print(f"    Trial {trial+1}: Inertia={kmeans.inertia_:.2f}, "
              f"Min cluster size={min_cluster_size} ({min_cluster_pct*100:.1f}%)")
        
        # Prefer initialization where all clusters have reasonable size
        if min_cluster_pct >= min_weight * 0.5:  # At least half of min_weight
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_kmeans = kmeans
    
    # Fallback if no good initialization found
    if best_kmeans is None:
        print("    Warning: No balanced k-means initialization found, using last trial")
        best_kmeans = kmeans
    
    labels = best_kmeans.labels_
    
    # Calculate initial weights with minimum constraint
    initial_weights = np.bincount(labels, minlength=n_components) / n_samples
    initial_weights = np.maximum(initial_weights, min_weight)
    initial_weights = initial_weights / initial_weights.sum()
    
    print(f"\n  Best k-means initialization:")
    print(f"    Cluster sizes: {np.bincount(labels, minlength=n_components)}")
    print(f"    Initial weights (with min constraint): {initial_weights}")
    print(f"    Cluster centers:\n{best_kmeans.cluster_centers_}")
    
    return initial_weights, best_kmeans.cluster_centers_, labels


if __name__ == "__main__":
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 
            'multimodal', 'two_moons', 'rings', 'BLR', 'BPR', 'Weibull', 
            'multimodal-5']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results_dir = os.path.join(os.path.dirname(__file__), 'baseline_results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    error_log_path = os.path.join(results_dir, 'em_mixture_errors.txt')
    error_log = open(error_log_path, 'w')
    
    # Stability hyperparameters
    min_weight = 0.005 # 0.01  # Each component gets at least 5% of data
    learning_rate = 1e-4 # 5e-5  # Reduced from 1e-4 for stability
    scale_clip = 5.0  # 2.0 # 3.0  # Clamp scale outputs to [-10, 10]
    gradient_clip = 5.0  # Increased from 1.0 to 5.0
    l2_reg = 1e-5 # 1e-4 #1e-5  # L2 regularization on scale networks
    
    print(f"\nStability Configuration:")
    print(f"  Minimum weight constraint: {min_weight} ({min_weight*100}%)")
    print(f"  Learning rate: {learning_rate} (reduced for stability)")
    print(f"  Scale clipping: ±{scale_clip}")
    print(f"  Gradient clipping: {gradient_clip}")
    print(f"  L2 regularization: {l2_reg}")
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training EM Mixture on {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            data = generate_data(dataset_name, n_samples=5000)  # from 1000
            data = data.to(device)
            print(f"Generated {len(data)} samples")
            
            dataset = TensorDataset(data)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
            
            # Create model with stability improvements
            em_model = MixtureOfFlows(
                n_components=5 , #3
                dim=2,
                n_flow_layers=6, #4
                learning_rate=learning_rate,
                device=device,
                min_weight=min_weight,
                scale_clip=scale_clip,
                gradient_clip=gradient_clip,
                l2_reg=l2_reg
            )
            
            # IMPROVED K-MEANS INITIALIZATION with stability checks
            print("\nInitializing components with improved k-means clustering...")
            initial_weights, cluster_centers, labels = initialize_components_with_kmeans(
                data, em_model.K, min_weight
            )
            
            em_model.mixture_weights = torch.tensor(
                initial_weights, 
                dtype=torch.float32, 
                device=device
            )
            
            total_params = sum(sum(p.numel() for p in flow.parameters()) for flow in em_model.flows)
            print(f"\nEM Mixture total parameters: {total_params:,}")
            
            print(f"\nTraining EM Mixture on {dataset_name}...")
            em_model.fit(dataloader, n_epochs=5000, verbose=True) # n_epochs=5000 
            
            em_model_eval = em_model
            for flow in em_model_eval.flows:
                flow.eval()
            
            with torch.no_grad():
                em_samples = em_model.sample(1000)
            
            # Check for extreme values in samples
            max_abs_value = em_samples.abs().max().item()
            if max_abs_value > 1000:
                print(f"\n⚠️ WARNING: Generated samples contain extreme values (max={max_abs_value:.2f})")
                print("  Model may still have stability issues despite safeguards.")
            else:
                print(f"\n✓ Sample quality check passed (max abs value: {max_abs_value:.2f})")
            
            em_path = os.path.join(results_dir, f'em_mixture_{dataset_name}.pkl')
            
            with open(em_path, 'wb') as f:
                pickle.dump({
                    'model': em_model,
                    'losses': em_model.log_likelihood_history,
                    'weight_history': em_model.weight_history,
                    'dataset': dataset_name,
                    'model_type': 'EM_Mixture_Stable',
                    'initial_weights': initial_weights,
                    'cluster_centers': cluster_centers,
                    'min_weight': min_weight,
                    'hyperparameters': {
                        'learning_rate': learning_rate,
                        'scale_clip': scale_clip,
                        'gradient_clip': gradient_clip,
                        'l2_reg': l2_reg
                    }
                }, f)
            print(f"EM Mixture model saved to {em_path}")
            
            viz_path = os.path.join(results_dir, f'em_mixture_{dataset_name}.png')
            visualize_results(
                data, em_samples, 
                em_model.log_likelihood_history, 
                dataset_name, viz_path,
                em_model
            )
            
            print(f"\nCompleted {dataset_name}")
            print(f"Final mixture weights: {em_model.mixture_weights.cpu().numpy()}")
            
        except Exception as e:
            error_msg = f"\nFailed on {dataset_name}:\n"
            error_msg += f"Error: {str(e)}\n"
            error_msg += f"Traceback:\n{traceback.format_exc()}\n"
            error_msg += "="*60 + "\n"
            
            print(error_msg)
            error_log.write(error_msg)
            error_log.flush()
            
            error_log.close()
            print(f"\nError encountered! Details saved to: {error_log_path}")
            print("Programme halted.")
            sys.exit(1)
    
    error_log.close()
    
    if os.path.getsize(error_log_path) == 0:
        os.remove(error_log_path)
        print("\nNo errors encountered - error log file not created.")
    
    print("\n" + "="*60)
    print("All EM Mixture training completed successfully!")
    print(f"Results saved to: {results_dir}")
    print("="*60)
