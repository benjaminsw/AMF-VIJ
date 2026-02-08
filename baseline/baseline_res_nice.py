"""
Baseline training script for NICE and Residual Flow models.
Trains models independently on various datasets and saves to pickle files.
"""

import torch
import torch.optim as optim
import numpy as np
import os
import pickle
import sys
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amf_vi.flows.nice import NICEFlow
from amf_vi.flows.residual_flow import ResidualFlow
from data.data_generator import generate_data

# Set seed for reproducibility
torch.manual_seed(2025)
np.random.seed(2025)

import tempfile
import gc

def train_flow_independently(flow, data, epochs=1000, lr=1e-3, batch_size=128, 
                            flow_name="Flow", patience=200):
    """Train a single flow model independently with batching and early stopping."""
    print(f"Training {flow_name}...")
    
    # Check if flow has trainable parameters
    params = list(flow.parameters())
    if len(params) == 0:
        print(f"  {flow_name} has no trainable parameters!")
        return [float('nan')] * epochs
    
    optimizer = optim.Adam(params, lr=lr)
    losses = []
    
    # Early stopping setup
    best_loss = float('inf')
    no_improve = 0
    n_samples = data.shape[0]
    
    # ===== OPTION 3: START - Create temp file for best model =====
    temp_model_path = tempfile.NamedTemporaryFile(delete=False, suffix='.pt').name
    # ===== OPTION 3: END =====
    
    for epoch in range(epochs):
        flow.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle data
        perm = torch.randperm(n_samples)
        
        for i in range(0, n_samples, batch_size):
            batch_indices = perm[i:i+batch_size]
            batch = data[batch_indices]
            
            optimizer.zero_grad()
            
            try:
                # Compute negative log-likelihood loss
                log_prob = flow.log_prob(batch)
                loss = -log_prob.mean()
                
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
                    optimizer.step()
                else:
                    print(f"  Warning: Loss doesn't require grad at epoch {epoch}")
                
                epoch_loss += loss.item()
                n_batches += 1
            
            except RuntimeError as e:
                print(f"  Error at epoch {epoch}: {e}")
                epoch_loss += float('nan')
                n_batches += 1
                continue
        
        avg_loss = epoch_loss / n_batches if n_batches > 0 else float('nan')
        losses.append(avg_loss)
        
        # ===== OPTION 3: START - Early stopping with disk save =====
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(flow.state_dict(), temp_model_path)  # Save to disk
            no_improve = 0
        else:
            no_improve += 1
        # ===== OPTION 3: END =====
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}, Best = {best_loss:.4f}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # ===== OPTION 3: START - Restore best model from disk =====
    if os.path.exists(temp_model_path):
        flow.load_state_dict(torch.load(temp_model_path))
        os.remove(temp_model_path)  # Clean up temp file
    # ===== OPTION 3: END =====
    
    print(f"  Final loss: {best_loss:.4f}")
    flow.eval()
    return losses

def visualize_results(data, nice_samples, resflow_samples, nice_losses, resflow_losses, 
                     dataset_name, save_path, nice_model, resflow_model):
    """Create visualization comparing both models."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    #data_np = data.cpu().numpy()
    data_np = data.detach().cpu().numpy()
    nice_np = nice_samples.cpu().numpy()
    resflow_np = resflow_samples.cpu().numpy()
    
    # Row 1: Data and samples
    axes[0, 0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, c='blue', s=20)
    axes[0, 0].set_title('Target Data')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(nice_np[:, 0], nice_np[:, 1], alpha=0.6, c='red', s=20)
    axes[0, 1].set_title('NICE Samples')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].scatter(resflow_np[:, 0], resflow_np[:, 1], alpha=0.6, c='green', s=20)
    axes[0, 2].set_title('Residual Flow Samples')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Training curves and statistics
    axes[1, 0].plot(nice_losses, label='NICE', color='red', alpha=0.7)
    axes[1, 0].plot(resflow_losses, label='Residual Flow', color='green', alpha=0.7)
    axes[1, 0].set_title('Training Losses')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Negative Log-Likelihood')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Statistics comparison
    stats_text = f"Data Mean: [{data.mean(0)[0]:.2f}, {data.mean(0)[1]:.2f}]\n"
    stats_text += f"Data Std: [{data.std(0)[0]:.2f}, {data.std(0)[1]:.2f}]\n\n"
    stats_text += f"NICE Mean: [{nice_samples.mean(0)[0]:.2f}, {nice_samples.mean(0)[1]:.2f}]\n"
    stats_text += f"NICE Std: [{nice_samples.std(0)[0]:.2f}, {nice_samples.std(0)[1]:.2f}]\n\n"
    stats_text += f"ResFlow Mean: [{resflow_samples.mean(0)[0]:.2f}, {resflow_samples.mean(0)[1]:.2f}]\n"
    stats_text += f"ResFlow Std: [{resflow_samples.std(0)[0]:.2f}, {resflow_samples.std(0)[1]:.2f}]"
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Statistics Comparison')
    
    # Model parameters
    nice_params = sum(p.numel() for p in nice_model.parameters())
    resflow_params = sum(p.numel() for p in resflow_model.parameters())
    
    param_text = f"NICE Parameters: {nice_params:,}\n"
    param_text += f"ResFlow Parameters: {resflow_params:,}\n\n"
    param_text += f"Final NICE Loss: {nice_losses[-1]:.4f}\n"
    param_text += f"Final ResFlow Loss: {resflow_losses[-1]:.4f}"
    
    axes[1, 2].text(0.1, 0.5, param_text, fontsize=10, verticalalignment='center')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Model Info')
    
    plt.suptitle(f'Baseline Flow Comparison - {dataset_name.title()}', fontsize=16)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Configuration
    # datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 
                # 'multimodal', 'two_moons', 'rings']
    
    datasets = ['banana', 'x_shape', 'bimodal_shared', 'bimodal_different', 
            'multimodal', 'two_moons', 'rings', 'BLR', 'BPR', 'Weibull', 
            'multimodal-5']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create results directory
    results_dir = os.path.join(os.path.dirname(__file__), 'baseline_results')
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # Train on each dataset
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training on {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            # Generate data
            data = generate_data(dataset_name, n_samples=5000)
            data = data.to(device)
            print(f"Generated {len(data)} samples")
            
            # Create models
            # nice_model = NICEFlow(dim=2, n_layers=4, hidden_dim=64).to(device)
            nice_model = NICEFlow(dim=2, n_layers=6, hidden_dim=256).to(device)
            resflow_model = ResidualFlow(
                dim=2, 
                n_blocks=6, 
                hidden_dim=64,
                n_layers_per_block=3,
                activation='elu',
                learnable_prior=True
            ).to(device)
            
            print(f"NICE parameters: {sum(p.numel() for p in nice_model.parameters()):,}")
            print(f"ResFlow parameters: {sum(p.numel() for p in resflow_model.parameters()):,}")
            
            # Train NICE
            nice_losses = train_flow_independently(
                nice_model, data, epochs=1000, lr=5e-3, flow_name="NICE"
            )
            
            # Train Residual Flow
            resflow_losses = train_flow_independently(
                resflow_model, data, epochs=5000, lr=5e-5, flow_name="Residual Flow", patience=50
            )
            
            # Generate samples for visualization
            nice_model.eval()
            resflow_model.eval()
            with torch.no_grad():
                nice_samples = nice_model.sample(1000)
                resflow_samples = resflow_model.sample(1000)
            
            # Save models
            nice_path = os.path.join(results_dir, f'nice_{dataset_name}.pkl')
            resflow_path = os.path.join(results_dir, f'resflow_{dataset_name}.pkl')
            
            with open(nice_path, 'wb') as f:
                pickle.dump({
                    'model': nice_model,
                    'losses': nice_losses,
                    'dataset': dataset_name,
                    'model_type': 'NICE'
                }, f)
            print(f"NICE model saved to {nice_path}")
            
            with open(resflow_path, 'wb') as f:
                pickle.dump({
                    'model': resflow_model,
                    'losses': resflow_losses,
                    'dataset': dataset_name,
                    'model_type': 'ResidualFlow'
                }, f)
            print(f"Residual Flow model saved to {resflow_path}")
            
            # Create visualization
            viz_path = os.path.join(results_dir, f'comparison_{dataset_name}.png')
            visualize_results(
                data, nice_samples, resflow_samples, 
                nice_losses, resflow_losses, 
                dataset_name, viz_path,
                nice_model, resflow_model 
            )
            
            print(f"Completed {dataset_name}")
            
        except Exception as e:
            print(f"Failed on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All baseline training completed!")
    print(f"Results saved to: {results_dir}")
    print("="*60)