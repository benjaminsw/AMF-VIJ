"""
  CHANGELOG v2.1.0:
  - Combined train+val data for both Stage 1 and Stage 2 training
  - Modified train_mixture_weights_moving_average to use single data parameter
  - Test data reserved exclusively for final evaluation and visualization
  """


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from amf_vi.flows.realnvp import RealNVPFlow
from amf_vi.flows.maf import MAFFlow
#from amf_vi.flows.iaf import IAFFlow
#from amf_vi.flows.gaussianization import GaussianizationFlow
#from amf_vi.flows.naf import NAFFlowSimplified
#from amf_vi.flows.glow import GlowFlow
#from amf_vi.flows.nice import NICEFlow
#from amf_vi.flows.tan import TANFlow
from amf_vi.flows.rbig import RBIGFlow
#from data.data_generator import generate_data
from data.data_cache import get_split_data
import numpy as np
import os
import pickle

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

class SequentialAMFVI(nn.Module):
    def __init__(self, dim=2, flow_types=None, weight_update_method='moving_average'):
        super().__init__()
        self.dim = dim
        
        if flow_types is None:
            flow_types = ['realnvp', 'maf', 'gaussianization']
        
        # Create flows - EXPANDED TO INCLUDE ALL AVAILABLE FLOWS
        self.flows = nn.ModuleList()
        for flow_type in flow_types:
            if flow_type == 'realnvp':
                self.flows.append(RealNVPFlow(dim, n_layers=8))
            elif flow_type == 'maf':
                self.flows.append(MAFFlow(dim, n_layers=8))
            elif flow_type == 'iaf':
                self.flows.append(IAFFlow(dim, n_layers=1))
            elif flow_type == 'gaussianization':
                self.flows.append(GaussianizationFlow(dim, n_layers=4, n_anchors=20))
            elif flow_type == 'rbig':  # NEW RBIG OPTION
                self.flows.append(RBIGFlow(dim, n_layers=4, n_bins=50))
            elif flow_type == 'naf':
                self.flows.append(NAFFlowSimplified(dim, n_layers=4, hidden_dim=32))
            elif flow_type == 'glow':  # NEW GLOW OPTION
                self.flows.append(GlowFlow(dim, n_steps=4, hidden_dim=32))
            elif flow_type == 'nice':  # NEW NICE OPTION
                self.flows.append(NICEFlow(dim, n_layers=4, hidden_dim=32))
            elif flow_type == 'spline':  # NEW SPLINE OPTION
                self.flows.append(SplineFlow(dim, n_layers=4, num_bins=8, hidden_dim=32))
            elif flow_type == 'tan':  # NEW TAN OPTION
                self.flows.append(TANFlow(dim, n_layers=4, hidden_dim=32, use_linear=True))
            else:
                raise ValueError(f"Unknown flow type: {flow_type}. Available types: "
                               f"['realnvp', 'maf', 'iaf', 'gaussianization', 'rbig', 'naf', 'glow', 'nice', 'spline', 'tan']")
        
        # Initialize weights as parameters (not log_weights for moving average)
        self.weights = nn.Parameter(torch.ones(len(self.flows)) / len(self.flows))
        self.weight_update_method = weight_update_method
        
        # Moving average parameters
        self.alpha = 0.9  # Moving average decay factor
        self.weight_lr = 0.01
        self.weight_history = []
        
        # Track if flows are trained
        self.flows_trained = False
        self.weights_trained = False

    def safe_log_prob_extraction(self, log_prob_tensor):
        """Extract log prob with NaN handling - Phase 1 fix"""
        mean_log_prob = log_prob_tensor.mean().item()
        
        if torch.isnan(torch.tensor(mean_log_prob)) or torch.isinf(torch.tensor(mean_log_prob)):
            # Fallback: return very low but finite log probability
            return -100.0  # Equivalent to very low probability
        return mean_log_prob
    
    def train_flows_independently(self, data, epochs=1000, lr=1e-4):
        """Stage 1: Train each flow independently."""
        print("🔄 Stage 1: Training flows independently...")
        
        flow_losses = []
        
        for i, flow in enumerate(self.flows):
            print(f"  Training flow {i+1}/{len(self.flows)}: {flow.__class__.__name__}")
            
            # Check if flow has trainable parameters
            params = list(flow.parameters())
            if len(params) == 0:
                print(f"    {flow.__class__.__name__} uses non-parametric fitting instead of gradient optimization")
                
                # Handle non-parametric flows (like RBIG)
                if hasattr(flow, 'fit_to_data'):
                    print("    Using fit_to_data() method...")
                    flow.fit_to_data(data, validate_reconstruction=False)  # Skip validation for speed
                    
                    # Compute actual loss trajectory for visualization consistency
                    losses = []
                    with torch.no_grad():
                        # Sample loss values to create a realistic trajectory
                        for epoch in range(0, epochs, max(1, epochs//20)):  # Sample 20 points
                            try:
                                log_prob = flow.log_prob(data)
                                loss = -log_prob.mean().item()
                                losses.append(loss)
                            except Exception as e:
                                print(f"      Warning: Could not compute loss at epoch {epoch}: {e}")
                                losses.append(float('inf'))
                    
                    # Interpolate to full epoch count for consistency with other flows
                    if len(losses) > 1:
                        import numpy as np
                        epoch_points = list(range(0, epochs, max(1, epochs//20)))
                        full_losses = np.interp(range(epochs), epoch_points, losses)
                        losses = full_losses.tolist()
                    else:
                        # Fallback: constant loss
                        final_loss = losses[0] if losses else 0.0
                        losses = [final_loss] * epochs
                        
                    print(f"    Non-parametric fitting completed. Final loss: {losses[-1]:.4f}")
                    
                else:
                    print(f"    Warning: {flow.__class__.__name__} has no fit_to_data method")
                    losses = [float('nan')] * epochs
                
                flow_losses.append(losses)
                continue
            
            # Standard gradient-based training for parametric flows
            print(f"    Using gradient-based optimization ({len(params)} parameter groups)")
            optimizer = optim.Adam(params, lr=lr)
            losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                try:
                    # Individual flow loss (negative log-likelihood)
                    log_prob = flow.log_prob(data)
                    loss = -log_prob.mean()
                    
                    # Check if loss requires grad
                    if loss.requires_grad:
                        loss.backward()
                        optimizer.step()
                    else:
                        print(f"    Warning: Loss doesn't require grad at epoch {epoch}")
                    
                except RuntimeError as e:
                    if "does not require grad" in str(e):
                        print(f"    Skipping gradient step at epoch {epoch}: {e}")
                        # Create a dummy loss for this step
                        dummy_loss = torch.tensor(float('nan'), requires_grad=True)
                        losses.append(dummy_loss.item())
                        continue
                    else:
                        raise e
                
                losses.append(loss.item())
                
                if epoch % 50 == 0:
                    print(f"    Epoch {epoch}: Loss = {loss.item():.4f}")
            
            flow_losses.append(losses)
            print(f"    Final loss: {losses[-1]:.4f}")
        
        self.flows_trained = True
        return flow_losses
    
    #def train_mixture_weights_moving_average(self, data, epochs=500):
    #def train_mixture_weights_moving_average(self, train_data, val_data, epochs=500):
    def train_mixture_weights_moving_average(self, data, epochs=500):
        """Stage 2: Learn weights using Moving Average of Likelihoods."""
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        print("🔄 Stage 2: Learning mixture weights (Moving Average)...")
        
        weight_losses = []
        
        for epoch in range(epochs):
            '''
            # Generate fresh data for each epoch to avoid bias
            dataset_name = getattr(self, 'dataset_name', 'multimodal')  # Default fallback
            fresh_data = generate_data(dataset_name, n_samples=1000)
            fresh_data = fresh_data.to(data.device)
            '''
            
            '''
            # Sample batch from validation data
            batch_size = min(2000, len(val_data))
            indices = torch.randperm(len(val_data), device=val_data.device)[:batch_size]
            val_batch = val_data[indices]
            '''
            
            
            # Sample batch from combined train+val data
            batch_size = min(2000, len(data))
            indices = torch.randperm(len(data), device=data.device)[:batch_size]
            data_batch = data[indices]

            
            # Get flow log probabilities on fresh data
            flow_log_probs = []
            for flow in self.flows:
                flow.eval()
                with torch.no_grad():
                    #log_prob = flow.log_prob(fresh_data)
                    #log_prob = flow.log_prob(val_batch)
                    log_prob = flow.log_prob(data_batch)
                    safe_log_prob = self.safe_log_prob_extraction(log_prob)
                    flow_log_probs.append(safe_log_prob)  # Use safe extraction
            
            # Convert to likelihoods and normalize (softmax)
            # flow_log_probs_tensor = torch.tensor(flow_log_probs)
            #flow_log_probs_tensor = torch.tensor(flow_log_probs, device=data.device)
            #flow_log_probs_tensor = torch.tensor(flow_log_probs, device=val_data.device)
            flow_log_probs_tensor = torch.tensor(flow_log_probs, device=data.device)
            normalized_likelihoods = F.softmax(flow_log_probs_tensor, dim=0)
            
            # Moving average update: weight_i = α * old_weight_i + (1-α) * normalized_likelihood_i
            with torch.no_grad():
                old_weights = self.weights.data.clone()
                new_weights = self.alpha * old_weights + (1 - self.alpha) * normalized_likelihoods
                self.weights.data = new_weights
            
            # Compute mixture log probability for loss tracking (use original data for consistency)
            #batch_weights = self.weights.unsqueeze(0).expand(data.size(0), -1)
            #batch_weights = self.weights.unsqueeze(0).expand(train_data.size(0), -1)
            batch_weights = self.weights.unsqueeze(0).expand(data.size(0), -1)
            flow_predictions = []
            for flow in self.flows:
                flow.eval()
                with torch.no_grad():
                    #log_prob = flow.log_prob(data)
                    #log_prob = flow.log_prob(train_data)
                    log_prob = flow.log_prob(data)
                    # Handle potential NaN in individual predictions
                    if torch.any(torch.isnan(log_prob)) or torch.any(torch.isinf(log_prob)):
                        log_prob = torch.full_like(log_prob, -100.0)  # Replace NaN/inf with low prob
                    flow_predictions.append(log_prob.unsqueeze(1))
            
            flow_predictions = torch.cat(flow_predictions, dim=1)
            weighted_log_probs = flow_predictions + torch.log(batch_weights + 1e-8)
            mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
            loss = -mixture_log_prob.mean()
            
            weight_losses.append(loss.item())
            
            if epoch % 100 == 0:
                current_weights = self.weights.detach().cpu().numpy()
                print(f"    Epoch {epoch}: Loss = {loss.item():.4f}, Weights = {current_weights}")
        
        final_weights = self.weights.detach().cpu().numpy()
        print(f"    Final weights: {final_weights}")
        
        self.weights_trained = True
        self.weight_history = weight_losses
        return weight_losses
    
    def get_flow_predictions(self, x):
        """Get predictions from all pre-trained flows."""
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        flow_log_probs = []
        for flow in self.flows:
            flow.eval()
            with torch.no_grad():
                log_prob = flow.log_prob(x)
                # Handle potential NaN in predictions
                if torch.any(torch.isnan(log_prob)) or torch.any(torch.isinf(log_prob)):
                    log_prob = torch.full_like(log_prob, -100.0)  # Replace NaN/inf with low prob
                flow_log_probs.append(log_prob.unsqueeze(1))
        
        return torch.cat(flow_log_probs, dim=1)  # [batch, n_flows]
    
    def forward(self, x):
        """Forward pass with learned or uniform weights."""
        if not self.flows_trained:
            raise RuntimeError("Model must be trained first!")
        
        # Get flow predictions
        flow_predictions = self.get_flow_predictions(x)
        
        # Use learned weights if available, otherwise uniform
        if self.weights_trained:
            weights = self.weights
        else:
            weights = torch.ones(len(self.flows), device=x.device) / len(self.flows)
        
        batch_size = x.size(0)
        batch_weights = weights.unsqueeze(0).expand(batch_size, -1)
        
        # Compute mixture log probability
        weighted_log_probs = flow_predictions + torch.log(batch_weights + 1e-8)
        mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
        
        return {
            'log_prob': flow_predictions,
            'weights': batch_weights,
            'mixture_log_prob': mixture_log_prob,
        }
    
    def log_prob(self, x):
        """Compute log probability of data under the mixture model."""
        return self.forward(x)['mixture_log_prob']
    
    def sample(self, n_samples):
        """Sample from the mixture with learned or uniform weights."""
        device = next(self.parameters()).device
        
        # Get current weights
        if self.weights_trained:
            weights = self.weights.detach().cpu().numpy()
        else:
            weights = np.ones(len(self.flows)) / len(self.flows)
        
        # Handle NaN weights - use uniform if any NaN detected
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            print("⚠️  Warning: NaN weights detected, using uniform weights for sampling")
            weights = np.ones(len(self.flows)) / len(self.flows)
        
        # Ensure weights are normalized and valid
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(self.flows)) / len(self.flows)
        
        # Sample according to learned weights
        flow_indices = np.random.choice(len(self.flows), size=n_samples, p=weights)
        unique_indices, counts = np.unique(flow_indices, return_counts=True)
        
        all_samples = []
        
        for idx, count in zip(unique_indices, counts):
            flow = self.flows[idx]
            flow.eval()
            with torch.no_grad():
                samples = flow.sample(count)
                all_samples.append(samples)
        
        return torch.cat(all_samples, dim=0)

#def train_sequential_amf_vi(dataset_name='multimodal', flow_types=None, show_plots=True, save_plots=False):
def train_sequential_amf_vi(dataset_name='multimodal', flow_types=None, show_plots=True, save_plots=False, n_samples=100_000):
    """Train sequential AMF-VI with learnable weights."""
    
    print(f"🚀 Sequential AMF-VI Experiment on {dataset_name}")
    print("=" * 60)
    
    # Use provided flow_types or default
    if flow_types is None:
        flow_types = ['realnvp', 'maf', 'gaussianization']
    
    print(f"Using flows: {flow_types}")
    
    '''
    # Generate data
    data = generate_data(dataset_name, n_samples=1000)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    '''
    
    split_data = get_split_data(dataset_name, n_samples=n_samples)
    train_data = split_data['train']
    val_data = split_data['val']
    test_data = split_data['test']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device
    '''
    
    # ✅ Combine train+val for training flows and mixture
    train_val_data = torch.cat([train_data, val_data], dim=0).to(device)
    test_data = test_data.to(device)
    
    # Create sequential model with specified flow types
    model = SequentialAMFVI(dim=2, flow_types=flow_types, weight_update_method='moving_average')
    model.dataset_name = dataset_name  # Set dataset name for fresh data generation
    model.results_dir = os.path.join("./", "results")
    model = model.to(device)
    train_epochs = 3000
    ma_epochs = 1000
    
    # Stage 1: Train flows independently
    #flow_losses = model.train_flows_independently(data, epochs=train_epochs, lr=1e-3)
    '''
    flow_losses = model.train_flows_independently(train_data, epochs=train_epochs, lr=5e-5)
    
    # Stage 2: Learn mixture weights using moving average
    #weight_losses = model.train_mixture_weights_moving_average(data, epochs=train_epochs)
    
    weight_losses = model.train_mixture_weights_moving_average(
      train_data=train_data,
      val_data=val_data,
      epochs=ma_epochs 
    )
    '''
    flow_losses = model.train_flows_independently(train_val_data, epochs=train_epochs, lr=1e-3)

    # Stage 2: Learn mixture weights using moving average
    weight_losses = model.train_mixture_weights_moving_average(
    data=train_val_data,
    epochs=ma_epochs 
    )
    
    # Evaluation and visualization
    print("\n🎨 Generating visualizations...")
    
    model.eval()
    with torch.no_grad():
        # Generate samples
        model_samples = model.sample(1000)
        
        # Individual flow samples
        flow_samples = {}
        for i, flow_type in enumerate(flow_types):
            flow_samples[flow_type] = model.flows[i].sample(1000)
        
        # Create visualization with dynamic subplot arrangement
        n_flows = len(flow_types)
        n_cols = max(3, n_flows)  # Ensure enough columns
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))

        # Plot target data
        #data_np = data.cpu().numpy()
        data_np = test_data.cpu().numpy()
        axes[0, 0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, c='blue', s=20)
        axes[0, 0].set_title('Target Data')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot sequential model samples
        model_np = model_samples.cpu().numpy()
        axes[0, 1].scatter(model_np[:, 0], model_np[:, 1], alpha=0.6, c='red', s=20)
        axes[0, 1].set_title('Sequential AMF-VI Samples')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot training losses in axes[0, 3]
        # Plot training losses in axes[0, 3]
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
        if flow_losses:
            for i, (flow_type, losses) in enumerate(zip(flow_types, flow_losses)):
                axes[0, 2].plot(losses, label=flow_type.upper(), 
                               color=colors[i % len(colors)], linewidth=1, alpha=0.7)

        # Plot weight learning loss on same scale
        if weight_losses:
            axes[0, 2].plot(weight_losses, label='Weight Learning', color='red', linewidth=2)

        axes[0, 2].set_title('Training Losses')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()

        # Plot individual flows from the second row on
        for i, (flow_type, samples) in enumerate(flow_samples.items()):
            col = i
            row = 1
            
            if col < n_cols:
                samples_np = samples.cpu().numpy()
                axes[row, col].scatter(samples_np[:, 0], samples_np[:, 1], 
                                     alpha=0.6, c=colors[i % len(colors)], s=20)
                axes[row, col].set_title(f'{flow_type.upper()} Flow')
                axes[row, col].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(4, n_cols):  # Hide unused top row subplots after position 3
            axes[0, i].set_visible(False)
        for i in range(len(flow_samples), n_cols):  # Hide unused bottom row subplots
            axes[1, i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f'Sequential AMF-VI Results - {dataset_name.title()}', fontsize=16)
        
        # Save plot if requested
        if save_plots:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'sequential_amf_vi_results_{dataset_name}.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {plot_path}")
        
        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        # Print analysis
        print("\n📊 Analysis:")
        #print(f"Target data mean: {data.mean(dim=0).cpu().numpy()}")
        print(f"Target data mean: {test_data.mean(dim=0).cpu().numpy()}")
        print(f"Sequential model mean: {model_samples.mean(dim=0).cpu().numpy()}")
        #print(f"Target data std: {data.std(dim=0).cpu().numpy()}")
        print(f"Target data std: {test_data.std(dim=0).cpu().numpy()}")
        print(f"Sequential model std: {model_samples.std(dim=0).cpu().numpy()}")
        
        # Check flow diversity and learned weights
        print("\n🔍 Flow Specialization Analysis:")
        learned_weights = model.weights.detach().cpu().numpy()
        for i, (flow_type, samples) in enumerate(flow_samples.items()):
            mean = samples.mean(dim=0).cpu().numpy()
            std = samples.std(dim=0).cpu().numpy()
            weight = learned_weights[i]
            print(f"{flow_type.upper()}: Weight={weight:.3f}, Mean=[{mean[0]:.2f}, {mean[1]:.2f}], Std=[{std[0]:.2f}, {std[1]:.2f}]")
        
        # Model complexity analysis
        print("\n🏗️ Model Architecture:")
        total_params = 0
        for i, flow in enumerate(model.flows):
            n_params = sum(p.numel() for p in flow.parameters())
            total_params += n_params
            print(f"{flow_types[i].upper()}: {n_params:,} parameters")
        print(f"Total parameters: {total_params:,}")
        print(f"Weight parameters: {model.weights.numel()}")
    
    # Save trained model
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model, 
            'flow_losses': flow_losses, 
            'weight_losses': weight_losses,
            'dataset': dataset_name,
            'metadata': split_data['metadata'] # Save metadata for reproducibility
        }, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, flow_losses, weight_losses

# Example usage in __main__ section:
if __name__ == "__main__":
    # Run on 7 datasets with configurable flow types
    datasets = [
        'banana',
        'x_shape',
        'bimodal_shared',
        'bimodal_different',
        'multimodal',
        'two_moons',
        'rings',
        "BLR",
        "BPR",
        "Weibull",
        "multimodal-5",
    ]
    
    # Configure flow types here for consistency
    # Available options: 'realnvp', 'maf', 'iaf', 'gaussianization', 'naf', 'glow', 'nice', 'spline', 'tan'
    flow_types = ['realnvp', 'maf', 'rbig']  # You can change this to any combination
    
    print(f"🚀 Running experiments with flows: {flow_types}")
    print(f"📊 Available flow types: ['realnvp', 'maf', 'iaf', 'gaussianization', 'rbig', 'naf', 'glow', 'nice', 'spline', 'tan']")
    
    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training {len(flow_types)}-flow model on {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            model, flow_losses, weight_losses = train_sequential_amf_vi(
                dataset_name=dataset_name,
                flow_types=flow_types,
                show_plots=False, 
                save_plots=True,
                n_samples=1_000_000  # ✅ Set 1M samples here
            )
            
            print(f"✅ Completed {len(flow_types)}-flow model on {dataset_name}")
            
        except Exception as e:
            print(f"❌ Failed on {dataset_name}: {e}")
            continue
