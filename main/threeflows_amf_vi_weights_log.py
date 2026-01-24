"""
Version: 2.0.0
AMF-VI Sequential Training with Generate-Once-Cache-Forever data strategy.

CHANGELOG v2.0.0:
- Integrated data caching system for reproducible train/val/test splits
- Modified Stage 2 to use held-out validation data instead of fresh generation
- Removed fallback to fresh data generation (cache-only approach)
- Updated train_mixture_weights_moving_average() signature to accept val_data
- All data now sourced from cached stratified splits
"""

##################################################
# AMF-VI with Cached Data Splits (v2.0.0)
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from amf_vi.flows.realnvp import RealNVPFlow
from amf_vi.flows.maf import MAFFlow
from amf_vi.flows.rbig import RBIGFlow
from data.data_cache import get_split_data  # NEW: Import caching utility
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
            flow_types = ['realnvp', 'maf', 'rbig']
        
        # Create flows
        self.flows = nn.ModuleList()
        for flow_type in flow_types:
            if flow_type == 'realnvp':
                self.flows.append(RealNVPFlow(dim, n_layers=8))
            elif flow_type == 'maf':
                self.flows.append(MAFFlow(dim, n_layers=8))
            elif flow_type == 'rbig':
                self.flows.append(RBIGFlow(dim, n_layers=4, n_bins=50))
            else:
                raise ValueError(f"Unknown flow type: {flow_type}. Available types: "
                               f"['realnvp', 'maf', 'rbig']")
        
        # Initialize weights as parameters
        K = len(self.flows)
        self.weights = nn.Parameter(torch.full((K,), 1.0 / K))
        
        # Hyperparams
        self.alpha = getattr(self, "alpha", 0.9)  # EMA factor
        self.resp_temperature = getattr(self, "resp_temperature", 1.0)
        self.weight_update_method = weight_update_method
        self.weight_lr = 0.01
        self.weight_history = []
        
        # Track training status
        self.flows_trained = False
        self.weights_trained = False

        self.dataset_name = getattr(self, "dataset_name", "unknown_dataset")
        self.results_dir = getattr(self, "results_dir", os.path.join(os.path.dirname(__file__), "..", "results"))

        # Stage-2 histories
        self.weights_history = []
        self.responsibilities_history = []
        self.weight_losses = []

    def safe_log_prob_extraction(self, log_prob_tensor):
        """Extract log prob with NaN handling."""
        mean_log_prob = log_prob_tensor.mean().item()
        
        if torch.isnan(torch.tensor(mean_log_prob)) or torch.isinf(torch.tensor(mean_log_prob)):
            return -100.0  # Fallback for numerical issues
        return mean_log_prob
    
    def train_flows_independently(self, data, epochs=1000, lr=1e-4):
        """Stage 1: Train each flow independently on training data."""
        print("🔄 Stage 1: Training flows independently...")
        
        flow_losses = []
        
        for i, flow in enumerate(self.flows):
            print(f"  Training flow {i+1}/{len(self.flows)}: {flow.__class__.__name__}")
            
            # Check if flow has trainable parameters
            params = list(flow.parameters())
            if len(params) == 0:
                print(f"    {flow.__class__.__name__} uses non-parametric fitting")
                
                # Handle non-parametric flows (like RBIG)
                if hasattr(flow, 'fit_to_data'):
                    print("    Using fit_to_data() method...")
                    flow.fit_to_data(data, validate_reconstruction=False)
                    
                    # Compute loss trajectory for visualization
                    losses = []
                    with torch.no_grad():
                        for epoch in range(0, epochs, max(1, epochs//20)):
                            try:
                                log_prob = flow.log_prob(data)
                                loss = -log_prob.mean().item()
                                losses.append(loss)
                            except Exception as e:
                                print(f"      Warning: Could not compute loss at epoch {epoch}: {e}")
                                losses.append(float('inf'))
                    
                    # Interpolate to full epoch count
                    if len(losses) > 1:
                        epoch_points = list(range(0, epochs, max(1, epochs//20)))
                        full_losses = np.interp(range(epochs), epoch_points, losses)
                        losses = full_losses.tolist()
                    else:
                        final_loss = losses[0] if losses else 0.0
                        losses = [final_loss] * epochs
                        
                    print(f"    Non-parametric fitting completed. Final loss: {losses[-1]:.4f}")
                else:
                    print(f"    Warning: {flow.__class__.__name__} has no fit_to_data method")
                    losses = [float('nan')] * epochs
                
                flow_losses.append(losses)
                continue
            
            # Standard gradient-based training
            print(f"    Using gradient-based optimization ({len(params)} parameter groups)")
            optimizer = optim.Adam(params, lr=lr)
            losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                
                try:
                    log_prob = flow.log_prob(data)
                    loss = -log_prob.mean()
                    
                    if loss.requires_grad:
                        loss.backward()
                        optimizer.step()
                    else:
                        print(f"    Warning: Loss doesn't require grad at epoch {epoch}")
                    
                except RuntimeError as e:
                    if "does not require grad" in str(e):
                        print(f"    Skipping gradient step at epoch {epoch}: {e}")
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

    def train_mixture_weights_moving_average(self, train_data, val_data, epochs=500):
        """
        Stage 2: Learn mixture weights using held-out validation data.
        
        CHANGED v2.0.0: Now uses val_data instead of generating fresh data each epoch.
        
        Args:
            train_data: Original training data (for final loss computation on training set)
            val_data: Held-out validation data (for weight learning via responsibilities)
            epochs: Number of weight update epochs
        """
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")

        print("🔄 Stage 2: Learning mixture weights (Moving Average on Validation Data)...")
        print(f"   Using validation set: {val_data.shape[0]} samples")

        # Reset histories
        self.weights_history = []
        self.responsibilities_history = []
        self.weight_losses = []

        for epoch in range(epochs):
            # === CHANGED: Use validation data instead of fresh generation ===
            # Sample a batch from val_data (shuffle for variability)
            batch_size = min(2000, len(val_data))
            indices = torch.randperm(len(val_data), device=val_data.device)[:batch_size]
            val_batch = val_data[indices]
            
            # Compute average log-probs per expert on validation batch
            flow_log_probs = []
            for flow in self.flows:
                flow.eval()
                with torch.no_grad():
                    log_prob = flow.log_prob(val_batch)
                    safe_log_prob = self.safe_log_prob_extraction(log_prob)
                    flow_log_probs.append(safe_log_prob)

            # Softmax normalisation to get "responsibility target"
            flow_log_probs_tensor = torch.tensor(flow_log_probs, device=val_batch.device, dtype=torch.float32)
            normalized_likelihoods = torch.softmax(flow_log_probs_tensor, dim=0)  # shape [K]

            # EMA update on weights
            with torch.no_grad():
                old_weights = self.weights.data.clone()
                new_weights = self.alpha * old_weights + (1 - self.alpha) * normalized_likelihoods
                self.weights.data = new_weights

            # Compute mixture loss on training data (for monitoring)
            batch_weights = self.weights.unsqueeze(0).expand(train_data.size(0), -1)
            flow_predictions = []
            for flow in self.flows:
                flow.eval()
                with torch.no_grad():
                    log_prob = flow.log_prob(train_data)
                    if torch.any(torch.isnan(log_prob)) or torch.any(torch.isinf(log_prob)):
                        log_prob = torch.full_like(log_prob, -100.0)
                    flow_predictions.append(log_prob.unsqueeze(1))
            flow_predictions = torch.cat(flow_predictions, dim=1)
            weighted_log_probs = flow_predictions + torch.log(batch_weights + 1e-8)
            mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
            loss = -mixture_log_prob.mean()

            # Log histories
            self.weight_losses.append(float(loss.item()))
            self.weights_history.append(self.weights.detach().cpu().numpy().copy())
            self.responsibilities_history.append(normalized_likelihoods.detach().cpu().numpy().copy())

            # Early stopping check
            if epoch > 50:
                weight_change = np.abs(self.weights_history[-1] - self.weights_history[-2]).max()
                recent_losses = self.weight_losses[-10:]
                loss_change = abs(recent_losses[-1] - recent_losses[0])
                if weight_change < 1e-6 and loss_change < 1e-6:
                    print(f"    ✅ Converged at epoch {epoch}")
                    break
        
            if epoch % 5 == 0:
                print(f"    Epoch {epoch:04d} | Loss {loss.item():.4f} | Weights {self.weights.detach().cpu().numpy()}")

        # Final weights
        final_weights = self.weights.detach().cpu().numpy()
        print(f"    Final weights: {final_weights}")

        self.weights_trained = True

        # Save Stage-2 histories
        try:
            os.makedirs(self.results_dir, exist_ok=True)
            dataset_name = getattr(self, 'dataset_name', 'unknown')
            fname = f"{dataset_name}_histories.npz"
            fpath = os.path.join(self.results_dir, fname)

            np.savez(
                fpath,
                weights_history=_to_np(self.weights_history),  # [T, K]
                responsibilities_history=_to_np(self.responsibilities_history),  # [T, K]
                weight_losses=_to_np(self.weight_losses),  # [T]
                final_weights=_to_np(final_weights),  # [K]
                dataset=np.array(dataset_name),
                experts=np.array([type(f).__name__ for f in self.flows], dtype=object),
            )
            print(f"✅ Saved Stage-2 histories to {fpath}")
        except Exception as e:
            print(f"[warn] Could not save Stage-2 histories: {e}")

        return self.weight_losses

    def get_flow_predictions(self, x):
        """Get predictions from all pre-trained flows."""
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        flow_log_probs = []
        for flow in self.flows:
            flow.eval()
            with torch.no_grad():
                log_prob = flow.log_prob(x)
                if torch.any(torch.isnan(log_prob)) or torch.any(torch.isinf(log_prob)):
                    log_prob = torch.full_like(log_prob, -100.0)
                flow_log_probs.append(log_prob.unsqueeze(1))
        
        return torch.cat(flow_log_probs, dim=1)
    
    def forward(self, x):
        """Forward pass with learned or uniform weights."""
        if not self.flows_trained:
            raise RuntimeError("Model must be trained first!")
        
        flow_predictions = self.get_flow_predictions(x)
        
        if self.weights_trained:
            weights = self.weights
        else:
            weights = torch.ones(len(self.flows), device=x.device) / len(self.flows)
        
        batch_size = x.size(0)
        batch_weights = weights.unsqueeze(0).expand(batch_size, -1)
        
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
        
        if self.weights_trained:
            weights = self.weights.detach().cpu().numpy()
        else:
            weights = np.ones(len(self.flows)) / len(self.flows)
        
        # Handle NaN weights
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            print("⚠️  Warning: NaN weights detected, using uniform weights")
            weights = np.ones(len(self.flows)) / len(self.flows)
        
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(self.flows)) / len(self.flows)
        
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


def _to_np(x):
    """Safely convert tensors/lists to numpy array."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    try:
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return np.asarray([])
            if hasattr(x[0], "detach"):
                return np.stack([x_i.detach().cpu().numpy() for x_i in x], axis=0)
            return np.asarray(x)
    except Exception:
        pass
    return np.asarray(x)


def train_sequential_amf_vi(dataset_name='multimodal', flow_types=None, 
                           show_plots=True, save_plots=False, n_samples=100_000):
    """
    Train sequential AMF-VI with cached data splits.
    
    CHANGED v2.0.0: Uses cached train/val/test splits instead of generating fresh data.
    
    Args:
        dataset_name: Dataset name
        flow_types: List of flow types to use
        show_plots: Whether to display plots
        save_plots: Whether to save plots to disk
        n_samples: Total samples to generate (split 60/20/20 train/val/test)
    """
    
    print(f"🚀 Sequential AMF-VI Experiment on {dataset_name}")
    print("=" * 60)
    
    if flow_types is None:
        flow_types = ['realnvp', 'maf', 'rbig']
    
    print(f"Using flows: {flow_types}")
    
    # === CHANGED: Load cached splits instead of generating fresh data ===
    print(f"\n📂 Loading/generating cached data splits (n={n_samples})...")
    split_data = get_split_data(dataset_name, n_samples=n_samples)
    
    train_data = split_data['train']
    val_data = split_data['val']
    test_data = split_data['test']
    
    print(f"✅ Data splits ready:")
    print(f"   Train: {train_data.shape}")
    print(f"   Val:   {val_data.shape}")
    print(f"   Test:  {test_data.shape}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Create model
    model = SequentialAMFVI(dim=2, flow_types=flow_types, weight_update_method='moving_average')
    model.dataset_name = dataset_name
    model.results_dir = os.path.join("./", "results")
    model = model.to(device)
    
    train_epochs = 3000
    ma_epochs = 500
    
    # === Stage 1: Train flows on training data ===
    flow_losses = model.train_flows_independently(train_data, epochs=train_epochs, lr=5e-5)
    
    # === Stage 2: Learn mixture weights using validation data ===
    weight_losses = model.train_mixture_weights_moving_average(
        train_data=train_data,  # For loss monitoring
        val_data=val_data,      # For weight learning
        epochs=ma_epochs
    )

    # === Evaluation and visualization ===
    print("\n🎨 Generating visualizations...")

    try:
        from scipy.stats import gaussian_kde
        _HAVE_SCIPY = True
    except Exception:
        _HAVE_SCIPY = False

    def _kde_grid(xy, n=200, pad=0.10):
        x, y = xy[:, 0], xy[:, 1]
        xmin, xmax = np.percentile(x, [1, 99])
        ymin, ymax = np.percentile(y, [1, 99])
        dx, dy = xmax - xmin, ymax - ymin
        xmin, xmax = xmin - pad * dx, xmax + pad * dx
        ymin, ymax = ymin - pad * dy, ymax + pad * dy
        X, Y = np.meshgrid(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n))
        return X, Y, np.vstack([X.ravel(), Y.ravel()])

    def _plot_density(ax, pts, *, fill=True, cmap='Blues', levels=None, linecolor='k', linewidths=1.0, alpha=0.85):
        if _HAVE_SCIPY and len(pts) > 5:
            X, Y, XY = _kde_grid(pts)
            kde = gaussian_kde(pts.T)
            Z = kde(XY).reshape(X.shape)
            if levels is None:
                qs = [0.50, 0.60, 0.80, 0.90, 0.97]
                levels = np.quantile(Z.ravel(), qs)
            if fill:
                ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=alpha, antialiased=True)
            ax.contour(X, Y, Z, levels=levels, colors=linecolor, linewidths=linewidths, alpha=alpha)
        else:
            ax.hexbin(pts[:, 0], pts[:, 1], gridsize=60, mincnt=1, cmap=cmap, alpha=alpha, linewidths=0)

    def _scatter(ax, pts, color='tab:blue', s=6, alpha=0.25):
        ax.scatter(pts[:, 0], pts[:, 1], s=s, c=color, alpha=alpha, edgecolors='none')

    model.eval()
    with torch.no_grad():
        # Generate samples
        model_samples = model.sample(1000).cpu()

        # Individual flows
        flow_samples = {}
        for i, flow_type in enumerate(flow_types):
            flow_samples[flow_type] = model.flows[i].sample(1000).cpu()

        # Figure layout
        n_flows = len(flow_types)
        n_cols = max(3, n_flows)
        fig, axes = plt.subplots(2, n_cols, figsize=(3.6 * n_cols, 7), sharex='col', sharey='row')
        for ax in axes.ravel():
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.2)

        # Colors
        flow_colors = ['#3A86FF', '#FF006E', '#8338EC', '#FB5607', '#43AA8B', '#6A4C93']
        target_cmap = 'Greys'
        model_cmap = 'Reds'

        # Use test data for visualization (held-out set)
        data_np = test_data.cpu().numpy()
        all_pts = [data_np, model_samples] + [v for v in flow_samples.values()]
        big = np.vstack(all_pts)
        xmin, xmax = np.percentile(big[:, 0], [1, 99])
        ymin, ymax = np.percentile(big[:, 1], [1, 99])
        dx, dy = xmax - xmin, ymax - ymin
        xmin, xmax = xmin - 0.1 * dx, xmax + 0.1 * dx
        ymin, ymax = ymin - 0.1 * dy, ymax + 0.1 * dy

        # (0,0) Target
        ax = axes[0, 0]
        _plot_density(ax, data_np, fill=True, cmap=target_cmap, linecolor='k', linewidths=0.6, alpha=0.85)
        _scatter(ax, data_np, color='black', s=4, alpha=0.15)
        ax.set_title('Target (test data)', fontweight='bold')

        # (0,1) AMF-VI vs target
        ax = axes[0, 1]
        _plot_density(ax, data_np, fill=False, linecolor='0.4', linewidths=0.6, alpha=0.6)
        _plot_density(ax, model_samples, fill=True, cmap=model_cmap, linecolor='darkred', linewidths=0.8, alpha=0.85)
        _scatter(ax, model_samples, color='tab:red', s=4, alpha=0.3)
        ax.set_title('AMF-VI samples + density', fontweight='bold')

        # (0,2) Loss curves
        ax = axes[0, 2]
        colors = ['#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if flow_losses:
            for i, (flow_type, losses) in enumerate(zip(flow_types, flow_losses)):
                ax.plot(losses, label=flow_type.upper(), color=colors[i % len(colors)], linewidth=1.2, alpha=0.9)
        if weight_losses:
            ax.plot(weight_losses, label='Gating/Weights', color='red', linewidth=1.6, alpha=0.9)
        ax.set_title('Training losses', fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend(frameon=False)

        # Second row: each flow vs target
        for i, (flow_type, samples) in enumerate(flow_samples.items()):
            ax = axes[1, i]
            _plot_density(ax, data_np, fill=False, linecolor='0.4', linewidths=0.6, alpha=0.6)
            _plot_density(ax, samples, fill=True, cmap='Blues',
                          linecolor=flow_colors[i % len(flow_colors)], linewidths=0.8, alpha=0.85)
            _scatter(ax, samples, color=flow_colors[i % len(flow_colors)], s=4, alpha=0.12)
            ax.set_title(f'{flow_type.upper()} samples + density')

        # Hide unused slots
        for j in range(3, n_cols):
            axes[0, j].set_visible(False)
        for j in range(len(flow_samples), n_cols):
            axes[1, j].set_visible(False)

        # Apply common limits
        for ax in axes.ravel():
            if ax.get_visible():
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

        plt.tight_layout(rect=[0, 0.02, 1, 0.93])
        plt.suptitle(f'AMF-VI vs Target and Flow Components — {dataset_name.title()}', fontsize=14, fontweight='bold')

        # Save/show
        if save_plots:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'sequential_amf_vi_results_{dataset_name}.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot saved to {plot_path}")

        if show_plots:
            plt.show()
        else:
            plt.close(fig)

        # Print analysis (using test data)
        print("\n📊 Analysis (on test set):")
        print(f"Target data mean: {test_data.mean(dim=0).cpu().numpy()}")
        print(f"Sequential model mean: {model_samples.mean(dim=0).cpu().numpy()}")
        print(f"Target data std: {test_data.std(dim=0).cpu().numpy()}")
        print(f"Sequential model std: {model_samples.std(dim=0).cpu().numpy()}")

        # Flow specialization
        print("\n🔍 Flow Specialization Analysis:")
        learned_weights = model.weights.detach().cpu().numpy()
        for i, (flow_type, samples) in enumerate(flow_samples.items()):
            mean = samples.mean(dim=0).cpu().numpy()
            std = samples.std(dim=0).cpu().numpy()
            weight = learned_weights[i]
            print(f"{flow_type.upper()}: Weight={weight:.3f}, Mean=[{mean[0]:.2f}, {mean[1]:.2f}], Std=[{std[0]:.2f}, {std[1]:.2f}]")
        
        # Model architecture
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
            'metadata': split_data['metadata']  # Save split metadata
        }, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, flow_losses, weight_losses


if __name__ == "__main__":
    print(os.getcwd())
    
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
    
    flow_types = ['realnvp', 'maf', 'rbig']
    
    print(f"🚀 Running experiments with flows: {flow_types}")
    print(f"📦 Using Generate-Once-Cache-Forever data strategy (seed=2025)")
    
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
                n_samples=1_000_000  # Will be split 600k/200k/200k
            )
            
            print(f"✅ Completed {len(flow_types)}-flow model on {dataset_name}")
            
        except Exception as e:
            print(f"❌ Failed on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue