"""
Version: 3.0.0
Abbr: AMFVI-DMAP-TAU

CHANGELOG v3.0.0:
- Replaced EMA weight update with Dirichlet-MAP closed-form solution
- Removed: alpha (EMA momentum) → Added: alpha_dirichlet (Dirichlet prior)
- Weight update: π = (N_k + α_dirichlet - 1) / Σ(N_k + α_dirichlet - 1)
- Updated τ default: 1.1 → 1.2 for slightly flatter responsibilities
- Method renamed: train_mixture_weights_moving_average → train_mixture_weights_dirichlet_map
- Implements proper Dirichlet-MAP update as in paper Eq. (22-27) with EM-style responsibilities

CHANGELOG v2.2.0:
- Added log πk prior term to Stage-2 softmax: softmax((log_qk + log_πk) / τ) (SEMA-PRIOR)
- Implements EM-style posterior weighting per paper Eq. (22): rk = softmax((log_qk + log_πk) / τ)
- log_pi computed from current self.weights.data with clamp(min=1e-8) for numerical stability
- Builds on v2.1.0 (SEMA-TAU); τ scaling retained

CHANGELOG v2.1.0:
- Added temperature τ=1.1 to Stage-2 softmax responsibility estimation (SEMA-TAU)
- τ passed as constructor param (default=1.1 per paper spec)
- All other logic unchanged from OG baseline
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from amf_vi.flows.realnvp import RealNVPFlow
from amf_vi.flows.maf import MAFFlow
from amf_vi.flows.rbig import RBIGFlow
from data.data_cache import get_split_data
import numpy as np
import os
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

class SequentialAMFVI(nn.Module):
    def __init__(self, dim=2, flow_types=None, weight_update_method='dirichlet_map', tau=1.2, alpha_dirichlet=1.5):
        """
        Sequential AMF-VI with Dirichlet-MAP weight learning.
        
        Args:
            dim: Data dimensionality
            flow_types: List of flow architecture names
            weight_update_method: Update method (kept for compatibility, only 'dirichlet_map' supported)
            tau: Temperature for responsibility softmax (default: 1.2)
            alpha_dirichlet: Dirichlet prior concentration (default: 1.5)
        """
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.alpha_dirichlet = alpha_dirichlet
        
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
                raise ValueError(f"Unknown flow type: {flow_type}. Available: ['realnvp', 'maf', 'rbig']")
        
        # Initialize weights uniformly
        self.weights = nn.Parameter(torch.ones(len(self.flows)) / len(self.flows))
        self.weight_update_method = weight_update_method
        
        # Remove EMA params (no longer needed)
        self.weight_history = []
        
        # Track training status
        self.flows_trained = False
        self.weights_trained = False

    def safe_log_prob_extraction(self, log_prob_tensor):
        """Extract log prob with NaN handling."""
        mean_log_prob = log_prob_tensor.mean().item()
        
        if torch.isnan(torch.tensor(mean_log_prob)) or torch.isinf(torch.tensor(mean_log_prob)):
            logger.warning("NaN/Inf log_prob detected, returning -100.0")
            return -100.0
        return mean_log_prob
    
    def train_flows_independently(self, data, epochs=1000, lr=1e-4):
        """Stage 1: Train each flow independently."""
        logger.info("🔄 Stage 1: Training flows independently...")
        
        flow_losses = []
        
        for i, flow in enumerate(self.flows):
            logger.info(f"  Training flow {i+1}/{len(self.flows)}: {flow.__class__.__name__}")
            
            # Check if flow has trainable parameters
            params = list(flow.parameters())
            if len(params) == 0:
                logger.info(f"    {flow.__class__.__name__} uses non-parametric fitting")
                
                # Handle non-parametric flows (like RBIG)
                if hasattr(flow, 'fit_to_data'):
                    logger.info("    Using fit_to_data() method...")
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
                                logger.error(f"      Error computing loss at epoch {epoch}: {e}")
                                losses.append(float('inf'))
                    
                    # Interpolate to full epoch count
                    if len(losses) > 1:
                        epoch_points = list(range(0, epochs, max(1, epochs//20)))
                        full_losses = np.interp(range(epochs), epoch_points, losses)
                        losses = full_losses.tolist()
                    else:
                        final_loss = losses[0] if losses else 0.0
                        losses = [final_loss] * epochs
                        
                    logger.info(f"    Non-parametric fitting completed. Final loss: {losses[-1]:.4f}")
                else:
                    logger.error(f"    Error: {flow.__class__.__name__} has no fit_to_data method")
                    losses = [float('nan')] * epochs
                
                flow_losses.append(losses)
                continue
            
            # Standard gradient-based training
            logger.info(f"    Using gradient-based optimization ({len(params)} parameter groups)")
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
                        logger.warning(f"    Loss doesn't require grad at epoch {epoch}")
                    
                except RuntimeError as e:
                    if "does not require grad" in str(e):
                        logger.warning(f"    Skipping gradient step at epoch {epoch}: {e}")
                        losses.append(float('nan'))
                        continue
                    else:
                        raise e
                
                losses.append(loss.item())
                
                if epoch % 50 == 0:
                    logger.info(f"    Epoch {epoch}: Loss = {loss.item():.4f}")
            
            flow_losses.append(losses)
            logger.info(f"    Final loss: {losses[-1]:.4f}")
        
        self.flows_trained = True
        return flow_losses
    
    def train_mixture_weights_dirichlet_map(self, data, epochs=500, eps=1e-8):
        """
        Stage 2: Learn weights using Dirichlet-MAP update.
        
        Weight update formula:
            π = (N_k + α_dirichlet - 1) / Σ(N_k + α_dirichlet - 1)
        
        where N_k = Σ_n r_nk are expected counts from softmax responsibilities:
            r_nk = softmax_k((log q_k(x_n) + log π_k) / τ)
        
        Args:
            data: Training data
            epochs: Number of Stage-2 iterations
            eps: Numerical stability epsilon
        """
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        logger.info(f"🔄 Stage 2: Learning mixture weights (Dirichlet-MAP | τ={self.tau} | α={self.alpha_dirichlet})...")
        
        K = len(self.flows)
        device = data.device
        
        # Prepare Dirichlet prior
        if isinstance(self.alpha_dirichlet, (int, float)):
            alpha = torch.ones(K, device=device) * self.alpha_dirichlet
        else:
            alpha = torch.tensor(self.alpha_dirichlet, device=device)
        
        weight_losses = []
        
        for epoch in range(epochs):
            # Sample batch from data
            batch_size = min(2000, len(data))
            indices = torch.randperm(len(data), device=device)[:batch_size]
            data_batch = data[indices]

            # E-step: compute per-sample log probs [N, K]
            logps = []
            for flow in self.flows:
                flow.eval()
                with torch.no_grad():
                    lp = flow.log_prob(data_batch)
                    lp = lp.view(-1)
                    logps.append(lp)
            logps = torch.stack(logps, dim=1)  # [N, K]
            
            # log π_k (current weights)
            log_pi = torch.log(self.weights.data.clamp(min=eps))
            
            # Tempered responsibilities: r = softmax((log q_k + log π_k) / τ)
            logits = (logps + log_pi) / self.tau
            r = F.softmax(logits, dim=1)  # [N, K]
            
            # M-step: Dirichlet MAP update
            Nk = r.sum(dim=0)  # Expected counts [K]
            pi_unnorm = Nk + alpha - 1.0
            pi_unnorm = torch.clamp(pi_unnorm, min=eps)
            pi_new = pi_unnorm / pi_unnorm.sum()
            
            # Update weights (no gradients)
            self.weights.data.copy_(pi_new)
            
            # Compute mixture log probability for loss tracking
            batch_weights = self.weights.unsqueeze(0).expand(data.size(0), -1)
            flow_predictions = []
            for flow in self.flows:
                flow.eval()
                with torch.no_grad():
                    log_prob = flow.log_prob(data)
                    if torch.any(torch.isnan(log_prob)) or torch.any(torch.isinf(log_prob)):
                        log_prob = torch.full_like(log_prob, -100.0)
                    flow_predictions.append(log_prob.unsqueeze(1))
            
            flow_predictions = torch.cat(flow_predictions, dim=1)
            weighted_log_probs = flow_predictions + torch.log(batch_weights + eps)
            mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
            loss = -mixture_log_prob.mean()
            
            weight_losses.append(loss.item())
            
            if epoch % 100 == 0:
                current_weights = self.weights.detach().cpu().numpy()
                neff = float(np.exp(-np.sum(current_weights * np.log(current_weights + eps))))
                logger.info(f"    Epoch {epoch}: Loss = {loss.item():.4f}, Weights = {current_weights}, Neff = {neff:.3f}")
        
        final_weights = self.weights.detach().cpu().numpy()
        logger.info(f"    Final weights: {final_weights}")
        
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
                if torch.any(torch.isnan(log_prob)) or torch.any(torch.isinf(log_prob)):
                    log_prob = torch.full_like(log_prob, -100.0)
                flow_log_probs.append(log_prob.unsqueeze(1))
        
        return torch.cat(flow_log_probs, dim=1)  # [batch, n_flows]
    
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
        
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            logger.warning("⚠️  Warning: NaN weights detected, using uniform weights for sampling")
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


def train_sequential_amf_vi(dataset_name='multimodal', flow_types=None, show_plots=True, save_plots=False, 
                            n_samples=100_000, tau=1.2, alpha_dirichlet=1.5):
    """Train sequential AMF-VI with Dirichlet-MAP weight learning."""
    
    logger.info(f"🚀 Sequential AMF-VI v3.0.0 (τ={tau} | α_dir={alpha_dirichlet} | DMAP-TAU) on {dataset_name}")
    logger.info("=" * 60)
    
    if flow_types is None:
        flow_types = ['realnvp', 'maf', 'rbig']
    
    logger.info(f"Using flows: {flow_types}")
    
    split_data = get_split_data(dataset_name, n_samples=n_samples)
    train_data = split_data['train']
    val_data = split_data['val']
    test_data = split_data['test']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    test_data  = test_data.to(device)

    # Create model with Dirichlet-MAP
    model = SequentialAMFVI(dim=2, flow_types=flow_types, weight_update_method='dirichlet_map', 
                            tau=tau, alpha_dirichlet=alpha_dirichlet)
    model.dataset_name = dataset_name
    model.results_dir = os.path.join("./", "results")
    model = model.to(device)

    train_epochs = 3000
    map_epochs = 1000
    
    # Stage 1: Train flows on train data
    flow_losses = model.train_flows_independently(train_data, epochs=train_epochs, lr=1e-3)

    # Stage 2: Learn mixture weights on val data using Dirichlet-MAP
    weight_losses = model.train_mixture_weights_dirichlet_map(
        data=val_data,
        epochs=map_epochs,
    )
    
    # Evaluation and visualization
    logger.info("\n🎨 Generating visualizations...")
    
    model.eval()
    with torch.no_grad():
        model_samples = model.sample(1000)
        
        flow_samples = {}
        for i, flow_type in enumerate(flow_types):
            flow_samples[flow_type] = model.flows[i].sample(1000)
        
        n_flows = len(flow_types)
        n_cols = max(3, n_flows)
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 8))

        data_np = test_data.cpu().numpy()
        axes[0, 0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, c='blue', s=20)
        axes[0, 0].set_title('Target Data (test)')
        axes[0, 0].grid(True, alpha=0.3)

        model_np = model_samples.cpu().numpy()
        axes[0, 1].scatter(model_np[:, 0], model_np[:, 1], alpha=0.6, c='red', s=20)
        axes[0, 1].set_title('AMF-VI Samples')
        axes[0, 1].grid(True, alpha=0.3)

        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta']
        if flow_losses:
            for i, (flow_type, losses) in enumerate(zip(flow_types, flow_losses)):
                axes[0, 2].plot(losses, label=flow_type.upper(), 
                               color=colors[i % len(colors)], linewidth=1, alpha=0.7)
        if weight_losses:
            axes[0, 2].plot(weight_losses, label=f'Weights (DMAP τ={tau})', color='red', linewidth=2)

        axes[0, 2].set_title('Training Losses')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()

        for i, (flow_type, samples) in enumerate(flow_samples.items()):
            col = i
            if col < n_cols:
                samples_np = samples.cpu().numpy()
                axes[1, col].scatter(samples_np[:, 0], samples_np[:, 1], 
                                     alpha=0.6, c=colors[i % len(colors)], s=20)
                axes[1, col].set_title(f'{flow_type.upper()} Flow')
                axes[1, col].grid(True, alpha=0.3)

        for i in range(3, n_cols):
            axes[0, i].set_visible(False)
        for i in range(len(flow_samples), n_cols):
            axes[1, i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f'AMF-VI v3.0.0 (τ={tau}, α_dir={alpha_dirichlet} | DMAP-TAU) — {dataset_name.title()}', fontsize=16)
        
        if save_plots:
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, f'sequential_amf_vi_results_{dataset_name}.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"✅ Plot saved to {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close(fig)
        
        logger.info("\n📊 Analysis (test set):")
        logger.info(f"Target data mean: {test_data.mean(dim=0).cpu().numpy()}")
        logger.info(f"Sequential model mean: {model_samples.mean(dim=0).cpu().numpy()}")
        logger.info(f"Target data std: {test_data.std(dim=0).cpu().numpy()}")
        logger.info(f"Sequential model std: {model_samples.std(dim=0).cpu().numpy()}")
        
        logger.info("\n🔍 Flow Specialization Analysis:")
        learned_weights = model.weights.detach().cpu().numpy()
        neff = float(np.exp(-np.sum(learned_weights * np.log(learned_weights + 1e-8))))
        for i, (flow_type, samples) in enumerate(flow_samples.items()):
            mean = samples.mean(dim=0).cpu().numpy()
            std = samples.std(dim=0).cpu().numpy()
            weight = learned_weights[i]
            logger.info(f"{flow_type.upper()}: Weight={weight:.3f}, Mean=[{mean[0]:.2f}, {mean[1]:.2f}], Std=[{std[0]:.2f}, {std[1]:.2f}]")
        logger.info(f"Neff = {neff:.3f}")
        
        logger.info("\n🏗️ Model Architecture:")
        total_params = 0
        for i, flow in enumerate(model.flows):
            n_params = sum(p.numel() for p in flow.parameters())
            total_params += n_params
            logger.info(f"{flow_types[i].upper()}: {n_params:,} parameters")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Weight parameters: {model.weights.numel()}")
    
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
            'tau': tau,
            'alpha_dirichlet': alpha_dirichlet,
            'version': '3.0.0',
            'abbr': 'AMFVI-DMAP-TAU',
            'metadata': split_data['metadata'],
        }, f)
    logger.info(f"✅ Model saved to {model_path}")
    
    return model, flow_losses, weight_losses


if __name__ == "__main__":
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
        "Real-GMM2",
    ]
    
    flow_types = ['realnvp', 'maf', 'rbig']
    TAU = 1.2
    ALPHA_DIRICHLET = 1.5

    logger.info(f"🚀 AMF-VI v3.0.0 | flows={flow_types} | τ={TAU} | α_dir={ALPHA_DIRICHLET} | DMAP-TAU")
    
    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {len(flow_types)}-flow model on {dataset_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            model, flow_losses, weight_losses = train_sequential_amf_vi(
                dataset_name=dataset_name,
                flow_types=flow_types,
                show_plots=False, 
                save_plots=True,
                n_samples=500_000,
                tau=TAU,
                alpha_dirichlet=ALPHA_DIRICHLET,
            )
            logger.info(f"✅ Completed {dataset_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
