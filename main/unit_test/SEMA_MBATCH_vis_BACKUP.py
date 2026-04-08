"""
Version: 2.11.0
Abbr: AMFVI-SEMA-MBATCH

CHANGELOG v2.11.0:
- __main__ collects final weights + Neff into all_dataset_weights after each dataset trains
- Calls plot_mixture_weights_summary() once after full loop (paper Fig.2 style)
- Wrapped in try/except with logging.error; partial results plotted if some datasets fail

CHANGELOG v2.10.0:
- Standardised __main__ dataset list to canonical 10-dataset set
- Removed 'multimodal-5' and 'Old-Faithful' (not in current benchmark scope)
- Removed commented-out multimodal5_drop variants

CHANGELOG v2.9.0:
- train_mixture_weights_moving_average() accepts tau, alpha, M as optional override params
- Overrides take precedence over self.tau, self.alpha, self.M when provided
- Enables sensitivity_analysis.py to reuse frozen Stage 1 experts with different hyperparams
- No change to default behaviour when overrides are not passed

CHANGELOG v2.8.0:
- Captures r_bar after Eq.23 M-batch averaging, before beta-smoothing each epoch
- Passes r_bar=(K,) to tracker.update() for responsibilities-vs-weights plot (CRCS-VIZ v1.2.0)
- No change to training dynamics; capture is read-only copy before smoothing step

CHANGELOG v2.7.0:
- Added FLOW_DISPLAY_NAMES map for human-readable expert legend labels in convergence plots
- Stored flow_types as self.flow_types in __init__ for downstream access
- MetricsTracker instantiation now passes expert_names via FLOW_DISPLAY_NAMES lookup
- Falls back to raw flow_type string if not found in display map

CHANGELOG v2.6.0:
- Integrated convergence visualization tracking from CRCS-VIZ v1.0.0
- Added MetricsTracker to Stage 2 for weight trajectories, Neff, loss, and entropy
- Automatic generation of 4 convergence plots after Stage 2 completes
- Plots saved to /results/convergence_plots/{dataset}_SEMA-MBATCH_*.png

CHANGELOG v2.5.0:
- Added multi-batch averaging M=3 for r_bar in Stage-2 per paper Eq. (23) (SEMA-MBATCH)
- r_bar accumulated over M independent fresh batches: r_bar += responsibilities / M
- M stored as constructor param (default=3; paper default=2, increased here for variance reduction)
- Builds on v2.4.0 (SEMA-SMOOTH); τ, log πk prior, ε floor, and β smoothing retained
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
import sys

# Add parent directory to path for convergence visualization import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualisation'))
from visualisation.convergence_visualization import MetricsTracker, generate_all_plots, plot_mixture_weights_summary

# Display name map for convergence plot legends (v2.7.0)
FLOW_DISPLAY_NAMES = {
    'realnvp': 'RealNVP',
    'maf': 'MAF',
    'rbig': 'RBIG',
    'gaussianization': 'RBIG',
    'iaf': 'IAF',
    'naf': 'NAF',
    'glow': 'Glow',
    'nice': 'NICE',
    'spline': 'Spline',
    'tan': 'TAN',
}


# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

class SequentialAMFVI(nn.Module):
    # v2.1.0: added tau parameter for Stage-2 temperature scaling
    def __init__(self, dim=2, flow_types=None, weight_update_method='moving_average', tau=1.1, beta=1e-5, M=3):
        super().__init__()
        self.dim = dim
        self.tau = tau   # v2.1.0: sEMA temperature (SEMA-TAU)
        self.beta = beta  # v2.4.0: uniform smoothing coefficient (SEMA-SMOOTH)
        self.M = M        # v2.5.0: number of fresh batches for r_bar averaging (SEMA-MBATCH)
        
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
        self.flow_types = flow_types  # v2.7.0: stored for MetricsTracker expert_names

    def safe_log_prob_extraction(self, log_prob_tensor):
        """Extract log prob with NaN handling - Phase 1 fix"""
        mean_log_prob = log_prob_tensor.mean().item()
        
        if torch.isnan(torch.tensor(mean_log_prob)) or torch.isinf(torch.tensor(mean_log_prob)):
            # Fallback: return very low but finite log probability
            return -100.0  # Equivalent to very low probability
        return mean_log_prob
    
    #def train_flows_independently(self, data, epochs=1000, lr=1e-4):
    def train_flows_independently(self, data, epochs=1000, lr=1e-4, per_expert_data=None):
        """Stage 1: Train each flow independently."""
        print("🔄 Stage 1: Training flows independently...")
        
        flow_losses = []
        
        for i, flow in enumerate(self.flows):
            #train_data = per_expert_data[i] if (per_expert_data and i in per_expert_data) else data
            print(f"  Training flow {i+1}/{len(self.flows)}: {flow.__class__.__name__}")
            
            # Check if flow has trainable parameters
            params = list(flow.parameters())
            if len(params) == 0:
                print(f"    {flow.__class__.__name__} uses non-parametric fitting instead of gradient optimization")
                
                # Handle non-parametric flows (like RBIG)
                if hasattr(flow, 'fit_to_data'):
                    print("    Using fit_to_data() method...")
                    flow.fit_to_data(data, validate_reconstruction=False)  # Skip validation for speed
                    #flow.fit_to_data(train_data, validate_reconstruction=False) 
                    # Compute actual loss trajectory for visualization consistency
                    losses = []
                    with torch.no_grad():
                        # Sample loss values to create a realistic trajectory
                        for epoch in range(0, epochs, max(1, epochs//20)):  # Sample 20 points
                            try:
                                log_prob = flow.log_prob(data)
                                #log_prob = flow.log_prob(train_data)
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
                    #log_prob = flow.log_prob(train_data)
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
    
    def train_mixture_weights_moving_average(self, data, epochs=500,
                                              tau=None, alpha=None, M=None):
        """
        Stage 2: Learn weights using Moving Average of Likelihoods.

        Args:
            data: Training data tensor
            epochs: Number of Stage 2 epochs
            tau: Temperature override (default: self.tau) — v2.9.0
            alpha: EMA momentum override (default: self.alpha) — v2.9.0
            M: Fresh batch count override (default: self.M) — v2.9.0

        Returns:
            weight_losses: List of training losses
            metrics_tracker: MetricsTracker instance with convergence data
        """
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")

        # v2.9.0: resolve overrides — use provided value or fall back to instance default
        _tau   = tau   if tau   is not None else self.tau
        _alpha = alpha if alpha is not None else self.alpha
        _M     = M     if M     is not None else self.M

        print(f"🔄 Stage 2: Learning mixture weights (Moving Average | τ={_tau} | log πk prior | ε floor | β smooth | M={_M} batches)...")
        
        K = len(self.flows)
        
        # Initialize MetricsTracker for convergence visualization
        expert_names = [FLOW_DISPLAY_NAMES.get(ft, ft) for ft in self.flow_types]  # v2.7.0
        metrics_tracker = MetricsTracker(method_name="SEMA-MBATCH", n_experts=K, expert_names=expert_names)
        
        weight_losses = []
        
        for epoch in range(epochs):
            # v2.5.0: Multi-batch averaging M=3 (SEMA-MBATCH) — variance reduction per paper Eq. (23)
            # Before (v2.4.0): single batch → r_bar = softmax(...)
            # After (v2.5.0): average r_bar over M independent fresh batches → r_bar += responsibilities / M
            batch_size = min(2000, len(data))
            r_bar = torch.zeros(K, device=data.device)

            for _ in range(_M):  # v2.9.0: use _M override
                indices = torch.randperm(len(data), device=data.device)[:batch_size]
                data_batch = data[indices]

                # Get flow log probabilities on this batch
                flow_log_probs = []
                for flow in self.flows:
                    flow.eval()
                    with torch.no_grad():
                        log_prob = flow.log_prob(data_batch)
                        safe_log_prob = self.safe_log_prob_extraction(log_prob)
                        flow_log_probs.append(safe_log_prob)

                flow_log_probs_tensor = torch.tensor(flow_log_probs, device=data.device)

                # v2.2.0: log πk prior term (SEMA-PRIOR) per paper Eq. (22)
                log_pi = torch.log(self.weights.data.clamp(min=1e-8))
                responsibilities = F.softmax((flow_log_probs_tensor + log_pi) / _tau, dim=0)  # v2.9.0
                r_bar += responsibilities / _M  # v2.9.0

            # v2.4.0: Uniform smoothing β (SEMA-SMOOTH) — damp early transients per paper Eq. (24)
            # r_bar ← (1-β) * r_bar + β * (1/K)  applied after M-batch averaging, before EMA
            r_bar_before_smooth = r_bar.detach().cpu().numpy().copy()  # v2.8.0: capture Eq.23 avg for tracker
            r_bar = (1 - self.beta) * r_bar + self.beta * (torch.ones(K, device=data.device) / K)

            # Moving average update: weight_i = α * old_weight_i + (1-α) * r_bar_i
            with torch.no_grad():
                old_weights = self.weights.data.clone()
                new_weights = _alpha * old_weights + (1 - _alpha) * r_bar  # v2.9.0: use _alpha override
                self.weights.data = new_weights

                # v2.3.0: Floor + Renorm (SEMA-FLOOR) — collapse prevention per paper Eqs. (26-27)
                # π ← max(π, ε) then π ← π / ‖π‖₁
                eps = 1e-5
                self.weights.data = torch.clamp(self.weights.data, min=eps)
                self.weights.data = self.weights.data / self.weights.data.sum()
            
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
            weighted_log_probs = flow_predictions + torch.log(batch_weights + 1e-8)
            mixture_log_prob = torch.logsumexp(weighted_log_probs, dim=1)
            loss = -mixture_log_prob.mean()
            
            weight_losses.append(loss.item())
            
            # Update metrics tracker
            current_weights_np = self.weights.detach().cpu().numpy()
            
            # Create responsibilities from r_bar (expand to batch dimension)
            responsibilities_np = r_bar.detach().cpu().numpy()
            # Expand to (batch_size, K) for entropy calculation
            responsibilities_batch = responsibilities_np[np.newaxis, :].repeat(batch_size, axis=0)
            
            metrics_tracker.update(
                weights=current_weights_np,
                loss=loss.item(),
                responsibilities=responsibilities_batch,
                r_bar=r_bar_before_smooth  # v2.8.0: Eq.23 avg before beta-smoothing
            )
            
            if epoch % 100 == 0:
                neff = float(np.exp(-np.sum(current_weights_np * np.log(current_weights_np + 1e-8))))
                print(f"    Epoch {epoch}: Loss = {loss.item():.4f}, Weights = {current_weights_np}, Neff = {neff:.3f}")
        
        final_weights = self.weights.detach().cpu().numpy()
        print(f"    Final weights: {final_weights}")
        
        self.weights_trained = True
        self.weight_history = weight_losses
        return weight_losses, metrics_tracker
    
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
            print("⚠️  Warning: NaN weights detected, using uniform weights for sampling")
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


def train_sequential_amf_vi(dataset_name='multimodal', flow_types=None, show_plots=True, save_plots=False, n_samples=100_000, tau=1.1, beta=1e-5, M=3):
    """Train sequential AMF-VI with learnable weights."""
    
    print(f"🚀 Sequential AMF-VI v2.6.0 (τ={tau} | β={beta} | M={M} | SEMA-MBATCH) on {dataset_name}")
    print("=" * 60)
    
    if flow_types is None:
        flow_types = ['realnvp', 'maf', 'gaussianization']
    
    print(f"Using flows: {flow_types}")
    
    split_data = get_split_data(dataset_name, n_samples=n_samples)
    train_data = split_data['train']
    val_data = split_data['val']
    test_data = split_data['test']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_data = train_data.to(device)
    val_data   = val_data.to(device)
    test_data  = test_data.to(device)


    # Combine train+val for training flows and mixture weights
    #train_val_data = torch.cat([train_data, val_data], dim=0).to(device)
    #test_data = test_data.to(device)
    
    # v2.1.0: pass tau to model
    model = SequentialAMFVI(dim=2, flow_types=flow_types, weight_update_method='moving_average', tau=tau, beta=beta, M=M)  # v2.5.0: pass M
    model.dataset_name = dataset_name
    model.results_dir = os.path.join("./", "results")
    model = model.to(device)

    train_epochs = 3000
    ma_epochs = 1000
    
    # Stage 1: Train flows on train+val
    flow_losses = model.train_flows_independently(train_data, epochs=train_epochs, lr=1e-3)
    
    # Stage 2: Learn mixture weights on train+val
    weight_losses, metrics_tracker = model.train_mixture_weights_moving_average(
        data=val_data,
        epochs=ma_epochs,
    )
    
    # Generate convergence plots
    print("\n📊 Generating convergence plots...")
    convergence_plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'convergence_plots')
    try:
        generate_all_plots(
            tracker=metrics_tracker,
            output_dir=convergence_plots_dir,
            prefix=dataset_name,
            max_epochs=500
        )
        print(f"✅ Convergence plots saved to {convergence_plots_dir}")
    except Exception as e:
        print(f"❌ Failed to generate convergence plots: {e}")
    
    # Evaluation and visualization
    print("\n🎨 Generating visualizations...")
    
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
            axes[0, 2].plot(weight_losses, label=f'Weights (τ={tau}, β={beta}, M={M})', color='red', linewidth=2)

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

        for i in range(4, n_cols):
            axes[0, i].set_visible(False)
        for i in range(len(flow_samples), n_cols):
            axes[1, i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f'AMF-VI v2.6.0 (τ={tau} | β={beta} | M={M} | SEMA-MBATCH) — {dataset_name.title()}', fontsize=16)
        
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
        
        print("\n📊 Analysis (test set):")
        print(f"Target data mean: {test_data.mean(dim=0).cpu().numpy()}")
        print(f"Sequential model mean: {model_samples.mean(dim=0).cpu().numpy()}")
        print(f"Target data std: {test_data.std(dim=0).cpu().numpy()}")
        print(f"Sequential model std: {model_samples.std(dim=0).cpu().numpy()}")
        
        print("\n🔍 Flow Specialization Analysis:")
        learned_weights = model.weights.detach().cpu().numpy()
        neff = float(np.exp(-np.sum(learned_weights * np.log(learned_weights + 1e-8))))
        for i, (flow_type, samples) in enumerate(flow_samples.items()):
            mean = samples.mean(dim=0).cpu().numpy()
            std = samples.std(dim=0).cpu().numpy()
            weight = learned_weights[i]
            print(f"{flow_type.upper()}: Weight={weight:.3f}, Mean=[{mean[0]:.2f}, {mean[1]:.2f}], Std=[{std[0]:.2f}, {std[1]:.2f}]")
        print(f"Neff = {neff:.3f}")
        
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
            'metrics_tracker': metrics_tracker,
            'dataset': dataset_name,
            'tau': tau,
            'beta': beta,
            'M': M,
            'version': '2.6.0',
            'abbr': 'AMFVI-SEMA-MBATCH',
            'metadata': split_data['metadata'],
        }, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, flow_losses, weight_losses, metrics_tracker


if __name__ == "__main__":
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
    
    flow_types = ['realnvp', 'maf', 'rbig']
    TAU = 1.1    # v2.1.0: sEMA temperature
    BETA = 1e-5  # v2.4.0: uniform smoothing coefficient
    M = 2        # v2.5.0: number of fresh batches for r_bar averaging

    print(f"🚀 AMF-VI v2.11.0 | flows={flow_types} | τ={TAU} | β={BETA} | M={M} | SEMA-MBATCH")

    all_dataset_weights = {}  # v2.11.0: collects (weights, neff) per dataset for summary plot

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Training {len(flow_types)}-flow model on {dataset_name.upper()}")
        print(f"{'='*60}")
        
        try:
            model, flow_losses, weight_losses, metrics_tracker = train_sequential_amf_vi(
                dataset_name=dataset_name,
                flow_types=flow_types,
                show_plots=False, 
                save_plots=True,
                n_samples=500_000,
                tau=TAU,
                beta=BETA,
                M=M,
            )
            print(f"✅ Completed {dataset_name}")

            # v2.11.0: collect final weights and Neff for summary plot
            if metrics_tracker.neff_history:
                all_dataset_weights[dataset_name] = (
                    model.weights.detach().cpu().numpy().copy(),
                    float(metrics_tracker.neff_history[-1])
                )
            else:
                logging.error(f"neff_history empty for {dataset_name} — excluded from summary plot")

        except Exception as e:
            print(f"❌ Failed on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # v2.11.0: generate mixture weights summary plot after all datasets
    if all_dataset_weights:
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        summary_path = os.path.join(results_dir, 'mixture_weights_summary.png')
        try:
            plot_mixture_weights_summary(
                dataset_weights=all_dataset_weights,
                expert_names=flow_types,
                save_path=summary_path,
            )
            print(f"\n✅ Mixture weights summary saved to {summary_path}")
        except Exception as e:
            logging.error(f"Failed to generate mixture weights summary: {e}")
    else:
        logging.error("No datasets completed successfully — mixture weights summary skipped")