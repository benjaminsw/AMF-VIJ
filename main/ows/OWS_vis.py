"""
Version: 3.0.0
Abbr: AMFVI-OWS
Base: AMFVI-SEMA-MBATCH v2.6.0
Plan: IMPL-OWS-CORE-v1.0

CHANGELOG v3.0.0:
- Replaced sEMA heuristic (Stage 2) with Optimal Weight Solving via EM algorithm
- Removed sEMA hyperparameters: τ (tau), α (alpha), β (beta), M, batch_size
- Added precompute_log_probs() for one-time cached (N, K) log-likelihood matrix
- Added train_mixture_weights_em() with E-step/M-step, convergence check, monotonicity guard
- Added post-hoc safety net: fallback to best single expert if mixture NLL > best expert NLL
- Stage 2 hyperparameters reduced from 6 to 3: max_iter, tol, eps
- (R4) Updated convergence plots: x-axis = EM iterations, added convergence marker line
- Stage 1 completely unchanged
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
import logging

# Configure logging — errors always logged, info for diagnostics
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for convergence visualization import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualisation'))
from visualisation.convergence_visualization import MetricsTracker, generate_all_plots


# Set seed for reproducible experiments
torch.manual_seed(2025)
np.random.seed(2025)

class SequentialAMFVI(nn.Module):
    # v3.0.0: removed sEMA params (tau, beta, M, alpha); added OWS-EM params (max_iter, tol, eps)
    def __init__(self, dim=2, flow_types=None, max_iter=100, tol=1e-8, eps=0.0):
        super().__init__()
        self.dim = dim
        self.max_iter = max_iter  # v3.0.0: max EM iterations
        self.tol = tol            # v3.0.0: convergence threshold on NLL change
        self.eps = eps            # v3.0.0: optional floor (0 = no floor, strict guarantee)
        
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
        
        # Initialize weights as parameters
        self.weights = nn.Parameter(torch.ones(len(self.flows)) / len(self.flows))
        self.weight_history = []
        
        # Track if flows are trained
        self.flows_trained = False
        self.weights_trained = False
        
        # v3.0.0: EM convergence metadata (populated after Stage 2)
        self.em_converged_iter = None  # iteration where EM converged (None = hit max_iter)
        self.safety_net_triggered = False

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
    
    # =========================================================================
    # v3.0.0 (WP-OWS-S1): Precompute expert log-likelihoods — one-time cost
    # =========================================================================
    def precompute_log_probs(self, data):
        """Precompute log q_k(z_i) for all experts on full dataset.
        
        Args:
            data: Training data tensor (N, d)
        
        Returns:
            log_probs: Cached log-likelihood matrix (N, K)
        """
        K = len(self.flows)
        N = data.size(0)
        log_probs = torch.zeros(N, K, device=data.device)
        
        for k, flow in enumerate(self.flows):
            flow.eval()
            with torch.no_grad():
                lp = flow.log_prob(data)
                # Clamp to prevent -inf from breaking EM
                lp = torch.clamp(lp, min=-100.0)
                if torch.any(torch.isnan(lp)):
                    logger.error(f"NaN in log_prob for expert {k} ({flow.__class__.__name__})")
                    lp = torch.where(torch.isnan(lp), torch.tensor(-100.0, device=data.device), lp)
                if torch.any(torch.isinf(lp)):
                    logger.error(f"Inf in log_prob for expert {k} ({flow.__class__.__name__})")
                    lp = torch.where(torch.isinf(lp), torch.tensor(-100.0, device=data.device), lp)
                log_probs[:, k] = lp
        
        logger.info(f"Precomputed log-probs: shape={log_probs.shape}, "
                     f"min={log_probs.min().item():.2f}, max={log_probs.max().item():.2f}")
        return log_probs  # (N, K)
    
    # =========================================================================
    # v3.0.0 (WP-OWS-S2): EM Weight Solver — replaces sEMA entirely
    # =========================================================================
    def train_mixture_weights_em(self, data, max_iter=None, tol=None, eps=None):
        """
        Stage 2 (OWS): Learn mixture weights via EM on frozen experts.
        Guaranteed to decrease NLL at every step (or converge).
        
        Args:
            data: Training data tensor (N, d)
            max_iter: Maximum EM iterations (default: self.max_iter)
            tol: Convergence threshold on NLL change (default: self.tol)
            eps: Floor constraint, 0 = no floor for strict guarantee (default: self.eps)
        
        Returns:
            weight_losses: List of NLL per iteration
            metrics_tracker: MetricsTracker instance
        """
        if not self.flows_trained:
            raise RuntimeError("Flows must be trained first!")
        
        # Use instance defaults if not overridden
        max_iter = max_iter if max_iter is not None else self.max_iter
        tol = tol if tol is not None else self.tol
        eps = eps if eps is not None else self.eps
        
        K = len(self.flows)
        N = data.size(0)
        
        print(f"🔄 Stage 2: Learning mixture weights (OWS-EM | max_iter={max_iter} | tol={tol} | eps={eps})...")
        logger.info(f"Stage 2 (OWS-EM): N={N}, K={K}, max_iter={max_iter}, tol={tol}, eps={eps}")
        
        # --- Precompute log q_k(z_i) once ---
        log_Q = self.precompute_log_probs(data)  # (N, K)
        
        # Initialize MetricsTracker for convergence visualization
        metrics_tracker = MetricsTracker(method_name="OWS-EM", n_experts=K)
        
        # --- Initialise weights uniformly ---
        pi = torch.ones(K, device=data.device) / K
        
        weight_losses = []
        prev_nll = float('inf')
        self.em_converged_iter = None
        self.safety_net_triggered = False
        
        for t in range(max_iter):
            # === E-step: compute responsibilities γ_ik ===
            # log(π_k * q_k(z_i)) = log π_k + log q_k(z_i)
            log_joint = log_Q + torch.log(pi).unsqueeze(0)  # (N, K)
            
            # γ_ik = π_k q_k(z_i) / Σ_j π_j q_j(z_i)  via logsumexp
            log_denom = torch.logsumexp(log_joint, dim=1, keepdim=True)  # (N, 1)
            log_gamma = log_joint - log_denom  # (N, K)
            gamma = torch.exp(log_gamma)  # (N, K)
            
            # === M-step: update weights ===
            pi_new = gamma.mean(dim=0)  # (K,)  = (1/N) Σ_i γ_ik
            
            # Optional floor + renormalise
            if eps > 0:
                pi_new = torch.clamp(pi_new, min=eps)
                pi_new = pi_new / pi_new.sum()
            
            pi = pi_new
            
            # === Compute mixture NLL ===
            # NLL = -(1/N) Σ_i log[Σ_k π_k q_k(z_i)]
            log_mixture = torch.logsumexp(log_Q + torch.log(pi).unsqueeze(0), dim=1)  # (N,)
            nll = -log_mixture.mean().item()
            weight_losses.append(nll)
            
            # === Diagnostics ===
            pi_np = pi.detach().cpu().numpy()
            H = -float(np.sum(pi_np * np.log(pi_np + 1e-30)))
            neff = float(np.exp(H))
            
            # Update metrics tracker
            gamma_np = gamma.detach().cpu().numpy()
            metrics_tracker.update(
                weights=pi_np,
                loss=nll,
                responsibilities=gamma_np
            )
            
            if t % 10 == 0:
                print(f"    EM iter {t}: NLL={nll:.6f}, π={pi_np}, Neff={neff:.3f}")
            
            # === Convergence check ===
            nll_change = abs(prev_nll - nll)
            if nll_change < tol and t > 0:
                self.em_converged_iter = t
                print(f"    ✅ EM converged at iter {t}: ΔNLL={nll_change:.2e} < tol={tol}")
                logger.info(f"EM converged at iter {t}: ΔNLL={nll_change:.2e} < tol={tol}")
                break
            
            # === Monotonicity check (should never trigger for proper EM) ===
            if nll > prev_nll + 1e-10 and t > 0:
                logger.error(f"EM NLL INCREASED at iter {t}: {prev_nll:.6f} → {nll:.6f} "
                             f"(Δ={nll - prev_nll:.2e}). This should not happen with correct EM.")
            
            prev_nll = nll
        
        if self.em_converged_iter is None:
            logger.warning(f"EM did not converge within {max_iter} iterations (final ΔNLL={nll_change:.2e})")
        
        # === Post-hoc safety net ===
        # Compare mixture NLL to best single expert NLL
        best_expert_nll = float('inf')
        best_expert_idx = -1
        for k in range(K):
            expert_nll = -log_Q[:, k].mean().item()
            if expert_nll < best_expert_nll:
                best_expert_nll = expert_nll
                best_expert_idx = k
        
        if nll > best_expert_nll:
            self.safety_net_triggered = True
            logger.warning(
                f"SAFETY NET: Mixture NLL ({nll:.6f}) > best expert {best_expert_idx} "
                f"NLL ({best_expert_nll:.6f}). Falling back to single expert."
            )
            print(f"    ⚠️  SAFETY NET: Mixture NLL ({nll:.6f}) > best expert {best_expert_idx} "
                  f"NLL ({best_expert_nll:.6f}). Falling back to single expert.")
            pi = torch.zeros(K, device=data.device)
            pi[best_expert_idx] = 1.0
            # Recompute final NLL
            log_mixture = log_Q[:, best_expert_idx]
            nll = -log_mixture.mean().item()
            weight_losses.append(nll)
        
        # Store final weights
        with torch.no_grad():
            self.weights.data = pi
        
        final_weights = pi.detach().cpu().numpy()
        print(f"    Final weights: {final_weights}")
        print(f"    Final NLL: {nll:.6f}")
        logger.info(f"Final weights: {final_weights}, NLL: {nll:.6f}")
        
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


# v3.0.0 (WP-OWS-S3): Updated pipeline — EM replaces sEMA, sEMA params removed
def train_sequential_amf_vi(dataset_name='multimodal', flow_types=None, show_plots=True, save_plots=False,
                             n_samples=100_000, max_iter=100, tol=1e-8, eps=0.0):
    """Train sequential AMF-VI with OWS-EM weight solving."""
    
    print(f"🚀 Sequential AMF-VI v3.0.0 (OWS-EM | max_iter={max_iter} | tol={tol} | eps={eps}) on {dataset_name}")
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

    # v3.0.0: pass OWS-EM params to model (no tau, beta, M)
    model = SequentialAMFVI(dim=2, flow_types=flow_types, max_iter=max_iter, tol=tol, eps=eps)
    model.dataset_name = dataset_name
    model.results_dir = os.path.join("./", "results")
    model = model.to(device)

    train_epochs = 3000
    
    # Stage 1: Train flows (unchanged)
    flow_losses = model.train_flows_independently(train_data, epochs=train_epochs, lr=1e-3)
    
    # Stage 2: Learn mixture weights via OWS-EM on validation data
    weight_losses, metrics_tracker = model.train_mixture_weights_em(
        data=val_data,
        max_iter=max_iter,
        tol=tol,
        eps=eps,
    )
    
    # Generate convergence plots
    # v3.0.0 (R4): x-axis = EM iterations, convergence marker, no volatility/churn
    print("\n📊 Generating convergence plots...")
    convergence_plots_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'convergence_plots')
    try:
        generate_all_plots(
            tracker=metrics_tracker,
            output_dir=convergence_plots_dir,
            prefix=dataset_name
        )
        
        # v3.0.0 (R4): Additional EM-specific convergence plot with convergence marker
        _plot_em_convergence(
            weight_losses=weight_losses,
            converged_iter=model.em_converged_iter,
            safety_net_triggered=model.safety_net_triggered,
            dataset_name=dataset_name,
            output_dir=convergence_plots_dir,
        )
        
        print(f"✅ Convergence plots saved to {convergence_plots_dir}")
    except Exception as e:
        logger.error(f"Failed to generate convergence plots: {e}")
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
            # v3.0.0 (R4): label reflects OWS-EM, not sEMA
            axes[0, 2].plot(weight_losses, label=f'OWS-EM (iters={len(weight_losses)})', color='red', linewidth=2)

        axes[0, 2].set_title('Training Losses')
        axes[0, 2].set_xlabel('Epoch / EM Iteration')
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
        # v3.0.0: updated title
        plt.suptitle(f'AMF-VI v3.0.0 (OWS-EM | max_iter={max_iter} | tol={tol}) — {dataset_name.title()}', fontsize=16)
        
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
        
        # v3.0.0: EM-specific summary
        print(f"\n🔧 OWS-EM Summary:")
        print(f"EM iterations used: {len(weight_losses)}")
        print(f"Converged at iter: {model.em_converged_iter if model.em_converged_iter is not None else 'did not converge'}")
        print(f"Safety net triggered: {model.safety_net_triggered}")
        
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
    
    # v3.0.0: updated pickle metadata — no tau, beta, M
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model, 
            'flow_losses': flow_losses, 
            'weight_losses': weight_losses,
            'metrics_tracker': metrics_tracker,
            'dataset': dataset_name,
            'max_iter': max_iter,
            'tol': tol,
            'eps': eps,
            'em_converged_iter': model.em_converged_iter,
            'safety_net_triggered': model.safety_net_triggered,
            'version': '3.0.0',
            'abbr': 'AMFVI-OWS',
            'metadata': split_data['metadata'],
        }, f)
    print(f"✅ Model saved to {model_path}")
    
    return model, flow_losses, weight_losses, metrics_tracker


# =============================================================================
# v3.0.0 (R4): EM-specific convergence plot
# =============================================================================
def _plot_em_convergence(weight_losses, converged_iter, safety_net_triggered,
                         dataset_name, output_dir):
    """Generate EM-specific NLL convergence plot with convergence marker.
    
    x-axis = EM iteration (not epoch). Vertical line at convergence point.
    Any NLL increase is visually flagged (should never happen with correct EM).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    iters = list(range(len(weight_losses)))
    ax.plot(iters, weight_losses, 'b-o', markersize=3, linewidth=1.5, label='Mixture NLL')
    
    # Convergence marker
    if converged_iter is not None:
        ax.axvline(x=converged_iter, color='green', linestyle='--', linewidth=1.5,
                    label=f'Converged (iter {converged_iter})')
    
    # Safety net marker
    if safety_net_triggered and len(weight_losses) > 1:
        ax.axhline(y=weight_losses[-1], color='orange', linestyle=':', linewidth=1.5,
                    label='Safety net fallback')
    
    # Flag any NLL increases (should not happen)
    for i in range(1, len(weight_losses)):
        if weight_losses[i] > weight_losses[i-1] + 1e-10:
            ax.plot(i, weight_losses[i], 'rx', markersize=10, markeredgewidth=2)
            logger.error(f"NLL increase at iter {i}: {weight_losses[i-1]:.6f} → {weight_losses[i]:.6f}")
    
    ax.set_xlabel('EM Iteration')
    ax.set_ylabel('NLL')
    ax.set_title(f'OWS-EM Convergence — {dataset_name.title()}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f'{dataset_name}_OWS-EM_convergence.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"EM convergence plot saved to {plot_path}")


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
        #"multimodal5_drop0",
        #"multimodal5_drop1",
        #"multimodal5_drop2",
        "Real-GMM2",
        "Old-Faithful",
        "Iris-3Class",
    ]
    
    flow_types = ['realnvp', 'maf', 'rbig']
    MAX_ITER = 100   # v3.0.0: max EM iterations
    TOL = 1e-8       # v3.0.0: convergence threshold
    EPS = 0.0        # v3.0.0: no floor (strict monotone guarantee)

    print(f"🚀 AMF-VI v3.0.0 | flows={flow_types} | max_iter={MAX_ITER} | tol={TOL} | eps={EPS} | OWS-EM")
    
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
                max_iter=MAX_ITER,
                tol=TOL,
                eps=EPS,
            )
            print(f"✅ Completed {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed on {dataset_name}: {e}")
            print(f"❌ Failed on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
