"""
Collapse Resistance & Convergence Speed Visualization
Abbreviation: CRCS-VIZ
Version: 1.10.0

CHANGELOG v1.10.0:
- generate_all_plots(): plot_weight_trajectories() commented out; plot_weight_dynamics_neff() uncommented
- _weights.png now shows dual y-axis weights + Neff(t) (paper Fig.4 left panel style)
- No function signature changes; plot_weight_trajectories() remains intact

CHANGELOG v1.9.0:
- Added max_epochs: Optional[int] = None param to plot_weight_trajectories()
- Added max_epochs: Optional[int] = None param to plot_neff_evolution()
- Added max_epochs: Optional[int] = None param to plot_training_objective()
- Added max_epochs: Optional[int] = None param to plot_responsibility_entropy()
- generate_all_plots() now passes max_epochs to all four plot functions above

CHANGELOG v1.8.0:
- generate_all_plots() adds save_json=True param; exports all tracker arrays to {prefix}_metrics.json
- plot_responsibilities_vs_weights() removes r̄_k solid line (still tracked and exported to JSON)
- Legend drops from 3 to 2 lines per expert: π_k dashed + r_k(z_n) dotted+shaded only
- JSON export includes all history arrays: weights, neff, loss, r_bar, r_k_mean, r_k_std

CHANGELOG v1.7.0:
- MetricsTracker stores r_k_mean_history and r_k_std_history (K,) per epoch — true per-sample Eq.22 stats
- update() accepts r_k_mean and r_k_std optional params (K,) with shape validation
- plot_responsibilities_vs_weights() adds dotted line (r_k mean) + shaded std band per expert
- to_dict() exports r_k_mean and r_k_std arrays; legend updated to 3 lines per expert
- Requires SEMA_MBATCH_vis.py v2.12.0+ to populate r_k_mean/r_k_std (per-sample fix)

CHANGELOG v1.6.0:
- Added max_epochs: Optional[int] = None param to plot_responsibilities_vs_weights()
- Added max_epochs: Optional[int] = None param to plot_weight_dynamics_neff()
- Added max_epochs: Optional[int] = None param to generate_all_plots() with pass-through
- When set, both _resp_weights.png and _weight_dynamics_neff.png show only first max_epochs epochs
- Default None preserves existing behaviour; other plots unaffected

CHANGELOG:
- 1.5.0: New plot_mixture_weights_summary() for post-training all-dataset summary (paper Fig.2)
  * Input: dict[dataset_name -> (weights_array_K, neff_float)]; expert_names list; save_path
  * Horizontal stacked bar chart, one row per dataset, bottom-to-top order
  * Fixed colors: RealNVP=#4C72B0, MAF=#DD8452, RBIG=#55A868; legend order RBIG/MAF/REALNVP
  * Right-side N_eff=X.XX annotation; uses DISPLAY_NAMES for y-tick labels
- 1.4.0: Fixed compute_neff() to use Shannon entropy exp(H(pi)) per paper definition
  * Replaces inverse Simpson index 1/sum(pi_k^2) which gave different numerical values
  * Affects neff_history, _neff.png, and _weight_dynamics_neff.png outputs
- 1.3.0: New plot_weight_dynamics_neff(): dual y-axis combining weights (solid) and Neff (gray dashed)
  * Matches paper Fig.4 left panel style with Stage-2 epoch x-axis
  * generate_all_plots() calls new plot -> _weight_dynamics_neff.png
  * Existing plot_weight_trajectories() and plot_neff_evolution() unchanged
- 1.2.0: Added r_bar tracking and responsibilities-vs-weights plot
  * MetricsTracker stores r_bar_history (K,) per epoch — Eq.23 avg before beta-smoothing
  * update() accepts optional r_bar param (K,) validated against n_experts
  * New plot_responsibilities_vs_weights(): solid=r_bar, dashed=pi per expert with expert_names
  * generate_all_plots() calls new plot when r_bar_history is non-empty -> _resp_weights.png
- 1.1.0: Added expert_names support to MetricsTracker
  * __init__ accepts optional expert_names list; falls back to 'Expert k' if None
  * plot_weight_trajectories uses expert_names for legend labels
  * No changes to other plot functions or metric computations
- 1.0.0: Initial implementation with MetricsTracker and 4 core plot types
  * plot_weight_trajectories() - mixture weights over epochs
  * plot_neff_evolution() - effective number of experts
  * plot_training_objective() - training loss curve  
  * plot_responsibility_entropy() - mean entropy of responsibilities
  * Error logging to console for all operations
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """
    Track training metrics per epoch for convergence analysis.
    
    Stores:
    - weights: mixture weights π_k over epochs
    - neff: effective number of experts
    - loss: training objective value
    - entropy: mean responsibility entropy
    - r_bar: averaged responsibilities (K,) before beta-smoothing, Eq.23
    - r_k_mean: per-sample r_k(z_n) mean over batch (K,), true Eq.22  [v1.7.0]
    - r_k_std:  per-sample r_k(z_n) std  over batch (K,), true Eq.22  [v1.7.0]
    """
    
    def __init__(self, method_name: str, n_experts: int, expert_names: Optional[List[str]] = None):
        """
        Initialize metrics tracker.
        
        Args:
            method_name: Name of method (e.g., 'EMA', 'DMAP-TAU')
            n_experts: Number of expert flows
            expert_names: Optional list of expert names for plot legends (e.g., ['RealNVP', 'MAF', 'RBIG']).
                          Falls back to ['Expert 1', 'Expert 2', ...] if None or length mismatch.
        """
        self.method_name = method_name
        self.n_experts = n_experts

        # Validate expert_names
        if expert_names is not None and len(expert_names) == n_experts:
            self.expert_names = expert_names
        else:
            if expert_names is not None:
                logger.error(f"expert_names length {len(expert_names)} != n_experts {n_experts}; falling back to default labels")
            self.expert_names = [f'Expert {k+1}' for k in range(n_experts)]

        self.weights = []  # List of weight vectors per epoch
        self.neff_history = []
        self.loss_history = []
        self.entropy_history = []
        self.r_bar_history = []      # v1.2.0: per-epoch averaged responsibilities (K,) before beta-smoothing
        self.r_k_mean_history = []   # v1.7.0: per-epoch mean of per-sample r_k(z_n) (K,), Eq.22
        self.r_k_std_history  = []   # v1.7.0: per-epoch std  of per-sample r_k(z_n) (K,), Eq.22
        
        logger.info(f"Initialized MetricsTracker for {method_name} with {n_experts} experts: {self.expert_names}")
    
    def update(self, 
               weights: np.ndarray,
               loss: Optional[float] = None,
               responsibilities: Optional[np.ndarray] = None,
               r_bar: Optional[np.ndarray] = None,
               r_k_mean: Optional[np.ndarray] = None,
               r_k_std: Optional[np.ndarray] = None):
        """
        Update metrics for current epoch.
        
        Args:
            weights: Current mixture weights, shape (n_experts,)
            loss: Training loss value
            responsibilities: Responsibility matrix, shape (n_samples, n_experts)
            r_bar: Averaged responsibilities per expert (K,), Eq.23 before beta-smoothing
            r_k_mean: Per-sample r_k(z_n) mean over batch (K,), true Eq.22  [v1.7.0]
            r_k_std:  Per-sample r_k(z_n) std  over batch (K,), true Eq.22  [v1.7.0]
        """
        try:
            # Validate and store weights
            if weights.shape[0] != self.n_experts:
                logger.error(f"Weight shape mismatch: expected {self.n_experts}, got {weights.shape[0]}")
                return
            
            self.weights.append(weights.copy())
            
            # Compute Neff
            neff = compute_neff(weights)
            self.neff_history.append(neff)
            
            # Store loss if provided
            if loss is not None:
                if np.isnan(loss) or np.isinf(loss):
                    logger.warning(f"Invalid loss value: {loss}")
                    self.loss_history.append(np.nan)
                else:
                    self.loss_history.append(loss)
            
            # Compute and store entropy if responsibilities provided
            if responsibilities is not None:
                entropy = compute_entropy(responsibilities)
                self.entropy_history.append(entropy)

            # v1.2.0: store r_bar (K,) for resp-vs-weights plot
            if r_bar is not None:
                if r_bar.shape[0] != self.n_experts:
                    logger.error(f"r_bar shape mismatch: expected {self.n_experts}, got {r_bar.shape[0]}")
                else:
                    self.r_bar_history.append(r_bar.copy())

            # v1.7.0: store per-sample r_k mean and std (K,) for resp-vs-weights plot
            if r_k_mean is not None:
                if r_k_mean.shape[0] != self.n_experts:
                    logger.error(f"r_k_mean shape mismatch: expected {self.n_experts}, got {r_k_mean.shape[0]}")
                else:
                    self.r_k_mean_history.append(r_k_mean.copy())
            if r_k_std is not None:
                if r_k_std.shape[0] != self.n_experts:
                    logger.error(f"r_k_std shape mismatch: expected {self.n_experts}, got {r_k_std.shape[0]}")
                else:
                    self.r_k_std_history.append(r_k_std.copy())
                
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def to_dict(self) -> Dict:
        """Export metrics to dictionary."""
        return {
            'method': self.method_name,
            'n_experts': self.n_experts,
            'weights': np.array(self.weights),
            'neff': np.array(self.neff_history),
            'loss': np.array(self.loss_history),
            'entropy': np.array(self.entropy_history),
            'r_bar': np.array(self.r_bar_history),      # v1.2.0
            'r_k_mean': np.array(self.r_k_mean_history), # v1.7.0
            'r_k_std':  np.array(self.r_k_std_history),  # v1.7.0
        }
    
    def get_epochs(self) -> int:
        """Return number of recorded epochs."""
        return len(self.weights)


def compute_neff(weights: np.ndarray) -> float:
    """
    Compute effective number of experts.

    Formula: Neff = exp(H(π)) = exp(-Σ π_k log π_k)  — Shannon entropy (matches paper)

    Args:
        weights: Mixture weights, shape (n_experts,)
        
    Returns:
        Effective number of experts
    """
    try:
        # Handle NaN/Inf
        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
            logger.warning("NaN/Inf detected in weights for Neff computation")
            return np.nan
        
        # Ensure normalization
        weights_norm = weights / np.sum(weights)
        
        # Compute Neff via Shannon entropy: exp(H(pi)) — matches paper definition
        neff = float(np.exp(-np.sum(weights_norm * np.log(weights_norm + 1e-8))))
        
        return neff
        
    except Exception as e:
        logger.error(f"Error computing Neff: {e}")
        return np.nan


def compute_entropy(responsibilities: np.ndarray) -> float:
    """
    Compute mean entropy of responsibility distribution.
    
    H(r_i) = -Σ r_ik log(r_ik)
    
    Args:
        responsibilities: Responsibility matrix, shape (n_samples, n_experts)
        
    Returns:
        Mean entropy across samples
    """
    try:
        # Handle NaN/Inf
        if np.any(np.isnan(responsibilities)) or np.any(np.isinf(responsibilities)):
            logger.warning("NaN/Inf detected in responsibilities")
            return np.nan
        
        # Clip to avoid log(0)
        resp_clipped = np.clip(responsibilities, 1e-10, 1.0)
        
        # Compute entropy per sample
        entropy_per_sample = -np.sum(resp_clipped * np.log(resp_clipped), axis=1)
        
        # Return mean
        return np.mean(entropy_per_sample)
        
    except Exception as e:
        logger.error(f"Error computing entropy: {e}")
        return np.nan


def plot_weight_trajectories(tracker: MetricsTracker,
                            save_path: Optional[Union[str, Path]] = None,
                            figsize: tuple = (10, 6),
                            max_epochs: Optional[int] = None):
    """
    Plot A: Mixture weights over time.
    
    Args:
        tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        figsize: Figure size
        max_epochs: If set, plot only the first max_epochs epochs (v1.9.0)
    """
    try:
        weights = np.array(tracker.weights)  # Shape: (n_epochs, n_experts)
        if max_epochs is not None:
            weights = weights[:max_epochs]
        epochs = np.arange(len(weights))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each expert's weight trajectory
        for k in range(tracker.n_experts):
            ax.plot(epochs, weights[:, k], label=tracker.expert_names[k], linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('π_k', fontsize=12)
        ax.set_title(f'Mixture Weights Over Time - {tracker.method_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved weight trajectory plot to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting weight trajectories: {e}")


def plot_neff_evolution(tracker: MetricsTracker,
                       save_path: Optional[Union[str, Path]] = None,
                       figsize: tuple = (10, 6),
                       max_epochs: Optional[int] = None):
    """
    Plot B: Effective number of experts (Neff) vs epoch.
    
    Args:
        tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        figsize: Figure size
        max_epochs: If set, plot only the first max_epochs epochs (v1.9.0)
    """
    try:
        neff = np.array(tracker.neff_history)
        if max_epochs is not None:
            neff = neff[:max_epochs]
        epochs = np.arange(len(neff))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(epochs, neff, linewidth=2, color='#2E86AB')
        ax.axhline(y=tracker.n_experts, color='red', linestyle='--', 
                   label=f'Max (K={tracker.n_experts})', linewidth=1.5)
        ax.axhline(y=1, color='gray', linestyle='--', 
                   label='Collapse (Neff=1)', linewidth=1.5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Neff', fontsize=12)
        ax.set_title(f'Effective Number of Experts - {tracker.method_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.8, tracker.n_experts + 0.2)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved Neff evolution plot to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting Neff evolution: {e}")


def plot_training_objective(tracker: MetricsTracker,
                           save_path: Optional[Union[str, Path]] = None,
                           figsize: tuple = (10, 6),
                           max_epochs: Optional[int] = None):
    """
    Plot C: Training objective vs epoch.
    
    Args:
        tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        figsize: Figure size
        max_epochs: If set, plot only the first max_epochs epochs (v1.9.0)
    """
    try:
        if len(tracker.loss_history) == 0:
            logger.warning("No loss history to plot")
            return
        
        loss = np.array(tracker.loss_history)
        if max_epochs is not None:
            loss = loss[:max_epochs]
        epochs = np.arange(len(loss))
        
        # Filter out NaN values for plotting
        valid_mask = ~np.isnan(loss)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(epochs[valid_mask], loss[valid_mask], linewidth=2, color='#A23B72')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Training Objective', fontsize=12)
        ax.set_title(f'Training Objective (Mixture NLL / Posterior) - {tracker.method_name}', 
                    fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training objective plot to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting training objective: {e}")


def plot_responsibility_entropy(tracker: MetricsTracker,
                               save_path: Optional[Union[str, Path]] = None,
                               figsize: tuple = (10, 6),
                               max_epochs: Optional[int] = None):
    """
    Plot D: Mean responsibility entropy vs epoch.
    
    Args:
        tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        figsize: Figure size
        max_epochs: If set, plot only the first max_epochs epochs (v1.9.0)
    """
    try:
        if len(tracker.entropy_history) == 0:
            logger.warning("No entropy history to plot")
            return
        
        entropy = np.array(tracker.entropy_history)
        if max_epochs is not None:
            entropy = entropy[:max_epochs]
        epochs = np.arange(len(entropy))
        
        # Filter out NaN values
        valid_mask = ~np.isnan(entropy)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(epochs[valid_mask], entropy[valid_mask], linewidth=2, color='#F18F01')
        
        # Reference line for maximum entropy (uniform distribution)
        max_entropy = np.log(tracker.n_experts)
        ax.axhline(y=max_entropy, color='green', linestyle='--', 
                   label=f'Max Entropy (log K = {max_entropy:.2f})', linewidth=1.5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mean_i H(r_i)', fontsize=12)
        ax.set_title(f'Responsibility Entropy - {tracker.method_name}', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved responsibility entropy plot to {save_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting responsibility entropy: {e}")


def plot_responsibilities_vs_weights(tracker: MetricsTracker,
                                     save_path: Optional[Union[str, Path]] = None,
                                     figsize: tuple = (10, 6),
                                     max_epochs: Optional[int] = None):
    """
    Plot E: Averaged responsibilities (r_bar) vs mixture weights (pi) per expert over epochs.

    Solid lines  = r̄_k  (Eq.23 avg, before beta-smoothing)
    Dashed lines = π_k  (mixture weights after EMA)
    Dotted lines + shaded band = r_k(z_n) mean ± std over batch (true Eq.22)  [v1.7.0]
    Matches paper Fig.4 right panels.

    Args:
        tracker: MetricsTracker instance with r_bar_history populated
        save_path: Path to save figure (optional)
        figsize: Figure size
        max_epochs: If set, plot only the first max_epochs epochs (v1.6.0)
    """
    try:
        if len(tracker.r_bar_history) == 0:
            logger.warning("No r_bar history to plot — was r_bar passed to tracker.update()?")
            return

        r_bar = np.array(tracker.r_bar_history)   # (n_epochs, K)
        weights = np.array(tracker.weights)         # (n_epochs, K)

        # Align lengths in case of mismatch
        n_epochs = min(len(r_bar), len(weights))
        if len(r_bar) != len(weights):
            logger.warning(f"r_bar_history length {len(r_bar)} != weights length {len(weights)}; truncating to {n_epochs}")
        # v1.6.0: apply max_epochs cap after alignment
        if max_epochs is not None:
            n_epochs = min(n_epochs, max_epochs)
        r_bar   = r_bar[:n_epochs]
        weights = weights[:n_epochs]

        # v1.7.0: load per-sample r_k mean/std if available
        has_r_k = len(tracker.r_k_mean_history) > 0 and len(tracker.r_k_std_history) > 0
        if has_r_k:
            r_k_mean = np.array(tracker.r_k_mean_history)[:n_epochs]  # (n_epochs, K)
            r_k_std  = np.array(tracker.r_k_std_history)[:n_epochs]   # (n_epochs, K)
            if r_k_mean.shape[0] != n_epochs or r_k_std.shape[0] != n_epochs:
                logger.warning("r_k_mean/r_k_std length mismatch with n_epochs — skipping r_k overlay")
                has_r_k = False

        epochs = np.arange(n_epochs)
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax = plt.subplots(figsize=figsize)

        for k in range(tracker.n_experts):
            color = colors[k % len(colors)]
            name  = tracker.expert_names[k]
            # r̄_k solid line removed (v1.8.0) — still tracked in r_bar_history and exported to JSON
            # ax.plot(epochs, r_bar[:, k], color=color, linestyle='-', linewidth=1.5, label=f'r̄ {name}')
            # π_k — dashed (post-EMA weight)
            ax.plot(epochs, weights[:, k], color=color, linestyle='--', linewidth=1.5, label=f'π {name}')
            # r_k(z_n) mean ± std — dotted + shaded band  [v1.7.0]
            if has_r_k:
                ax.plot(epochs, r_k_mean[:, k], color=color, linestyle=':', linewidth=1.2,
                        label=f'rk {name}')
                ax.fill_between(epochs,
                                r_k_mean[:, k] - r_k_std[:, k],
                                r_k_mean[:, k] + r_k_std[:, k],
                                color=color, alpha=0.12)

        ax.set_xlabel('Stage-2 epoch', fontsize=12)
        ax.set_ylabel('Responsibility / Weight', fontsize=12)
        ax.set_title(f'Responsibilities vs Weights - {tracker.method_name}', fontsize=14)
        ncol = 2  # v1.8.0: 2 lines per expert (π dashed + rk dotted); r̄ removed
        ax.legend(ncol=ncol, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved responsibilities vs weights plot to {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Error plotting responsibilities vs weights: {e}")


def plot_weight_dynamics_neff(tracker: MetricsTracker,
                              save_path: Optional[Union[str, Path]] = None,
                              figsize: tuple = (10, 6),
                              max_epochs: Optional[int] = None):
    """
    Plot F: Stage-2 weight dynamics and Neff on dual y-axis.

    Left axis:  per-expert weight trajectories (solid, coloured)
    Right axis: Neff trajectory (gray dashed)
    Matches paper Fig.4 left panels.

    Args:
        tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        figsize: Figure size
        max_epochs: If set, plot only the first max_epochs epochs (v1.6.0)
    """
    try:
        weights = np.array(tracker.weights)      # (n_epochs, K)
        neff    = np.array(tracker.neff_history) # (n_epochs,)
        # v1.6.0: apply max_epochs cap
        if max_epochs is not None:
            weights = weights[:max_epochs]
            neff    = neff[:max_epochs]

        if len(weights) == 0:
            logger.warning("No weight history to plot")
            return

        epochs = np.arange(len(weights))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        fig, ax_w = plt.subplots(figsize=figsize)
        ax_n = ax_w.twinx()  # right axis for Neff

        # Left axis: per-expert weights (solid)
        for k in range(tracker.n_experts):
            color = colors[k % len(colors)]
            ax_w.plot(epochs, weights[:, k], color=color, linestyle='-',
                      linewidth=1.5, label=tracker.expert_names[k])

        # Right axis: Neff (gray dashed)
        if len(neff) > 0:
            ax_n.plot(epochs[:len(neff)], neff, color='gray', linestyle='--',
                      linewidth=1.5, label='N_eff')

        # Axis labels and limits
        ax_w.set_xlabel('Stage-2 epoch', fontsize=12)
        ax_w.set_ylabel(r'Weight \ $\pi_k$', fontsize=12)
        ax_n.set_ylabel(r'$N_{\mathrm{eff}}(t)$', fontsize=12)
        ax_w.set_ylim(0, 1)
        ax_n.set_ylim(1.4, tracker.n_experts + 0.2)

        ax_w.set_title('Stage-2 weight dynamics & N_eff', fontsize=14)

        # Combined legend from both axes
        lines_w, labels_w = ax_w.get_legend_handles_labels()
        lines_n, labels_n = ax_n.get_legend_handles_labels()
        ax_w.legend(lines_w + lines_n, labels_w + labels_n,
                    loc='upper right', fontsize=9)

        ax_w.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved weight dynamics & Neff plot to {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Error plotting weight dynamics & Neff: {e}")


def generate_all_plots(tracker: MetricsTracker,
                      output_dir: Union[str, Path] = "/home/benjamin/Documents/AMF-VIJ/results/convergence_plots",
                      prefix: str = "",
                      max_epochs: Optional[int] = None,
                      save_json: bool = True):
    """
    Generate all plot types and save to output directory.

    Args:
        tracker: MetricsTracker instance
        output_dir: Directory to save plots
        prefix: Prefix for filenames (e.g., dataset name)
        max_epochs: If set, _resp_weights and _weight_dynamics_neff plots show
                    only the first max_epochs epochs (v1.6.0)
        save_json: If True, export all tracker arrays to {prefix}_metrics.json (v1.8.0)
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename prefix
        file_prefix = f"{prefix}_{tracker.method_name}" if prefix else tracker.method_name
        
        # Generate all plots
        #plot_weight_trajectories(  # v1.10.0: replaced by plot_weight_dynamics_neff below
        #    tracker,
        #    save_path=output_dir / f"{file_prefix}_weights.png",
        #    max_epochs=max_epochs  # v1.9.0
        #)

        #plot_neff_evolution(
        #    tracker,
        #    save_path=output_dir / f"{file_prefix}_neff.png",
        #    max_epochs=max_epochs  # v1.9.0
        #)

        plot_weight_dynamics_neff(  # v1.10.0: active; dual y-axis weights + Neff(t)
            tracker,
            save_path=output_dir / f"{file_prefix}_weights.png",
            max_epochs=max_epochs  # v1.6.0
        )
        
        if len(tracker.loss_history) > 0:
            plot_training_objective(
                tracker,
                save_path=output_dir / f"{file_prefix}_loss.png",
                max_epochs=max_epochs  # v1.9.0
            )
        
        #if len(tracker.entropy_history) > 0:
        #    plot_responsibility_entropy(
        #        tracker,
        #        save_path=output_dir / f"{file_prefix}_entropy.png",
        #        max_epochs=max_epochs  # v1.9.0
        #    )

        if len(tracker.r_bar_history) > 0:  # v1.2.0
            plot_responsibilities_vs_weights(
                tracker,
                save_path=output_dir / f"{file_prefix}_resp_weights.png",
                max_epochs=max_epochs  # v1.6.0
            )

        # v1.8.0: export all tracker arrays to JSON for further analysis
        if save_json:
            import json
            metrics = tracker.to_dict()
            json_data = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in metrics.items()
            }
            json_path = output_dir / f"{file_prefix}_metrics.json"
            try:
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                logger.info(f"Metrics JSON saved to {json_path}")
            except Exception as je:
                logger.error(f"Failed to save metrics JSON: {je}")

        logger.info(f"Generated all plots for {tracker.method_name} in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating all plots: {e}")


# ---------------------------------------------------------------------------
# Display names for dataset axis labels (v1.5.0)
# ---------------------------------------------------------------------------
DISPLAY_NAMES = {
    'banana':         'Banana',
    'x_shape':        'X-Shaped',
    'bimodal_shared': 'Bimodal',
    'two_moons':      'Two Moons',
    'rings':          'Rings',
    'BLR':            'BLR',
    'BPR':            'BPR',
    'Weibull':        'Weibull',
    'Real-GMM2':      'Real-GMM2',
    'Iris-3Class':    'Iris-3Class',
}

# Fixed expert colors matching paper Fig.2 (v1.5.0)
EXPERT_COLORS = {
    'realnvp':         '#4C72B0',
    'maf':             '#DD8452',
    'rbig':            '#55A868',
    'gaussianization': '#55A868',  # alias for rbig
}
EXPERT_DISPLAY = {
    'realnvp': 'REALNVP',
    'maf':     'MAF',
    'rbig':    'RBIG',
    'gaussianization': 'RBIG',
}


def plot_mixture_weights_summary(
    dataset_weights: dict,
    expert_names: list,
    save_path: Union[str, Path],
) -> None:
    """
    Plot horizontal stacked bar chart of final mixture weights across all datasets.
    Matches paper Fig.2 style: one row per dataset, Neff annotation on right.

    Args:
        dataset_weights: dict mapping dataset_name -> (weights_array shape (K,), neff float)
        expert_names:    list of flow type strings e.g. ['realnvp', 'maf', 'rbig']
        save_path:       file path to save the PNG
    """
    if not dataset_weights:
        logger.error("plot_mixture_weights_summary: dataset_weights is empty — skipping plot")
        return

    try:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        datasets = list(dataset_weights.keys())
        n_datasets = len(datasets)
        n_experts  = len(expert_names)

        # Validate weight arrays
        for ds, (w, _) in dataset_weights.items():
            if len(w) != n_experts:
                logger.error(
                    f"plot_mixture_weights_summary: weight length {len(w)} != "
                    f"n_experts {n_experts} for dataset '{ds}' — skipping plot"
                )
                return

        fig, ax = plt.subplots(figsize=(10, max(4, 0.55 * n_datasets)))

        bar_height = 0.6
        y_positions = np.arange(n_datasets)

        # Legend order: RBIG / MAF / REALNVP (matches paper)
        legend_order = ['rbig', 'maf', 'realnvp', 'gaussianization']
        ordered_experts = sorted(
            range(n_experts),
            key=lambda i: legend_order.index(expert_names[i].lower())
            if expert_names[i].lower() in legend_order else i
        )

        # Build stacked bars
        lefts = np.zeros(n_datasets)
        handles = []
        for idx in ordered_experts:
            name = expert_names[idx].lower()
            color = EXPERT_COLORS.get(name, f'C{idx}')
            display = EXPERT_DISPLAY.get(name, expert_names[idx].upper())
            widths = np.array([dataset_weights[ds][0][idx] for ds in datasets])
            bars = ax.barh(
                y_positions, widths, left=lefts,
                height=bar_height, color=color, label=display
            )
            lefts += widths
            handles.append(bars)

        # Y-axis dataset labels
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            [DISPLAY_NAMES.get(ds, ds) for ds in datasets],
            fontsize=11
        )

        # Neff annotations on right
        for i, ds in enumerate(datasets):
            _, neff = dataset_weights[ds]
            ax.text(
                1.02, y_positions[i], f'N_eff={neff:.2f}',
                va='center', ha='left', fontsize=10,
                transform=ax.get_yaxis_transform()
            )

        ax.set_xlim(0, 1)
        ax.set_xlabel('Mixture weight', fontsize=12)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Legend above plot
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1,
                           color=EXPERT_COLORS.get(expert_names[i].lower(), f'C{i}'),
                           label=EXPERT_DISPLAY.get(expert_names[i].lower(),
                                                     expert_names[i].upper()))
            for i in ordered_experts
        ]
        ax.legend(
            handles=legend_handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.12),
            ncol=n_experts,
            frameon=False,
            fontsize=11
        )

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Mixture weights summary saved to {save_path}")

    except Exception as e:
        logger.error(f"plot_mixture_weights_summary failed: {e}")


# Example usage
if __name__ == "__main__":
    # Simulate tracking for demonstration
    logger.info("Running example usage...")
    
    # Create tracker
    tracker = MetricsTracker(method_name="DMAP-TAU", n_experts=3)
    
    # Simulate 100 epochs
    np.random.seed(42)
    for epoch in range(100):
        # Simulate converging weights
        base_weights = np.array([0.5, 0.3, 0.2])
        noise = np.random.randn(3) * 0.05 * (1 - epoch/100)
        weights = np.abs(base_weights + noise)
        weights /= weights.sum()
        
        # Simulate loss
        loss = 3.5 - 0.5 * (epoch / 100) + np.random.randn() * 0.1
        
        # Simulate responsibilities
        n_samples = 1000
        responsibilities = np.random.dirichlet(weights * 10, size=n_samples)
        
        # Update tracker
        tracker.update(weights, loss, responsibilities)
    
    # Generate all plots
    generate_all_plots(tracker, prefix="example_banana")
    
    logger.info("Example completed successfully")
