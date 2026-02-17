"""
Collapse Resistance & Convergence Speed Visualization
Abbreviation: CRCS-VIZ
Version: 1.0.0

CHANGELOG:
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
    """
    
    def __init__(self, method_name: str, n_experts: int):
        """
        Initialize metrics tracker.
        
        Args:
            method_name: Name of method (e.g., 'EMA', 'DMAP-TAU')
            n_experts: Number of expert flows
        """
        self.method_name = method_name
        self.n_experts = n_experts
        
        self.weights = []  # List of weight vectors per epoch
        self.neff_history = []
        self.loss_history = []
        self.entropy_history = []
        
        logger.info(f"Initialized MetricsTracker for {method_name} with {n_experts} experts")
    
    def update(self, 
               weights: np.ndarray,
               loss: Optional[float] = None,
               responsibilities: Optional[np.ndarray] = None):
        """
        Update metrics for current epoch.
        
        Args:
            weights: Current mixture weights, shape (n_experts,)
            loss: Training loss value
            responsibilities: Responsibility matrix, shape (n_samples, n_experts)
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
            'entropy': np.array(self.entropy_history)
        }
    
    def get_epochs(self) -> int:
        """Return number of recorded epochs."""
        return len(self.weights)


def compute_neff(weights: np.ndarray) -> float:
    """
    Compute effective number of experts.
    
    Formula: Neff = 1 / Σ(π_k²)
    
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
        
        # Compute Neff
        neff = 1.0 / np.sum(weights_norm ** 2)
        
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
                            figsize: tuple = (10, 6)):
    """
    Plot A: Mixture weights over time.
    
    Args:
        tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    try:
        weights = np.array(tracker.weights)  # Shape: (n_epochs, n_experts)
        epochs = np.arange(len(weights))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each expert's weight trajectory
        for k in range(tracker.n_experts):
            ax.plot(epochs, weights[:, k], label=f'Expert {k+1}', linewidth=2)
        
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
                       figsize: tuple = (10, 6)):
    """
    Plot B: Effective number of experts (Neff) vs epoch.
    
    Args:
        tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    try:
        neff = np.array(tracker.neff_history)
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
                           figsize: tuple = (10, 6)):
    """
    Plot C: Training objective vs epoch.
    
    Args:
        tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    try:
        if len(tracker.loss_history) == 0:
            logger.warning("No loss history to plot")
            return
        
        loss = np.array(tracker.loss_history)
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
                               figsize: tuple = (10, 6)):
    """
    Plot D: Mean responsibility entropy vs epoch.
    
    Args:
        tracker: MetricsTracker instance
        save_path: Path to save figure (optional)
        figsize: Figure size
    """
    try:
        if len(tracker.entropy_history) == 0:
            logger.warning("No entropy history to plot")
            return
        
        entropy = np.array(tracker.entropy_history)
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


def generate_all_plots(tracker: MetricsTracker,
                      output_dir: Union[str, Path] = "/home/benjamin/Documents/AMF-VIJ/results/convergence_plots",
                      prefix: str = ""):
    """
    Generate all 4 plot types and save to output directory.
    
    Args:
        tracker: MetricsTracker instance
        output_dir: Directory to save plots
        prefix: Prefix for filenames (e.g., dataset name)
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filename prefix
        file_prefix = f"{prefix}_{tracker.method_name}" if prefix else tracker.method_name
        
        # Generate all plots
        plot_weight_trajectories(
            tracker, 
            save_path=output_dir / f"{file_prefix}_weights.png"
        )
        
        plot_neff_evolution(
            tracker,
            save_path=output_dir / f"{file_prefix}_neff.png"
        )
        
        if len(tracker.loss_history) > 0:
            plot_training_objective(
                tracker,
                save_path=output_dir / f"{file_prefix}_loss.png"
            )
        
        if len(tracker.entropy_history) > 0:
            plot_responsibility_entropy(
                tracker,
                save_path=output_dir / f"{file_prefix}_entropy.png"
            )
        
        logger.info(f"Generated all plots for {tracker.method_name} in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating all plots: {e}")


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
