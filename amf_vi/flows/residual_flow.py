"""
Complete Residual Flow Implementation
=====================================

A comprehensive implementation of Residual Flows (Chen et al., 2019) with:
- Spectral normalization for Lipschitz constraints
- Fixed-point iteration for inversion
- Efficient log-determinant computation
- Training utilities and evaluation metrics
- Visualization tools

Reference: "Residual Flows for Invertible Generative Modeling"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Any
import math
from tqdm import tqdm
import warnings


# ============================================================================
# Core Components
# ============================================================================

class SpectralNorm(nn.Module):
    """
    Spectral normalization to enforce Lipschitz constraint.
    
    This ensures ||W||_2 <= 1 for weight matrix W, which is crucial for
    maintaining the contraction property in residual blocks.
    """
    
    def __init__(self, layer: nn.Module, n_power_iterations: int = 1):
        super().__init__()
        self.layer = layer
        self.n_power_iterations = n_power_iterations
        
        if hasattr(layer, 'weight'):
            w = layer.weight
            # Reshape weight to 2D matrix for spectral norm computation
            if w.dim() > 2:
                w_reshaped = w.view(w.size(0), -1)
            else:
                w_reshaped = w
            
            # Initialize u and v vectors for power iteration
            self.register_buffer('u', torch.randn(w_reshaped.size(0)))
            self.register_buffer('v', torch.randn(w_reshaped.size(1)))
    
    def _spectral_norm(self, w: torch.Tensor) -> torch.Tensor:
        """Compute spectral norm using power iteration method."""
        # Reshape weight to 2D
        if w.dim() > 2:
            w_mat = w.view(w.size(0), -1)
        else:
            w_mat = w
            
        u, v = self.u, self.v
        
        # Power iteration to find largest singular value
        for _ in range(self.n_power_iterations):
            # v = W^T u / ||W^T u||
            v = F.normalize(torch.mv(w_mat.t(), u), dim=0, eps=1e-12)
            # u = W v / ||W v||
            u = F.normalize(torch.mv(w_mat, v), dim=0, eps=1e-12)
        
        # Update buffers if training
        if self.training:
            self.u.copy_(u)
            self.v.copy_(v)
        
        # Compute spectral norm: σ = u^T W v
        sigma = torch.dot(u, torch.mv(w_mat, v))
        return sigma
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.layer, 'weight'):
            w = self.layer.weight
            sigma = self._spectral_norm(w)
            
            # Normalize weight by spectral norm: W := W / σ
            self.layer.weight.data = w / (sigma + 1e-6)
        
        return self.layer(x)


class LipschitzLinear(nn.Module):
    """Linear layer with Lipschitz constraint via spectral normalization."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.spectral_norm = SpectralNorm(self.linear)
        
        # Initialize weights using He initialization
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spectral_norm(x)


class ResidualBlock(nn.Module):
    """
    Residual block with spectral constraints for invertibility.
    
    Implements: y = x + α * g(x)
    where g(x) is a Lipschitz-constrained neural network and α < 1.
    
    The inverse is computed via fixed-point iteration:
    x = y - α * g(x)
    """
    
    def __init__(self, dim: int, hidden_dim: int = 64, n_layers: int = 3, 
                 activation: str = 'elu', init_scale: float = 0.9):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        
        # Build Lipschitz-constrained network
        layers = []
        current_dim = dim
        
        for i in range(n_layers):
            # Hidden layers
            if i < n_layers - 1:
                layers.extend([
                    LipschitzLinear(current_dim, hidden_dim),
                    self._get_activation(activation)
                ])
                current_dim = hidden_dim
            else:
                # Output layer
                layers.append(LipschitzLinear(current_dim, dim))
        
        self.net = nn.Sequential(*layers)
        
        # Learnable scaling factor α
        self.log_scale = nn.Parameter(torch.log(torch.tensor(init_scale)))
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function that preserves Lipschitz properties."""
        if activation.lower() == 'elu':
            return nn.ELU()
        elif activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            return nn.LeakyReLU(0.2)
        elif activation.lower() == 'swish':
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    @property
    def scale(self) -> torch.Tensor:
        """Get the scaling factor α = sigmoid(log_scale)."""
        return torch.sigmoid(self.log_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward transformation: y = x + α * g(x)."""
        residual = self.net(x)
        return x + self.scale * residual
    
    def inverse(self, y: torch.Tensor, n_iterations: int = 50, 
                tolerance: float = 1e-6) -> torch.Tensor:
        """
        Inverse using fixed-point iteration.
        
        Solves: x = y - α * g(x)
        """
        x = y.clone()
        
        for i in range(n_iterations):
            with torch.no_grad():
                residual = self.net(x)
                x_new = y - self.scale * residual
                
                # Check convergence
                diff = torch.max(torch.abs(x_new - x))
                if diff < tolerance:
                    break
                x = x_new
        else:
            warnings.warn(f"Fixed-point iteration did not converge after {n_iterations} iterations")
        
        return x
    
    def log_det_jacobian(self, x: torch.Tensor, method: str = 'exact') -> torch.Tensor:
        """
        Compute log-determinant of Jacobian matrix.
        
        For y = x + α * g(x), we have:
        J = I + α * J_g
        
        Methods:
        - 'exact': Compute full Jacobian (expensive)
        - 'trace': Use trace approximation (fast but approximate)
        - 'power_series': Power series approximation
        """
        if method == 'exact':
            return self._log_det_exact(x)
        elif method == 'trace':
            return self._log_det_trace_approx(x)
        elif method == 'power_series':
            return self._log_det_power_series(x)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _log_det_exact(self, x: torch.Tensor) -> torch.Tensor:
        """Exact log-determinant computation."""
        x = x.requires_grad_(True)
        y = self.forward(x)
        
        batch_size = x.size(0)
        log_det = torch.zeros(batch_size, device=x.device)
        
        for i in range(batch_size):
            jacobian = torch.zeros(self.dim, self.dim, device=x.device)
            
            for j in range(self.dim):
                grad = torch.autograd.grad(
                    y[i, j], x, retain_graph=True, create_graph=False
                )[0][i]
                jacobian[j] = grad
            
            # Add small diagonal term for numerical stability
            jacobian += 1e-6 * torch.eye(self.dim, device=x.device)
            log_det[i] = torch.logdet(jacobian)
        
        return log_det
    
    def _log_det_trace_approx(self, x: torch.Tensor) -> torch.Tensor:
        """Trace approximation: log|det(I + αJ_g)| ≈ α * trace(J_g)."""
        x = x.requires_grad_(True)
        residual = self.net(x)
        
        # Compute trace of Jacobian of g(x)
        trace_jac = torch.zeros(x.size(0), device=x.device)
        
        for i in range(self.dim):
            grad = torch.autograd.grad(
                residual[:, i].sum(), x, retain_graph=True, create_graph=False
            )[0]
            trace_jac += grad[:, i]
        
        # Approximation: log|det(I + αJ)| ≈ α * trace(J)
        return self.scale * trace_jac
    
    def _log_det_power_series(self, x: torch.Tensor, n_terms: int = 3) -> torch.Tensor:
        """Power series approximation for log-determinant."""
        # For small α, use: log|det(I + αJ)| ≈ α*tr(J) - α²/2*tr(J²) + α³/3*tr(J³) - ...
        # This is computationally intensive and rarely used in practice
        return self._log_det_trace_approx(x)  # Fallback to trace approximation


class ResidualFlow(nn.Module):
    """
    Complete Residual Flow model.
    
    Stacks multiple residual blocks to create an expressive normalizing flow.
    """
    
    def __init__(self, dim: int, n_blocks: int = 10, hidden_dim: int = 128,
                 n_layers_per_block: int = 4, activation: str = 'elu',
                 learnable_prior: bool = True):
        super().__init__()
        self.dim = dim
        self.n_blocks = n_blocks
        self.learnable_prior = learnable_prior
        
        # Create residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = ResidualBlock(
                dim=dim, 
                hidden_dim=hidden_dim,
                n_layers=n_layers_per_block,
                activation=activation
            )
            self.blocks.append(block)
        
        # Learnable prior parameters
        if learnable_prior:
            self.prior_mean = nn.Parameter(torch.zeros(dim))
            self.prior_log_std = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer('prior_mean', torch.zeros(dim))
            self.register_buffer('prior_log_std', torch.zeros(dim))
    
    def forward_and_log_det(self, x: torch.Tensor, 
                           log_det_method: str = 'trace') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through all residual blocks.
        
        Returns:
            z: Transformed samples in latent space
            log_det: Log-determinant of Jacobian
        """
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)
        
        for block in self.blocks:
            log_det = block.log_det_jacobian(z, method=log_det_method)
            z = block.forward(z)
            log_det_total += log_det
        
        return z, log_det_total
    
    def forward_only(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass without log-det computation for visualization."""
        z = x
        for block in self.blocks:
            z = block.forward(z)
        return z
    
    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse transformation from latent to data space."""
        x = z
        
        # Apply inverse blocks in reverse order
        for block in reversed(self.blocks):
            x = block.inverse(x)
        
        return x
    
    def log_prob(self, x: torch.Tensor, log_det_method: str = 'trace') -> torch.Tensor:
        """Compute log probability under the flow model."""
        z, log_det = self.forward_and_log_det(x, log_det_method)
        
        # Prior log probability
        if self.learnable_prior:
            prior_std = torch.exp(self.prior_log_std)
            log_prob_z = -0.5 * (((z - self.prior_mean) / prior_std)**2).sum(dim=1)
            log_prob_z -= 0.5 * z.size(1) * math.log(2 * math.pi)
            log_prob_z -= self.prior_log_std.sum()
        else:
            # Standard Gaussian prior
            log_prob_z = -0.5 * (z**2).sum(dim=1)
            log_prob_z -= 0.5 * z.size(1) * math.log(2 * math.pi)
        
        return log_prob_z + log_det
    
    def sample(self, n_samples: int, temperature: float = 1.0) -> torch.Tensor:
        """Sample from the flow model."""
        device = next(self.parameters()).device
        
        # Sample from prior
        if self.learnable_prior:
            prior_std = torch.exp(self.prior_log_std) * temperature
            z = torch.randn(n_samples, self.dim, device=device) * prior_std + self.prior_mean
        else:
            z = torch.randn(n_samples, self.dim, device=device) * temperature
        
        # Transform to data space
        return self.inverse(z)
    
    def get_block_scales(self) -> List[float]:
        """Get scaling factors of all blocks."""
        return [block.scale.item() for block in self.blocks]


# ============================================================================
# Training Utilities
# ============================================================================

class ResFlowTrainer:
    """Trainer class for Residual Flow models."""
    
    def __init__(self, model: ResidualFlow, optimizer: torch.optim.Optimizer,
                 device: torch.device = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, batch: torch.Tensor, regularization_weight: float = 0.01,
                   log_det_method: str = 'trace') -> float:
        """Single training step."""
        batch = batch.to(self.device)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Compute negative log-likelihood
        log_prob = self.model.log_prob(batch, log_det_method)
        nll_loss = -log_prob.mean()
        
        # Add regularization to keep scales reasonable
        scale_reg = sum(block.scale**2 for block in self.model.blocks)
        total_loss = nll_loss + regularization_weight * scale_reg
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return total_loss.item()
    
    def validate(self, val_data: torch.Tensor, log_det_method: str = 'trace') -> float:
        """Validation step."""
        val_data = val_data.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            log_prob = self.model.log_prob(val_data, log_det_method)
            val_loss = -log_prob.mean().item()
        
        return val_loss
    
    def train_epoch(self, train_data: torch.Tensor, batch_size: int = 100,
                    regularization_weight: float = 0.01, log_det_method: str = 'trace') -> float:
        """Train for one epoch."""
        n_samples = train_data.size(0)
        indices = torch.randperm(n_samples)
        epoch_losses = []
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = train_data[batch_indices]
            
            loss = self.train_step(batch, regularization_weight, log_det_method)
            epoch_losses.append(loss)
        
        return np.mean(epoch_losses)
    
    def train(self, train_data: torch.Tensor, n_epochs: int = 1000,
              batch_size: int = 100, val_data: Optional[torch.Tensor] = None,
              regularization_weight: float = 0.01, log_det_method: str = 'trace',
              verbose: bool = True, save_every: int = 100) -> Dict[str, List[float]]:
        """Full training loop."""
        
        best_val_loss = float('inf')
        best_model_state = None
        
        if verbose:
            pbar = tqdm(range(n_epochs), desc="Training ResFlow")
        else:
            pbar = range(n_epochs)
        
        for epoch in pbar:
            # Training
            train_loss = self.train_epoch(
                train_data, batch_size, regularization_weight, log_det_method
            )
            self.train_losses.append(train_loss)
            
            # Validation
            if val_data is not None:
                val_loss = self.validate(val_data, log_det_method)
                self.val_losses.append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
            
            # Progress update
            if verbose:
                desc = f"Train Loss: {train_loss:.4f}"
                if val_data is not None:
                    desc += f", Val Loss: {val_loss:.4f}"
                pbar.set_description(desc)
        
        # Restore best model if validation was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


# ============================================================================
# Evaluation and Testing
# ============================================================================

def test_invertibility(model: ResidualFlow, data: torch.Tensor, 
                      tolerance: float = 1e-2) -> Dict[str, Any]:
    """Test the invertibility of the flow model."""
    model.eval()
    
    with torch.no_grad():
        # Test subset for efficiency
        test_data = data[:100].to(next(model.parameters()).device)
        
        # Forward then inverse
        z, _ = model.forward_and_log_det(test_data)
        x_reconstructed = model.inverse(z)
        
        # Compute reconstruction errors
        abs_error = torch.abs(test_data - x_reconstructed)
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()
        std_error = abs_error.std().item()
        
        # Test individual blocks
        block_errors = []
        x_test = test_data[:10]
        
        for i, block in enumerate(model.blocks):
            y = block.forward(x_test)
            x_recovered = block.inverse(y)
            block_error = torch.abs(x_test - x_recovered).max().item()
            block_errors.append(block_error)
            x_test = y  # Chain for next block
    
    results = {
        'max_error': max_error,
        'mean_error': mean_error,
        'std_error': std_error,
        'block_errors': block_errors,
        'scales': model.get_block_scales(),
        'invertibility_passed': max_error < tolerance
    }
    
    return results


def compute_metrics(model: ResidualFlow, data: torch.Tensor) -> Dict[str, float]:
    """Compute various evaluation metrics."""
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    
    with torch.no_grad():
        # Log-likelihood
        log_prob = model.log_prob(data)
        nll = -log_prob.mean().item()
        
        # Sample from model
        n_samples = min(1000, data.size(0))
        samples = model.sample(n_samples)
        
        # Basic statistics comparison
        data_mean = data.mean(0).cpu().numpy()
        data_std = data.std(0).cpu().numpy()
        sample_mean = samples.mean(0).cpu().numpy()
        sample_std = samples.std(0).cpu().numpy()
        
        # Mean squared error between statistics
        mean_mse = np.mean((data_mean - sample_mean)**2)
        std_mse = np.mean((data_std - sample_std)**2)
    
    return {
        'negative_log_likelihood': nll,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'data_mean': data_mean.tolist(),
        'sample_mean': sample_mean.tolist(),
        'data_std': data_std.tolist(),
        'sample_std': sample_std.tolist()
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_flow_2d(model: ResidualFlow, data: torch.Tensor, 
                      save_path: Optional[str] = None, figsize: Tuple[int, int] = (15, 5)):
    """Visualize 2D flow results."""
    if data.shape[1] != 2:
        raise ValueError("This visualization is only for 2D data")
    
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    
    with torch.no_grad():
        # Generate samples
        samples = model.sample(1000)
        
        # Get latent representation
        z, _ = model.forward_and_log_det(data)
        
        # Convert to numpy
        data_np = data.cpu().numpy()
        samples_np = samples.cpu().numpy()
        z_np = z.cpu().numpy()
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original data
    axes[0].scatter(data_np[:, 0], data_np[:, 1], alpha=0.6, s=20, c='blue')
    axes[0].set_title('Original Data')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Generated samples
    axes[1].scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.6, s=20, c='red')
    axes[1].set_title('Generated Samples')
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    # Latent space
    axes[2].scatter(z_np[:, 0], z_np[:, 1], alpha=0.6, s=20, c='green')
    axes[2].set_title('Latent Space')
    axes[2].set_aspect('equal')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def plot_training_history(trainer: ResFlowTrainer, save_path: Optional[str] = None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Training loss
    axes[0].plot(trainer.train_losses, label='Training Loss', color='blue')
    if trainer.val_losses:
        axes[0].plot(trainer.val_losses, label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Block scales
    scales_history = []
    # This would require tracking scales during training
    # For now, just show final scales
    final_scales = trainer.model.get_block_scales()
    axes[1].bar(range(len(final_scales)), final_scales)
    axes[1].set_xlabel('Block Index')
    axes[1].set_ylabel('Scale Factor')
    axes[1].set_title('Final Block Scales')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# ============================================================================
# Example Usage and Demo
# ============================================================================

def demo_residual_flow():
    """Demonstrate Residual Flow on 2D datasets."""
    print("Residual Flow Demo")
    print("=" * 50)
    
    # Generate 2D moon dataset
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=2000, noise=0.1, random_state=42)
    X = torch.tensor(X, dtype=torch.float32)
    
    # Split into train/validation
    n_train = int(0.8 * len(X))
    train_data = X[:n_train]
    val_data = X[n_train:]
    
    print(f"Dataset: Two Moons")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    
    # Create model
    model = ResidualFlow(
        dim=2, 
        n_blocks=6, 
        hidden_dim=64, 
        n_layers_per_block=3,
        activation='elu',
        learnable_prior=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    trainer = ResFlowTrainer(model, optimizer)
    
    # Train model
    print("\nTraining model...")
    history = trainer.train(
        train_data=train_data,
        val_data=val_data,
        n_epochs=500,
        batch_size=64,
        regularization_weight=0.01,
        log_det_method='trace',
        verbose=True
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    
    # Test invertibility
    inv_results = test_invertibility(model, train_data, tolerance=1e-2)
    print(f"Invertibility test: {'PASSED' if inv_results['invertibility_passed'] else 'FAILED'}")
    print(f"Max reconstruction error: {inv_results['max_error']:.6f}")
    print(f"Mean reconstruction error: {inv_results['mean_error']:.6f}")
    
    # Compute metrics
    metrics = compute_metrics(model, val_data)
    print(f"Validation NLL: {metrics['negative_log_likelihood']:.4f}")
    print(f"Mean MSE: {metrics['mean_mse']:.6f}")
    print(f"Std MSE: {metrics['std_mse']:.6f}")
    
    # Block diagnostics
    scales = model.get_block_scales()
    print(f"Block scales: {[f'{s:.3f}' for s in scales]}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_flow_2d(model, val_data, save_path='resflow_demo_results.png')
    plot_training_history(trainer, save_path='resflow_demo_training.png')
    
    print("Demo completed!")
    
    return model, trainer, metrics




if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run demo
    model, trainer, metrics = demo_residual_flow()
