"""
Model classes for EM-style Mixture of Normalizing Flows.
Shared between training and evaluation scripts.
Updated with minimum weight constraint and stability improvements to prevent component collapse.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np


class AffineCouplingLayer(nn.Module):
    """Affine coupling layer for the flow with scale clamping for stability."""
    def __init__(self, dim: int, scale_clip: float = 10.0):
        super().__init__()
        self.dim = dim
        self.scale_clip = scale_clip  # Clamp scale to prevent explosion
        split = dim // 2
        
        # Scale network with careful initialization
        self.scale_net = nn.Sequential(
            nn.Linear(split, 64),
            nn.ReLU(),
            nn.Linear(64, dim - split)
        )
        
        # Initialize final layer with small weights for stability
        nn.init.normal_(self.scale_net[2].weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.scale_net[2].bias)
        
        # Translation network
        self.translate_net = nn.Sequential(
            nn.Linear(split, 64),
            nn.ReLU(),
            nn.Linear(64, dim - split)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        split = self.dim // 2
        x1, x2 = x[:, :split], x[:, split:]
        
        s = self.scale_net(x1)
        
        # CRITICAL: Clamp scale to prevent exponential explosion
        # This prevents exp(s) from becoming too large or too small
        s = torch.clamp(s, min=-self.scale_clip, max=self.scale_clip)
        
        t = self.translate_net(x1)
        
        y2 = x2 * torch.exp(s) + t
        # Add Output Clamping (Per-Layer)
        y2 = torch.clamp(y2, min=-100, max=100)  
        y = torch.cat([x1, y2], dim=1)
        
        log_det = torch.sum(s, dim=1)
        return y, log_det


class NormalizingFlow(nn.Module):
    """
    Normalizing flow component with affine coupling layers.
    """
    def __init__(self, dim: int, n_layers: int = 4, scale_clip: float = 10.0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim, scale_clip=scale_clip) for _ in range(n_layers)
        ])
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform z through the flow and compute log determinant."""
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det
    
    def log_prob(self, z: torch.Tensor, base_log_prob: torch.Tensor = None) -> torch.Tensor:
        """Compute log probability q_k(z|θ_k)."""
        x, log_det = self.forward(z)
        
        if base_log_prob is None:
            base_log_prob = -0.5 * torch.sum(x**2, dim=1) - \
                           0.5 * self.dim * np.log(2 * np.pi)
        
        return base_log_prob + log_det


class MixtureOfFlows:
    """
    Mixture of Normalizing Flows trained with EM-style algorithm.
    Includes minimum weight constraint and stability improvements.
    """
    def __init__(
        self,
        n_components: int,
        dim: int,
        n_flow_layers: int = 4,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        min_weight: float = 0.05,
        scale_clip: float = 10.0,
        gradient_clip: float = 5.0,
        l2_reg: float = 1e-5
    ):
        self.K = n_components
        self.dim = dim
        self.device = device
        self.min_weight = min_weight  # Minimum weight constraint (default 5%)
        self.gradient_clip = gradient_clip  # Gradient clipping (increased to 5.0)
        self.l2_reg = l2_reg  # L2 regularization strength
        
        self.flows = [
            NormalizingFlow(dim, n_flow_layers, scale_clip=scale_clip).to(device)
            for _ in range(n_components)
        ]
        
        self.mixture_weights = torch.ones(n_components, device=device) / n_components
        
        # Adam optimizer with weight decay for L2 regularization
        self.optimizers = [
            optim.Adam(flow.parameters(), lr=learning_rate, weight_decay=l2_reg)
            for flow in self.flows
        ]
        
        self.weight_history = []
        self.log_likelihood_history = []
    
    def compute_responsibilities(self, z_batch: torch.Tensor) -> torch.Tensor:
        """E-step: Compute responsibilities γ_k(z_i) for each component k."""
        B = z_batch.shape[0]
        log_responsibilities = torch.zeros(B, self.K, device=self.device)
        
        for k, flow in enumerate(self.flows):
            with torch.no_grad():
                log_q_k = flow.log_prob(z_batch)
                log_responsibilities[:, k] = torch.log(self.mixture_weights[k]) + log_q_k
        
        log_sum = torch.logsumexp(log_responsibilities, dim=1, keepdim=True)
        responsibilities = torch.exp(log_responsibilities - log_sum)
        
        return responsibilities
    
    def update_flow_parameters(self, z_batch: torch.Tensor, responsibilities: torch.Tensor):
        """M-step (part 1): Update flow parameters θ_k using responsibility-weighted gradient."""
        B = z_batch.shape[0]
        
        for k, (flow, optimizer) in enumerate(zip(self.flows, self.optimizers)):
            optimizer.zero_grad()
            
            log_q_k = flow.log_prob(z_batch)
            
            weighted_log_likelihood = torch.sum(
                responsibilities[:, k] * log_q_k
            ) / B
            
            loss = -weighted_log_likelihood
            
            # Add explicit L2 regularization on scale network parameters
            l2_loss = 0.0
            for layer in flow.layers:
                # Penalize large scale network weights
                for param in layer.scale_net.parameters():
                    l2_loss += torch.sum(param ** 2)
            
            # Combined loss (negative log-likelihood + L2 regularization)
            total_loss = loss + self.l2_reg * l2_loss
            total_loss.backward()
            
            # Gradient clipping (increased from 1.0 to 5.0)
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=self.gradient_clip)
            
            optimizer.step()
    
    def update_mixture_weights(self, responsibilities: torch.Tensor):
        """M-step (part 2): Update mixture weights π_k with minimum weight constraint."""
        B = responsibilities.shape[0]
        
        # Calculate new weights from responsibilities
        new_weights = torch.sum(responsibilities, dim=0) / B
        
        # Apply minimum weight constraint to prevent component collapse
        # Force all weights to stay above the minimum threshold
        self.mixture_weights = torch.clamp(new_weights, min=self.min_weight)
        
        # Renormalize to ensure weights sum to 1
        self.mixture_weights = self.mixture_weights / self.mixture_weights.sum()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch over all mini-batches."""
        total_log_likelihood = 0.0
        n_batches = 0
        
        for batch_idx, (z_batch,) in enumerate(dataloader):
            z_batch = z_batch.to(self.device)
            B = z_batch.shape[0]
            
            responsibilities = self.compute_responsibilities(z_batch)
            
            self.update_flow_parameters(z_batch, responsibilities)
            
            self.update_mixture_weights(responsibilities)
            
            log_lik = self.compute_log_likelihood(z_batch)
            total_log_likelihood += log_lik.sum().item()
            n_batches += B
            
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}: "
                      f"Weights = {self.mixture_weights.cpu().numpy()}, "
                      f"Avg Log-Lik = {log_lik.mean().item():.4f}")
        
        avg_log_likelihood = total_log_likelihood / n_batches
        return avg_log_likelihood
    
    def compute_log_likelihood(self, z_batch: torch.Tensor) -> torch.Tensor:
        """Compute log p(z) = log Σ_k[π_k * q_k(z|θ_k)] for monitoring."""
        B = z_batch.shape[0]
        log_mixture = torch.zeros(B, self.K, device=self.device)
        
        for k, flow in enumerate(self.flows):
            with torch.no_grad():
                log_q_k = flow.log_prob(z_batch)
                log_mixture[:, k] = torch.log(self.mixture_weights[k]) + log_q_k
        
        return torch.logsumexp(log_mixture, dim=1)
    
    def fit(self, dataloader: DataLoader, n_epochs: int, verbose: bool = True):
        """Complete training loop."""
        for epoch in range(n_epochs):
            avg_log_lik = self.train_epoch(dataloader, epoch)
            
            self.weight_history.append(self.mixture_weights.cpu().numpy().copy())
            self.log_likelihood_history.append(avg_log_lik)
            
            if verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch + 1}/{n_epochs}: "
                      f"Avg Log-Lik = {avg_log_lik:.4f}, "
                      f"Weights = {self.mixture_weights.cpu().numpy()}")
        
        if verbose:
            print(f"  Final loss: {avg_log_lik:.4f}")
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples from the learned mixture."""
        component_probs = self.mixture_weights.cpu().numpy()
        components = np.random.choice(self.K, size=n_samples, p=component_probs)
        
        samples = []
        for k in range(self.K):
            n_k = np.sum(components == k)
            if n_k > 0:
                z_base = torch.randn(n_k, self.dim, device=self.device)
                with torch.no_grad():
                    x_k, _ = self.flows[k](z_base)
                samples.append(x_k)
        
        return torch.cat(samples, dim=0)
