"""
Implementation Plan: EM-Style Training for Mixture of Normalizing Flows

This implementation follows a mini-batch EM algorithm where:
- E-step: Compute responsibilities (soft assignments) for each data point
- M-step: Update flow parameters and mixture weights based on responsibilities
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple
import numpy as np


class NormalizingFlow(nn.Module):
    """
    Base class for a normalizing flow component.
    Implements a bijective transformation with tractable likelihood.
    """
    def __init__(self, dim: int, n_layers: int = 4):
        super().__init__()
        self.dim = dim
        # Example: Stack of affine coupling layers
        self.layers = nn.ModuleList([
            AffineCouplingLayer(dim) for _ in range(n_layers)
        ])
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform z through the flow and compute log determinant.
        
        Args:
            z: Input samples [batch_size, dim]
        
        Returns:
            x: Transformed samples [batch_size, dim]
            log_det: Log determinant of Jacobian [batch_size]
        """
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, ld = layer(x)
            log_det += ld
        return x, log_det
    
    def log_prob(self, z: torch.Tensor, base_log_prob: torch.Tensor = None) -> torch.Tensor:
        """
        Compute log probability q_k(z|θ_k).
        
        Args:
            z: Input samples [batch_size, dim]
            base_log_prob: Optional precomputed base distribution log prob
        
        Returns:
            log_prob: Log probability [batch_size]
        """
        x, log_det = self.forward(z)
        
        # Base distribution (e.g., standard normal)
        if base_log_prob is None:
            base_log_prob = -0.5 * torch.sum(x**2, dim=1) - \
                           0.5 * self.dim * np.log(2 * np.pi)
        
        # Change of variables: log q(z) = log p(x) + log|det J|
        return base_log_prob + log_det


class AffineCouplingLayer(nn.Module):
    """Example coupling layer for the flow."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        split = dim // 2
        self.scale_net = nn.Sequential(
            nn.Linear(split, 64),
            nn.ReLU(),
            nn.Linear(64, dim - split)
        )
        self.translate_net = nn.Sequential(
            nn.Linear(split, 64),
            nn.ReLU(),
            nn.Linear(64, dim - split)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        split = self.dim // 2
        x1, x2 = x[:, :split], x[:, split:]
        
        s = self.scale_net(x1)
        t = self.translate_net(x1)
        
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=1)
        
        log_det = torch.sum(s, dim=1)
        return y, log_det


class MixtureOfFlows:
    """
    Mixture of Normalizing Flows trained with EM-style algorithm.
    """
    def __init__(
        self,
        n_components: int,
        dim: int,
        n_flow_layers: int = 4,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.K = n_components
        self.dim = dim
        self.device = device
        
        # Step 1: Initialize flow parameters θ_1, ..., θ_K
        self.flows = [
            NormalizingFlow(dim, n_flow_layers).to(device)
            for _ in range(n_components)
        ]
        
        # Step 2: Initialize mixture weights π_1, ..., π_K (uniform)
        self.mixture_weights = torch.ones(n_components, device=device) / n_components
        
        # Optimizers for each flow component
        self.optimizers = [
            optim.Adam(flow.parameters(), lr=learning_rate)
            for flow in self.flows
        ]
        
        # Track history
        self.weight_history = []
        self.log_likelihood_history = []
    
    def compute_responsibilities(
        self,
        z_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        E-step: Compute responsibilities γ_k(z_i) for each component k.
        
        γ_k(z_i) = π_k * q_k(z_i|θ_k) / Σ_j[π_j * q_j(z_i|θ_j)]
        
        Args:
            z_batch: Mini-batch of samples [B, dim]
        
        Returns:
            responsibilities: γ_k(z_i) for all k, i [B, K]
        """
        B = z_batch.shape[0]
        log_responsibilities = torch.zeros(B, self.K, device=self.device)
        
        # Compute log[π_k * q_k(z_i|θ_k)] for all k
        for k, flow in enumerate(self.flows):
            with torch.no_grad():
                log_q_k = flow.log_prob(z_batch)
                log_responsibilities[:, k] = torch.log(self.mixture_weights[k]) + log_q_k
        
        # Normalize using log-sum-exp trick for numerical stability
        # γ_k = exp(log_γ_k) / Σ_j exp(log_γ_j)
        log_sum = torch.logsumexp(log_responsibilities, dim=1, keepdim=True)
        responsibilities = torch.exp(log_responsibilities - log_sum)
        
        return responsibilities
    
    def update_flow_parameters(
        self,
        z_batch: torch.Tensor,
        responsibilities: torch.Tensor
    ):
        """
        M-step (part 1): Update flow parameters θ_k using responsibility-weighted gradient.
        
        grad_θ_k = Σ_i[γ_k(z_i) * ∇_θ_k log q_k(z_i|θ_k)] / B
        
        Args:
            z_batch: Mini-batch of samples [B, dim]
            responsibilities: γ_k(z_i) [B, K]
        """
        B = z_batch.shape[0]
        
        for k, (flow, optimizer) in enumerate(zip(self.flows, self.optimizers)):
            optimizer.zero_grad()
            
            # Compute log q_k(z_i|θ_k) with gradients
            log_q_k = flow.log_prob(z_batch)
            
            # Responsibility-weighted log likelihood
            # We want to maximize: Σ_i[γ_k(z_i) * log q_k(z_i|θ_k)]
            # Equivalent to minimizing the negative
            weighted_log_likelihood = torch.sum(
                responsibilities[:, k] * log_q_k
            ) / B
            
            loss = -weighted_log_likelihood
            loss.backward()
            optimizer.step()
    
    def update_mixture_weights(
        self,
        responsibilities: torch.Tensor
    ):
        """
        M-step (part 2): Update mixture weights π_k.
        
        π_k^new = Σ_i[γ_k(z_i)] / B
        Then normalize: π ← π^new / Σ_k[π_k^new]
        
        Args:
            responsibilities: γ_k(z_i) [B, K]
        """
        B = responsibilities.shape[0]
        
        # Average responsibility for each component
        new_weights = torch.sum(responsibilities, dim=0) / B
        
        # Normalize to ensure Σ_k π_k = 1
        self.mixture_weights = new_weights / torch.sum(new_weights)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
        """
        Train for one epoch over all mini-batches.
        
        Args:
            dataloader: DataLoader providing mini-batches
            epoch: Current epoch number
        
        Returns:
            avg_log_likelihood: Average log likelihood over epoch
        """
        total_log_likelihood = 0.0
        n_batches = 0
        
        for batch_idx, z_batch in enumerate(dataloader):
            z_batch = z_batch.to(self.device)
            B = z_batch.shape[0]
            
            # Step 1: E-step - Compute responsibilities
            responsibilities = self.compute_responsibilities(z_batch)
            
            # Step 2: M-step (part 1) - Update flow parameters
            self.update_flow_parameters(z_batch, responsibilities)
            
            # Step 3: M-step (part 2) - Update mixture weights
            self.update_mixture_weights(responsibilities)
            
            # Track log likelihood for monitoring
            log_lik = self.compute_log_likelihood(z_batch)
            total_log_likelihood += log_lik.sum().item()
            n_batches += B
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: "
                      f"Weights = {self.mixture_weights.cpu().numpy()}, "
                      f"Avg Log-Lik = {log_lik.mean().item():.4f}")
        
        avg_log_likelihood = total_log_likelihood / n_batches
        return avg_log_likelihood
    
    def compute_log_likelihood(
        self,
        z_batch: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log p(z) = log Σ_k[π_k * q_k(z|θ_k)] for monitoring.
        
        Args:
            z_batch: Samples [B, dim]
        
        Returns:
            log_likelihood: [B]
        """
        B = z_batch.shape[0]
        log_mixture = torch.zeros(B, self.K, device=self.device)
        
        for k, flow in enumerate(self.flows):
            with torch.no_grad():
                log_q_k = flow.log_prob(z_batch)
                log_mixture[:, k] = torch.log(self.mixture_weights[k]) + log_q_k
        
        return torch.logsumexp(log_mixture, dim=1)
    
    def fit(
        self,
        dataloader: DataLoader,
        n_epochs: int,
        verbose: bool = True
    ):
        """
        Complete training loop.
        
        Args:
            dataloader: DataLoader for training data
            n_epochs: Number of epochs to train
            verbose: Print progress
        """
        for epoch in range(n_epochs):
            avg_log_lik = self.train_epoch(dataloader, epoch)
            
            self.weight_history.append(self.mixture_weights.cpu().numpy().copy())
            self.log_likelihood_history.append(avg_log_lik)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{n_epochs} Complete")
                print(f"Average Log-Likelihood: {avg_log_lik:.4f}")
                print(f"Mixture Weights: {self.mixture_weights.cpu().numpy()}")
                print(f"{'='*60}\n")
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples from the learned mixture.
        
        Args:
            n_samples: Number of samples to generate
        
        Returns:
            samples: [n_samples, dim]
        """
        # Sample component assignments
        component_probs = self.mixture_weights.cpu().numpy()
        components = np.random.choice(self.K, size=n_samples, p=component_probs)
        
        samples = []
        for k in range(self.K):
            n_k = np.sum(components == k)
            if n_k > 0:
                # Sample from base distribution
                z_base = torch.randn(n_k, self.dim, device=self.device)
                # Transform through flow
                with torch.no_grad():
                    x_k, _ = self.flows[k](z_base)
                samples.append(x_k)
        
        return torch.cat(samples, dim=0)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Example usage of the mixture of flows implementation."""
    
    # Hyperparameters
    N_COMPONENTS = 3
    DIM = 2
    N_FLOW_LAYERS = 4
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 128
    N_EPOCHS = 50
    
    # Generate synthetic data (mixture of Gaussians)
    n_samples = 10000
    means = [[-3, -3], [0, 3], [3, -3]]
    data = []
    for mean in means:
        data.append(np.random.randn(n_samples // 3, 2) + mean)
    data = torch.FloatTensor(np.vstack(data))
    
    # Create dataloader
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize and train model
    model = MixtureOfFlows(
        n_components=N_COMPONENTS,
        dim=DIM,
        n_flow_layers=N_FLOW_LAYERS,
        learning_rate=LEARNING_RATE
    )
    
    print("Starting EM-style training for Mixture of Normalizing Flows...")
    print(f"Configuration: K={N_COMPONENTS}, dim={DIM}, "
          f"batch_size={BATCH_SIZE}, epochs={N_EPOCHS}")
    
    model.fit(dataloader, n_epochs=N_EPOCHS, verbose=True)
    
    # Generate samples from learned model
    generated_samples = model.sample(1000)
    print(f"\nGenerated {generated_samples.shape[0]} samples from learned mixture.")
    
    return model


if __name__ == "__main__":
    model = main()
