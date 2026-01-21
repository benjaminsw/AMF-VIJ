# VERSION: RNV-v1.0
# FILE: train.py
# PURPOSE: Training script with optimizer setup and logging
# DATE: 2025-01-20

import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import logging
from tqdm import tqdm
import numpy as np

# Add parent directory to path for data_preprocessing import
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from .realnvp import RealNVP  # Changed: relative import
from data_preprocessing import get_mnist_loaders, get_cifar10_loaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealNVPTrainer:
    """Trainer for RealNVP model."""
    
    def __init__(
        self,
        dataset='mnist',
        batch_size=64,
        hidden_channels=64,
        num_blocks=8,
        num_scales=2,
        lr=1e-3,
        weight_decay=5e-5,
        use_lr_decay=False,
        device='cuda'
    ):
        """
        Args:
            dataset: 'mnist' or 'cifar10'
            batch_size: Training batch size
            hidden_channels: Feature maps (32 or 64)
            num_blocks: Residual blocks (4 or 8)
            num_scales: Multi-scale levels (2 or 3)
            lr: Learning rate
            weight_decay: L2 regularization
            use_lr_decay: Whether to use learning rate decay
            device: 'cuda' or 'cpu'
        """
        self.dataset = dataset
        self.device = device
        self.use_lr_decay = use_lr_decay
        
        # Setup directories
        self.results_dir = '/home/benjamin/Documents/AMF-VIJ/real_data/results'
        self.checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        self.samples_dir = os.path.join(self.results_dir, 'samples')
        self.log_dir = os.path.join(self.results_dir, 'logs')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Load data
        logger.info(f"Loading {dataset} dataset...")
        if dataset == 'mnist':
            self.train_loader, self.test_loader = get_mnist_loaders(batch_size)
            input_shape = (1, 28, 28)
        elif dataset == 'cifar10':
            self.train_loader, self.test_loader = get_cifar10_loaders(batch_size)
            input_shape = (3, 32, 32)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Create model
        logger.info(f"Creating RealNVP model...")
        logger.info(f"  hidden_channels={hidden_channels}, num_blocks={num_blocks}, num_scales={num_scales}")
        
        self.model = RealNVP(
            input_shape=input_shape,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            num_scales=num_scales
        ).to(device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (optional)
        if use_lr_decay:
            self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
            logger.info("Using learning rate decay: step_size=50, gamma=0.5")
        else:
            self.scheduler = None
            logger.info(f"Using constant learning rate: {lr}")
        
        # Training state
        self.epoch = 0
        self.best_bpd = float('inf')
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def compute_loss(self, x):
        """
        Compute negative log-likelihood loss.
        
        Args:
            x: Batch of data [B, C, H, W]
            
        Returns:
            loss: Negative log-likelihood
            bpd: Bits per dimension
        """
        log_prob = self.model.log_prob(x)
        
        # Negative log-likelihood
        nll = -log_prob.mean()
        
        # Bits per dimension
        C, H, W = self.model.input_shape
        num_dims = C * H * W
        bpd = nll / (num_dims * np.log(2))
        
        return nll, bpd
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_bpd = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            
            try:
                # Forward pass
                nll, bpd = self.compute_loss(x)
                
                # Check for NaN/Inf
                if torch.isnan(nll) or torch.isinf(nll):
                    logger.error(f"NaN/Inf detected in loss at batch {batch_idx}")
                    logger.error(f"  x range: [{x.min():.2f}, {x.max():.2f}]")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                nll.backward()
                
                # Gradient clipping (optional safety)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += nll.item()
                total_bpd += bpd.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'NLL': f'{nll.item():.3f}',
                    'BPD': f'{bpd.item():.3f}'
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(self.train_loader)
        avg_bpd = total_bpd / len(self.train_loader)
        
        logger.info(f"Epoch {self.epoch} - Train NLL: {avg_loss:.3f}, BPD: {avg_bpd:.3f}")
        
        return avg_loss, avg_bpd
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0
        total_bpd = 0
        
        for x, _ in tqdm(self.test_loader, desc="Evaluating"):
            x = x.to(self.device)
            
            nll, bpd = self.compute_loss(x)
            
            total_loss += nll.item()
            total_bpd += bpd.item()
        
        avg_loss = total_loss / len(self.test_loader)
        avg_bpd = total_bpd / len(self.test_loader)
        
        logger.info(f"Epoch {self.epoch} - Test NLL: {avg_loss:.3f}, BPD: {avg_bpd:.3f}")
        
        return avg_loss, avg_bpd
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_bpd': self.best_bpd,
        }
        
        # Save latest
        path = os.path.join(self.checkpoint_dir, f'{self.dataset}_latest.pth')
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = os.path.join(self.checkpoint_dir, f'{self.dataset}_best.pth')
            torch.save(checkpoint, path)
            logger.info(f"✓ Saved best model (BPD: {self.best_bpd:.3f})")
    
    def train(self, num_epochs=100):
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss, train_bpd = self.train_epoch()
            
            # Evaluate
            test_loss, test_bpd = self.evaluate()
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                logger.info(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = test_bpd < self.best_bpd
            if is_best:
                self.best_bpd = test_bpd
            
            self.save_checkpoint(is_best=is_best)
            
            # Log to file
            log_path = os.path.join(self.log_dir, f'{self.dataset}_training.log')
            with open(log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.4f},{train_bpd:.4f},{test_loss:.4f},{test_bpd:.4f}\n")
        
        logger.info(f"Training complete! Best BPD: {self.best_bpd:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_channels', type=int, default=64, help='Feature maps (32 or 64)')
    parser.add_argument('--num_blocks', type=int, default=8, help='Residual blocks (4 or 8)')
    parser.add_argument('--num_scales', type=int, default=2, help='Multi-scale levels (2 or 3)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--use_lr_decay', action='store_true', help='Use learning rate decay')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = RealNVPTrainer(
        dataset=args.dataset,
        batch_size=args.batch_size,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        num_scales=args.num_scales,
        lr=args.lr,
        weight_decay=5e-5,
        use_lr_decay=args.use_lr_decay,
        device=args.device
    )
    
    # Train
    trainer.train(num_epochs=args.epochs)


# CHANGELOG
"""
v1.0 (2025-01-20):
- Complete training pipeline with optimizer setup
- Bits per dimension evaluation
- Checkpoint saving (latest + best)
- Learning rate scheduling (optional)
- Error logging for NaN/Inf detection
- Command-line arguments for experimentation
"""