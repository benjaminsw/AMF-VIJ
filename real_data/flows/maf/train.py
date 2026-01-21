# VERSION: MAF-v1.0
# FILE: train.py
# PURPOSE: Training script for MAF on MNIST and CIFAR-10
# DATE: 2025-01-21

import os
import sys
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import logging
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from maf import MAF
from data_preprocessing import get_mnist_loaders, get_cifar10_loaders

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MAFTrainer:
    """Trainer for MAF model."""
    
    def __init__(
        self,
        dataset='mnist',
        batch_size=64,
        hidden_dim=512,
        num_layers=5,
        num_hidden=2,
        lr=1e-4,
        weight_decay=5e-5,
        use_lr_decay=True,
        device='cuda'
    ):
        """
        Args:
            dataset: 'mnist' or 'cifar10'
            batch_size: Training batch size
            hidden_dim: Hidden units per MADE layer
            num_layers: Number of MADE layers
            num_hidden: Hidden layers per MADE
            lr: Learning rate
            weight_decay: L2 regularization
            use_lr_decay: Whether to use learning rate decay
            device: 'cuda' or 'cpu'
        """
        self.dataset = dataset
        self.device = device
        self.use_lr_decay = use_lr_decay
        
        # Setup directories
        base_dir = '/home/claude/maf_results'
        self.results_dir = os.path.join(base_dir, dataset)
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
            input_dim = 28 * 28
            use_random_degrees = False
        elif dataset == 'cifar10':
            self.train_loader, self.test_loader = get_cifar10_loaders(batch_size)
            input_dim = 3 * 32 * 32
            use_random_degrees = True  # For high-dim data
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # Create model
        logger.info(f"Creating MAF model...")
        logger.info(f"  input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, num_hidden={num_hidden}")
        
        self.model = MAF(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_hidden=num_hidden,
            use_random_degrees=use_random_degrees,
            use_batch_norm=True
        ).to(device)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        if use_lr_decay:
            self.scheduler = StepLR(self.optimizer, step_size=50, gamma=0.5)
            logger.info("Using learning rate decay: step_size=50, gamma=0.5")
        else:
            self.scheduler = None
            logger.info(f"Using constant learning rate: {lr}")
        
        # Training state
        self.epoch = 0
        self.best_bpd = float('inf')
        
        # Compute dimensionality for BPD
        if dataset == 'mnist':
            self.num_dims = 1 * 28 * 28
        else:
            self.num_dims = 3 * 32 * 32
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def compute_loss(self, x):
        """
        Compute negative log-likelihood loss.
        
        Args:
            x: Batch of data [B, C, H, W] or [B, D]
            
        Returns:
            loss: Negative log-likelihood
            bpd: Bits per dimension
        """
        log_prob = self.model.log_prob(x)
        
        # Negative log-likelihood
        nll = -log_prob.mean()
        
        # Bits per dimension
        bpd = nll / (self.num_dims * np.log(2))
        
        return nll, bpd
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_bpd = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(self.device)
            
            try:
                # Forward pass
                nll, bpd = self.compute_loss(x)
                
                # Check for NaN/Inf
                if torch.isnan(nll) or torch.isinf(nll):
                    logger.error(f"NaN/Inf detected at batch {batch_idx}")
                    logger.error(f"  x range: [{x.min():.2f}, {x.max():.2f}]")
                    continue
                
                # Backward pass
                self.optimizer.zero_grad()
                nll.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += nll.item()
                total_bpd += bpd.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'NLL': f'{nll.item():.3f}',
                    'BPD': f'{bpd.item():.3f}'
                })
                
            except RuntimeError as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        if num_batches == 0:
            logger.error("No successful batches in epoch!")
            return float('inf'), float('inf')
        
        avg_loss = total_loss / num_batches
        avg_bpd = total_bpd / num_batches
        
        logger.info(f"Epoch {self.epoch} - Train NLL: {avg_loss:.3f}, BPD: {avg_bpd:.3f}")
        
        return avg_loss, avg_bpd
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        total_loss = 0
        total_bpd = 0
        num_batches = 0
        
        for x, _ in tqdm(self.test_loader, desc="Evaluating"):
            x = x.to(self.device)
            
            try:
                nll, bpd = self.compute_loss(x)
                
                if not (torch.isnan(nll) or torch.isinf(nll)):
                    total_loss += nll.item()
                    total_bpd += bpd.item()
                    num_batches += 1
            except RuntimeError as e:
                logger.error(f"Error in evaluation: {e}")
                continue
        
        if num_batches == 0:
            logger.error("No successful batches in evaluation!")
            return float('inf'), float('inf')
        
        avg_loss = total_loss / num_batches
        avg_bpd = total_bpd / num_batches
        
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
        path = os.path.join(self.checkpoint_dir, 'latest.pth')
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best.pth')
            torch.save(checkpoint, path)
            logger.info(f"✓ Saved best model (BPD: {self.best_bpd:.3f})")
    
    def train(self, num_epochs=100):
        """Train the model."""
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        # Initialize log file
        log_path = os.path.join(self.log_dir, 'training.log')
        with open(log_path, 'w') as f:
            f.write("epoch,train_nll,train_bpd,test_nll,test_bpd\n")
        
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
            with open(log_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{train_bpd:.6f},"
                       f"{test_loss:.6f},{test_bpd:.6f}\n")
        
        logger.info(f"Training complete! Best BPD: {self.best_bpd:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto', action='store_true', help='Train both MNIST and CIFAR-10')
    parser.add_argument('--dataset', type=str, default=None, choices=['mnist', 'cifar10'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_hidden', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_lr_decay', action='store_true', default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Auto-run both datasets if --auto or no dataset specified
    if args.auto or args.dataset is None:
        configs = [
            {'dataset': 'mnist', 'hidden_dim': 512, 'num_layers': 5},
            {'dataset': 'cifar10', 'hidden_dim': 1024, 'num_layers': 10}
        ]
    else:
        configs = [{
            'dataset': args.dataset,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers
        }]
    
    # Train each configuration
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting training: {config['dataset'].upper()}")
        logger.info(f"{'='*60}\n")
        
        trainer = MAFTrainer(
            dataset=config['dataset'],
            batch_size=args.batch_size,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_hidden=args.num_hidden,
            lr=args.lr,
            weight_decay=5e-5,
            use_lr_decay=args.use_lr_decay,
            device=args.device
        )
        
        trainer.train(num_epochs=args.epochs)


# CHANGELOG
"""
v1.0 (2025-01-21):
- Training loop with NLL and BPD computation
- Automatic configuration for MNIST (512 hidden, 5 layers) and CIFAR-10 (1024 hidden, 10 layers)
- Gradient clipping and NaN detection
- Learning rate scheduling
- Checkpoint saving (best and latest)
- CSV logging for comparison
"""