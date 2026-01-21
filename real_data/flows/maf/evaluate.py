# VERSION: MAF-v1.0
# FILE: evaluate.py
# PURPOSE: Evaluation and sample generation for MAF
# DATE: 2025-01-21

import os
import sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from maf import MAF
from data_preprocessing import get_mnist_loaders, get_cifar10_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, dataset='mnist', hidden_dim=512, num_layers=5, 
               num_hidden=2, device='cuda'):
    """Load trained MAF model."""
    
    if dataset == 'mnist':
        input_dim = 28 * 28
        use_random_degrees = False
    elif dataset == 'cifar10':
        input_dim = 3 * 32 * 32
        use_random_degrees = True
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    model = MAF(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_hidden=num_hidden,
        use_random_degrees=use_random_degrees,
        use_batch_norm=True
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    logger.info(f"Best BPD: {checkpoint.get('best_bpd', 'N/A')}")
    
    return model


@torch.no_grad()
def compute_test_metrics(model, test_loader, num_dims, device='cuda'):
    """Compute NLL and BPD on test set."""
    
    total_nll = 0
    num_batches = 0
    
    for x, _ in tqdm(test_loader, desc="Computing metrics"):
        x = x.to(device)
        
        try:
            log_prob = model.log_prob(x)
            nll = -log_prob.mean()
            
            if not (torch.isnan(nll) or torch.isinf(nll)):
                total_nll += nll.item()
                num_batches += 1
        except RuntimeError as e:
            logger.error(f"Error in batch: {e}")
            continue
    
    if num_batches == 0:
        logger.error("No successful batches!")
        return None, None
    
    avg_nll = total_nll / num_batches
    avg_bpd = avg_nll / (num_dims * np.log(2))
    
    logger.info(f"Test NLL: {avg_nll:.4f}")
    logger.info(f"Test BPD: {avg_bpd:.4f}")
    
    return avg_nll, avg_bpd


@torch.no_grad()
def generate_samples(model, num_samples, dataset, device='cuda'):
    """
    Generate samples from the model.
    
    Note: This is slow because MAF requires D sequential passes.
    """
    
    logger.info(f"Generating {num_samples} samples (this may take a while)...")
    
    samples = model.sample(num_samples, device=device)
    
    # Inverse preprocessing
    samples = torch.sigmoid(samples) * 256  # Reverse logit transform
    samples = torch.clamp(samples / 256.0, 0, 1)  # Normalize to [0, 1]
    
    # Reshape for images
    if dataset == 'mnist':
        samples = samples.view(-1, 1, 28, 28)
    elif dataset == 'cifar10':
        samples = samples.view(-1, 3, 32, 32)
    
    return samples


def save_samples(samples, save_path, nrow=8):
    """Save generated samples as image grid."""
    
    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
    torchvision.utils.save_image(grid, save_path)
    logger.info(f"Saved samples to {save_path}")


def visualize_samples(samples, save_path=None, nrow=8):
    """Visualize generated samples."""
    
    grid = torchvision.utils.make_grid(samples, nrow=nrow, padding=2, normalize=False)
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    
    # Remove channel dimension if grayscale
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np, cmap='gray' if len(grid_np.shape) == 2 else None)
    plt.axis('off')
    plt.title('Generated Samples (MAF)')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved visualization to {save_path}")
    
    plt.close()


def save_results_summary(dataset, test_nll, test_bpd, save_path):
    """Save evaluation results to file."""
    
    with open(save_path, 'w') as f:
        f.write(f"MAF Evaluation Results - {dataset.upper()}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Test NLL: {test_nll:.6f}\n")
        f.write(f"Test BPD: {test_bpd:.6f}\n")
    
    logger.info(f"Saved results summary to {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'])
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to checkpoint (default: auto-find best.pth)')
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--num_layers', type=int, default=None)
    parser.add_argument('--num_hidden', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Auto-configure based on dataset
    if args.dataset == 'mnist':
        hidden_dim = args.hidden_dim or 512
        num_layers = args.num_layers or 5
        num_dims = 784
    else:  # cifar10
        hidden_dim = args.hidden_dim or 1024
        num_layers = args.num_layers or 10
        num_dims = 3072
    
    # Auto-find checkpoint if not provided
    if args.checkpoint is None:
        base_dir = f'/home/claude/maf_results/{args.dataset}'
        args.checkpoint = os.path.join(base_dir, 'checkpoints', 'best.pth')
        logger.info(f"Using checkpoint: {args.checkpoint}")
    
    # Load model
    logger.info("Loading model...")
    model = load_model(
        args.checkpoint,
        dataset=args.dataset,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_hidden=args.num_hidden,
        device=args.device
    )
    
    # Setup results directory
    results_dir = f'/home/claude/maf_results/{args.dataset}'
    samples_dir = os.path.join(results_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    
    # Compute test metrics
    logger.info("Computing test metrics...")
    if args.dataset == 'mnist':
        _, test_loader = get_mnist_loaders(args.batch_size)
    else:
        _, test_loader = get_cifar10_loaders(args.batch_size)
    
    test_nll, test_bpd = compute_test_metrics(model, test_loader, num_dims, device=args.device)
    
    if test_nll is not None:
        # Save results summary
        summary_path = os.path.join(results_dir, 'results_summary.txt')
        save_results_summary(args.dataset, test_nll, test_bpd, summary_path)
    
    # Generate samples
    logger.info("Generating samples...")
    samples = generate_samples(model, args.num_samples, args.dataset, device=args.device)
    
    # Save and visualize
    samples_path = os.path.join(samples_dir, f'{args.dataset}_samples.png')
    visualize_samples(samples, save_path=samples_path)
    
    logger.info("Evaluation complete!")
    if test_nll is not None:
        logger.info(f"Final Test NLL: {test_nll:.4f}")
        logger.info(f"Final Test BPD: {test_bpd:.4f}")


# CHANGELOG
"""
v1.0 (2025-01-21):
- Test set NLL and BPD evaluation
- Sample generation (slow due to sequential MAF inverse)
- Visualization and saving utilities
- Results summary export
- Auto-configuration per dataset
"""