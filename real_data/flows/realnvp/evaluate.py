# VERSION: RNV-v1.0
# FILE: evaluate.py
# PURPOSE: Evaluation and sample generation
# DATE: 2025-01-20

import os
import sys
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

# Add parent directory to path for data_preprocessing import
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from realnvp import RealNVP  # Changed: relative import
from data_preprocessing import get_mnist_loaders, get_cifar10_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path, dataset='mnist', hidden_channels=64, num_blocks=8, num_scales=2, device='cuda'):
    """Load trained RealNVP model."""
    
    if dataset == 'mnist':
        input_shape = (1, 28, 28)
    elif dataset == 'cifar10':
        input_shape = (3, 32, 32)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    model = RealNVP(
        input_shape=input_shape,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks,
        num_scales=num_scales
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
    logger.info(f"Best BPD: {checkpoint.get('best_bpd', 'N/A')}")
    
    return model


@torch.no_grad()
def compute_test_bpd(model, test_loader, device='cuda'):
    """Compute bits per dimension on test set."""
    
    total_bpd = 0
    num_batches = 0
    
    for x, _ in tqdm(test_loader, desc="Computing BPD"):
        x = x.to(device)
        
        log_prob = model.log_prob(x)
        nll = -log_prob.mean()
        
        C, H, W = model.input_shape
        num_dims = C * H * W
        bpd = nll / (num_dims * np.log(2))
        
        total_bpd += bpd.item()
        num_batches += 1
    
    avg_bpd = total_bpd / num_batches
    logger.info(f"Test BPD: {avg_bpd:.4f}")
    
    return avg_bpd


@torch.no_grad()
def generate_samples(model, num_samples=64, device='cuda'):
    """Generate samples from the model."""
    
    samples = model.sample(num_samples, device=device)
    
    # Inverse preprocessing
    samples = torch.sigmoid(samples) * 256  # Reverse logit transform
    samples = torch.clamp(samples / 256.0, 0, 1)  # Clip and normalize to [0, 1]
    
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
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid_np.squeeze(), cmap='gray' if grid_np.shape[2] == 1 else None)
    plt.axis('off')
    plt.title('Generated Samples')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved visualization to {save_path}")
    
    plt.close()


@torch.no_grad()
def interpolate_latents(model, num_steps=8, device='cuda'):
    """Generate interpolation between two random latents."""
    
    C, H, W = model.input_shape
    
    # Compute final latent shape
    final_channels = C * (4 ** (model.num_scales - 1))
    final_height = H // (2 ** (model.num_scales - 1))
    final_width = W // (2 ** (model.num_scales - 1))
    
    # Sample two random latents
    z1 = torch.randn(1, final_channels, final_height, final_width).to(device)
    z2 = torch.randn(1, final_channels, final_height, final_width).to(device)
    
    # Interpolate
    alphas = torch.linspace(0, 1, num_steps).to(device)
    interpolations = []
    
    for alpha in alphas:
        z = (1 - alpha) * z1 + alpha * z2
        x, _ = model.forward(z, reverse=True)
        
        # Inverse preprocessing
        x = torch.sigmoid(x) * 256
        x = torch.clamp(x / 256.0, 0, 1)
        
        interpolations.append(x)
    
    return torch.cat(interpolations, dim=0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=8)
    parser.add_argument('--num_scales', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Load model
    logger.info("Loading model...")
    model = load_model(
        args.checkpoint,
        dataset=args.dataset,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        num_scales=args.num_scales,
        device=args.device
    )
    
    # Compute test BPD
    logger.info("Computing test BPD...")
    if args.dataset == 'mnist':
        _, test_loader = get_mnist_loaders(args.batch_size)
    else:
        _, test_loader = get_cifar10_loaders(args.batch_size)
    
    test_bpd = compute_test_bpd(model, test_loader, device=args.device)
    
    # Generate samples
    logger.info("Generating samples...")
    samples = generate_samples(model, num_samples=args.num_samples, device=args.device)
    
    results_dir = '/home/benjamin/Documents/AMF-VIJ/real_data/results/samples'
    save_path = os.path.join(results_dir, f'{args.dataset}_samples.png')
    visualize_samples(samples, save_path=save_path)
    
    # Generate interpolations
    logger.info("Generating interpolations...")
    interp = interpolate_latents(model, num_steps=8, device=args.device)
    interp_path = os.path.join(results_dir, f'{args.dataset}_interpolation.png')
    visualize_samples(interp, save_path=interp_path, nrow=8)
    
    logger.info("Evaluation complete!")
    logger.info(f"Final Test BPD: {test_bpd:.4f}")


# CHANGELOG
"""
v1.0 (2025-01-20):
- Test set BPD evaluation
- Sample generation from prior
- Latent interpolation
- Visualization and saving utilities
"""