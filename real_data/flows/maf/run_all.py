#!/usr/bin/env python3
# VERSION: MAF-v1.0
# FILE: run_all.py
# PURPOSE: Quick start script for training, evaluation, and comparison
# DATE: 2025-01-21

"""
Quick Start Script for MAF Implementation

Usage:
    python run_all.py --mode train    # Train both MNIST and CIFAR-10
    python run_all.py --mode eval     # Evaluate both models
    python run_all.py --mode compare  # Generate comparison report
    python run_all.py --mode all      # Do everything
"""

import sys
import argparse
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a command and log its execution."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}\n")
    
    try:
        result = subprocess.run(cmd, check=True)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with error code {e.returncode}")
        return False


def train_models(args):
    """Train MAF on MNIST and CIFAR-10."""
    logger.info("Starting training phase...")
    
    cmd = [
        sys.executable, 'train.py',
        '--auto',
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--device', args.device
    ]
    
    return run_command(cmd, "MAF Training (MNIST + CIFAR-10)")


def evaluate_models(args):
    """Evaluate trained models."""
    logger.info("Starting evaluation phase...")
    
    success = True
    
    for dataset in ['mnist', 'cifar10']:
        cmd = [
            sys.executable, 'evaluate.py',
            '--dataset', dataset,
            '--num_samples', str(args.num_samples),
            '--device', args.device
        ]
        
        success &= run_command(cmd, f"MAF Evaluation ({dataset.upper()})")
    
    return success


def compare_results():
    """Generate comparison report."""
    logger.info("Starting comparison phase...")
    
    cmd = [sys.executable, 'compare_results.py']
    success = run_command(cmd, "Results Comparison")
    
    # Also generate LaTeX table
    if success:
        cmd = [sys.executable, 'compare_results.py', '--latex']
        run_command(cmd, "LaTeX Table Generation")
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description='MAF Quick Start Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train both datasets with 50 epochs
  python run_all.py --mode train --epochs 50
  
  # Evaluate with custom sample count
  python run_all.py --mode eval --num_samples 100
  
  # Full pipeline
  python run_all.py --mode all --epochs 100
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'eval', 'compare', 'all'],
        help='Execution mode'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=64,
        help='Number of samples to generate (default: 64)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Execute based on mode
    success = True
    
    if args.mode in ['train', 'all']:
        success &= train_models(args)
    
    if args.mode in ['eval', 'all']:
        if success or args.mode == 'eval':
            success &= evaluate_models(args)
    
    if args.mode in ['compare', 'all']:
        if success or args.mode == 'compare':
            success &= compare_results()
    
    # Final summary
    logger.info("\n" + "="*60)
    if success:
        logger.info("✓ All operations completed successfully!")
        logger.info("\nResults available in:")
        logger.info("  - /home/claude/maf_results/mnist/")
        logger.info("  - /home/claude/maf_results/cifar10/")
        logger.info("  - /home/claude/maf_results/comparison_summary.txt")
    else:
        logger.error("✗ Some operations failed. Check logs above.")
    logger.info("="*60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())


# CHANGELOG
"""
v1.0 (2025-01-21):
- Quick start script for training, evaluation, and comparison
- Supports train, eval, compare, and all modes
- Configurable epochs, batch size, and sample count
- Automatic error handling and logging
"""