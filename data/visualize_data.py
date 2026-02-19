"""
visualize_data.py — AMF-VI Dataset Visualizer
Version: v1.1.0
Author: AMF-VI Project

Changelog (v1.1.0):
- Added version tracking header and VERSION constant
- Fixed garbled mu/sigma encoding in stats text box
- Added per-dataset n_samples override for slow MCMC real datasets (BLR, BPR, Weibull, Real-GMM2)
- Added error logging with timestamps for failed dataset generation
- Added progress print per dataset during grid plotting

Usage: python visualize_data.py [dataset_names...]
"""

# ── Version ─────────────────────────────────────────────────────────────────
VERSION = "v1.1.0"

import logging
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    from data_generator import generate_data, get_available_datasets
except ImportError as e:
    logger.error(f"Failed to import data_generator: {e}")
    sys.exit(1)

# ── Constants ────────────────────────────────────────────────────────────────
# Real Bayesian datasets use MCMC — cap at smaller n to avoid long waits
REAL_DATASETS = {"BLR", "BPR", "Weibull", "Real-GMM2"}
REAL_N_SAMPLES = 1000   # safe default for MCMC datasets
COLORS = [
    "steelblue", "crimson", "forestgreen", "darkorange",
    "mediumpurple", "chocolate", "teal", "slategray",
    "hotpink", "olive", "royalblue", "coral",
]


# ── Plot helpers ─────────────────────────────────────────────────────────────
def plot_dataset(data, title, ax=None, color="steelblue"):
    """Plot single dataset scatter with mean/std stats box."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    data_np = data.detach().cpu().numpy()
    ax.scatter(
        data_np[:, 0], data_np[:, 1],
        alpha=0.5, s=10, c=color,
        edgecolors="none",
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("X1", fontsize=9)
    ax.set_ylabel("X2", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # Fixed encoding: use unicode escapes to avoid garbling
    mean = data_np.mean(axis=0)
    std  = data_np.std(axis=0)
    stats_text = (
        f"\u03bc=({mean[0]:.2f}, {mean[1]:.2f})\n"
        f"\u03c3=({std[0]:.2f}, {std[1]:.2f})\n"
        f"n={len(data_np)}"
    )
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes, va="top", ha="left", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
    )
    return ax


def plot_all_datasets(datasets=None, n_samples=1000, save_path=None):
    """
    Plot all datasets in a grid.
    Real MCMC datasets automatically use REAL_N_SAMPLES to avoid slowdowns.
    """
    if datasets is None:
        datasets = get_available_datasets()

    n_datasets = len(datasets)
    n_cols = 4
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for i, dataset_name in enumerate(datasets):
        ax = axes_flat[i]
        # Use smaller n for slow MCMC datasets
        n = REAL_N_SAMPLES if dataset_name in REAL_DATASETS else n_samples
        label = f"[{i+1}/{n_datasets}] {dataset_name} (n={n})"
        logger.info(f"Generating {label}...")

        t0 = time.time()
        try:
            data = generate_data(dataset_name, n_samples=n)
            elapsed = time.time() - t0
            logger.info(f"  Done in {elapsed:.1f}s")
            color = COLORS[i % len(COLORS)]
            plot_dataset(data, dataset_name, ax, color)
        except Exception as e:
            logger.error(f"Failed to generate '{dataset_name}': {e}")
            ax.text(
                0.5, 0.5, f"Error:\n{dataset_name}\n{e}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="red",
            )
            ax.set_title(dataset_name, fontsize=10)

    # Hide unused subplots
    for j in range(n_datasets, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"AMF-VI Datasets  ({VERSION})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {e}")

    return fig


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    logger.info(f"visualize_data.py {VERSION}")

    parser = argparse.ArgumentParser(description="Visualize AMF-VI datasets")
    parser.add_argument("datasets", nargs="*", default=None,
                        help="Dataset names (default: all)")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Samples for synthetic datasets (default: 1000)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to file (e.g. datasets.png)")
    parser.add_argument("--list", action="store_true",
                        help="List available datasets and exit")
    args = parser.parse_args()

    available = get_available_datasets()

    if args.list:
        print(f"Available datasets ({len(available)}):")
        for name in available:
            tag = " [MCMC]" if name in REAL_DATASETS else ""
            print(f"  - {name}{tag}")
        return

    datasets = args.datasets if args.datasets else available

    # Validate
    invalid = [d for d in datasets if d not in available]
    if invalid:
        logger.error(f"Unknown datasets: {invalid}. Available: {available}")
        sys.exit(1)

    logger.info(f"Plotting {len(datasets)} datasets: {datasets}")
    fig = plot_all_datasets(datasets, args.n_samples, args.save)
    plt.show()


if __name__ == "__main__":
    main()