"""
File: qualitative_density_contours.py | Version: 1.2.2 | Date: 2026-03-19
Abbreviation: QUAL-CONTOUR
Plan ID: IP-QUAL-CONTOUR-v1.0

Qualitative density contour plots matching paper Fig.3.
Produces an N_datasets x 6 grid:
    True | RealNVP | MAF | RBIG | AMF-VI | AMF-VI-sEMA

sEMA models loaded from:   sema_results_dir/trained_model_{ds}.pkl
AMF-VI models loaded from: amfvi_results_dir/trained_model_{ds}.pkl

flow_types order is read from model.flow_types (v2.7.0 attribute) — not hardcoded.

CHANGELOG:
- 1.2.2 (2026-03-19): Further tighten margins; strengthen OOD clipping via 3-sigma
  * MARGIN_FRAC reduced 0.05 → 0.02 for tighter axes on compact datasets
  * _draw_contour: new true_stats param (x_mean,x_std,y_mean,y_std); clips samples
    to mean±3σ of true data instead of grid bounds — stronger OOD rejection
  * true_stats computed after true_samples load; passed to all 5 _draw_contour calls
- 1.2.1 (2026-03-19): Fix small plots and RealNVP overflow on real targets
  * MARGIN_FRAC reduced 0.15 → 0.05 for tighter axes on compact-support datasets
  * _draw_contour: clips samples to grid bounds before KDE — prevents poorly-fitted
    models (e.g. RealNVP on BLR/BPR/Weibull) producing flat blobs filling the cell
  * Shows 'out of bounds' label if <10 samples remain after clipping
- 1.2.0 (2026-03-19): Use full dataset (train+val+test) for true_samples; chunked KDE
  * true_samples now concatenates all three splits (~100k) instead of test only (~20k)
  * Removed n_samples cap on true_samples; cap retained for model .sample() calls only
  * _kde_grid: evaluates positions in CHUNK_SIZE=10_000 chunks to avoid OOM
  * N_SAMPLES renamed to N_MODEL_SAMPLES for clarity; added CHUNK_SIZE=10_000 constant
- 1.1.3 (2026-03-18): Fix DEFAULT_AMFVI_RESULTS_DIR to point to main/results
  * Changed results_AMF_VI_C -> main/results; AMF-VI pkl files located there
  * No fallback — FileNotFoundError raised if pkl missing (existing behaviour)
- 1.1.2 (2026-03-18): Fix pickle deserialization — register 'SEMA_MBATCH_vis' in sys.modules
  * Added sys.modules['SEMA_MBATCH_vis'] = sys.modules['main.unit_test.SEMA_MBATCH_vis']
  * Resolves 'Can't get attribute SequentialAMFVI' error on pickle.load for all datasets
- 1.1.1 (2026-03-14): Fix DEFAULT_SEMA_RESULTS_DIR to match SEMA-MBATCH-vis save path
  * Changed os.path.join(BASE_DIR, 'results') -> os.path.join(BASE_DIR, 'main', 'results')
  * Resolves to /home/benjamin/Documents/AMF-VIJ/main/results where pkl files are saved
- 1.1.0 (2026-03-14): Align dataset list with SEMA-MBATCH-vis v2.6.0
  * CANONICAL_DATASETS: added 'multimodal-5', removed 'Iris-3Class'
  * DISPLAY_NAMES: added 'multimodal-5' -> 'Multimodal-5', removed 'Iris-3Class' entry
  * No logic changes; purely dataset-scope alignment
- 1.0.0 (2026-03-12): Initial implementation
  * plot_qualitative_contours(): main entry point, accepts any subset of canonical datasets
  * _load_pickle(): loads pkl, raises FileNotFoundError (no silent fallback)
  * _get_flow_samples(): samples from individual flow by name via model.flow_types index
  * _kde_grid(): scipy gaussian_kde on shared grid bounds per row
  * _draw_contour(): contourf + contour, Blues cmap, 10 levels; blank axis on failure
  * CLI: python qualitative_density_contours.py [ds1 ds2 ...] [--sema DIR] [--amfvi DIR]
  * Missing sEMA pkl -> logging.error + skip row; missing AMF-VI pkl -> blank col with N/A
"""

import os
import sys
import logging
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde

# Required for unpickling SequentialAMFVI
from main.unit_test.SEMA_MBATCH_vis import SequentialAMFVI, train_sequential_amf_vi  # noqa: F401
import main.unit_test.SEMA_MBATCH_vis
sys.modules['SEMA_MBATCH_vis'] = sys.modules['main.unit_test.SEMA_MBATCH_vis']

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] QUAL-CONTOUR: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_SEMA_RESULTS_DIR  = os.path.join(BASE_DIR, 'main', 'results')
DEFAULT_AMFVI_RESULTS_DIR = os.path.join(BASE_DIR, 'main', 'results')

CANONICAL_DATASETS = [
    'banana', 'x_shape', 'bimodal_shared', 'two_moons', 'rings',
    'multimodal-5', 'BLR', 'BPR', 'Weibull', 'Real-GMM2',
]

DISPLAY_NAMES = {
    'banana':         'Banana',
    'x_shape':        'X-Shaped',
    'bimodal_shared': 'Bimodal',
    'two_moons':      'Two Moons',
    'rings':          'Rings',
    'multimodal-5':   'Multimodal-5',
    'BLR':            'BLR',
    'BPR':            'BPR',
    'Weibull':        'Weibull',
    'Real-GMM2':      'Real-GMM2',
}

FLOW_DISPLAY_NAMES = {
    'realnvp':         'RealNVP',
    'maf':             'MAF',
    'rbig':            'RBIG',
    #'gaussianization': 'RBIG',
}

COL_HEADERS = ['True', 'RealNVP', 'MAF', 'RBIG', 'AMF-VI', 'AMF-VI-sEMA']
N_COLS = len(COL_HEADERS)

CONTOUR_LEVELS  = 10
CMAP            = 'Blues'
N_MODEL_SAMPLES = 5000   # samples drawn from each model for KDE
CHUNK_SIZE      = 10_000 # KDE grid evaluation chunk size to avoid OOM
N_GRID          = 200
MARGIN_FRAC     = 0.02   # fractional axis margin around data (reduced from 0.05)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_pickle(results_dir: str, dataset_name: str) -> dict:
    """
    Load trained_model_{dataset_name}.pkl from results_dir.
    Raises FileNotFoundError if missing — no silent fallback.
    """
    path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pickle not found: {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def _get_flow_index(model, flow_name: str) -> int:
    """
    Resolve flow index from model.flow_types by name (case-insensitive).
    Raises ValueError if not found.
    """
    if not hasattr(model, 'flow_types'):
        raise ValueError("model.flow_types attribute missing — pkl may be from pre-v2.7.0")
    for i, ft in enumerate(model.flow_types):
        if ft.lower() == flow_name.lower() or \
           FLOW_DISPLAY_NAMES.get(ft.lower(), '').lower() == flow_name.lower():
            return i
    raise ValueError(
        f"Flow '{flow_name}' not found in model.flow_types={model.flow_types}"
    )


def _sample_flow(model, flow_name: str, n: int) -> np.ndarray:
    """
    Sample n points from the named individual flow within the mixture model.
    Returns (n, 2) numpy array.
    """
    import torch
    idx = _get_flow_index(model, flow_name)
    flow = model.flows[idx]
    flow.eval()
    with torch.no_grad():
        samples = flow.sample(n)
    arr = samples.detach().cpu().numpy()
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _sample_mixture(model, n: int) -> np.ndarray:
    """Sample n points from the full mixture (uses model.sample())."""
    import torch
    model.eval()
    with torch.no_grad():
        samples = model.sample(n)
    arr = samples.detach().cpu().numpy() if hasattr(samples, 'detach') else np.array(samples)
    return arr


def _compute_grid_bounds(true_samples: np.ndarray, margin: float = MARGIN_FRAC):
    """Compute shared x/y grid bounds from true samples with margin."""
    x_min, x_max = true_samples[:, 0].min(), true_samples[:, 0].max()
    y_min, y_max = true_samples[:, 1].min(), true_samples[:, 1].max()
    x_rng = x_max - x_min
    y_rng = y_max - y_min
    return (
        x_min - margin * x_rng, x_max + margin * x_rng,
        y_min - margin * y_rng, y_max + margin * y_rng,
    )


def _kde_grid(samples: np.ndarray, x_min, x_max, y_min, y_max,
              n_grid: int = N_GRID) -> tuple:
    """
    Evaluate gaussian_kde on a regular grid in chunks to avoid OOM.
    Returns (xx, yy, zz) for contour plotting.
    """
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, n_grid),
        np.linspace(y_min, y_max, n_grid),
    )
    positions = np.vstack([xx.ravel(), yy.ravel()])
    kde = gaussian_kde(samples[:, :2].T)
    # Evaluate in chunks to avoid OOM on large grids
    n_positions = positions.shape[1]
    zz_flat = np.empty(n_positions)
    for start in range(0, n_positions, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_positions)
        zz_flat[start:end] = kde(positions[:, start:end])
    zz = zz_flat.reshape(xx.shape)
    return xx, yy, zz


def _draw_contour(ax, samples: np.ndarray, x_min, x_max, y_min, y_max,
                  n_grid: int = N_GRID, label: str = '',
                  true_stats: tuple = None):
    """
    Draw filled + line contours on ax from samples.
    Clips samples to mean±3σ of true data (if true_stats provided) before KDE
    to prevent out-of-distribution overflow (e.g. RealNVP on real targets).
    Falls back to grid bounds clip if true_stats not provided.
    Logs error and leaves axis blank on failure — no crash.

    Args:
        true_stats: (x_mean, x_std, y_mean, y_std) from true_samples
    """
    try:
        # Clip to mean±3σ of true data to exclude OOD samples
        if true_stats is not None:
            x_mean, x_std, y_mean, y_std = true_stats
            x_lo = x_mean - 3 * x_std
            x_hi = x_mean + 3 * x_std
            y_lo = y_mean - 3 * y_std
            y_hi = y_mean + 3 * y_std
        else:
            x_lo, x_hi, y_lo, y_hi = x_min, x_max, y_min, y_max

        samples = samples[
            (samples[:, 0] >= x_lo) & (samples[:, 0] <= x_hi) &
            (samples[:, 1] >= y_lo) & (samples[:, 1] <= y_hi)
        ]
        if len(samples) < 10:
            logger.error(f"Too few samples inside clip bounds for '{label}' ({len(samples)})")
            ax.text(0.5, 0.5, 'out of\nbounds', transform=ax.transAxes,
                    ha='center', va='center', color='red', fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            return
        xx, yy, zz = _kde_grid(samples, x_min, x_max, y_min, y_max, n_grid)
        ax.contourf(xx, yy, zz, levels=CONTOUR_LEVELS, cmap=CMAP, alpha=0.85)
        ax.contour(xx, yy, zz, levels=CONTOUR_LEVELS, colors='black',
                   linewidths=0.4, alpha=0.7)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
    except Exception as e:
        logger.error(f"KDE/contour failed for '{label}': {e}")
        ax.text(0.5, 0.5, 'KDE\nerror', transform=ax.transAxes,
                ha='center', va='center', color='red', fontsize=8)

    ax.set_xticks([])
    ax.set_yticks([])


def _blank_axis(ax, message: str = 'N/A'):
    """Mark an axis as unavailable."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha='center', va='center', color='gray', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def plot_qualitative_contours(
    datasets: list,
    sema_results_dir: str  = DEFAULT_SEMA_RESULTS_DIR,
    amfvi_results_dir: str = DEFAULT_AMFVI_RESULTS_DIR,
    save_path: str         = None,
    n_samples: int         = N_MODEL_SAMPLES,
    n_grid: int            = N_GRID,
):
    """
    Generate paper Fig.3-style qualitative density contour grid.

    Args:
        datasets:          list of dataset names to plot (rows); any subset of CANONICAL_DATASETS
        sema_results_dir:  directory containing AMF-VI-sEMA pkl files
        amfvi_results_dir: directory containing AMF-VI pkl files (results_AMF_VI_C)
        save_path:         output PNG path; defaults to sema_results_dir/qualitative_contours.png
        n_samples:         samples per model per dataset for KDE
        n_grid:            KDE evaluation grid resolution
    """
    # ----- validate -----
    if not datasets:
        logger.error("plot_qualitative_contours: datasets list is empty — nothing to plot")
        return

    if save_path is None:
        save_path = os.path.join(sema_results_dir, 'qualitative_contours.png')

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(datasets)
    fig_h  = max(3, 2.5 * n_rows)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 2.8, fig_h))

    # Ensure axes is always 2D
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers (top row only)
    for col_idx, header in enumerate(COL_HEADERS):
        axes[0, col_idx].set_title(header, fontsize=10, fontweight='normal', pad=4)

    # ----- data import (deferred to avoid circular imports at module level) -----
    try:
        from data.data_cache import get_split_data
    except ImportError as e:
        logger.error(f"Cannot import get_split_data: {e}")
        return

    # ----- per-dataset row -----
    for row_idx, ds in enumerate(datasets):
        row_axes = axes[row_idx]
        ds_label = DISPLAY_NAMES.get(ds, ds)
        logger.info(f"Processing row {row_idx+1}/{n_rows}: {ds_label}")

        # Row y-label (dataset name on left)
        row_axes[0].set_ylabel(ds_label, fontsize=10, rotation=0,
                               labelpad=55, va='center')

        # -- load sEMA model (required; skip row if missing) --
        try:
            sema_pkl  = _load_pickle(sema_results_dir, ds)
            sema_model = sema_pkl['model']
            sema_model.eval()
        except FileNotFoundError as e:
            logger.error(f"sEMA pkl missing for '{ds}' — skipping row: {e}")
            for ax in row_axes:
                _blank_axis(ax, 'sEMA\nnot found')
            continue
        except Exception as e:
            logger.error(f"Failed to load sEMA pkl for '{ds}': {e}")
            for ax in row_axes:
                _blank_axis(ax, 'load\nerror')
            continue

        # -- read flow_types from model (recommendation: not hardcoded) --
        if not hasattr(sema_model, 'flow_types'):
            logger.error(
                f"model.flow_types missing for '{ds}' (pre-v2.7.0 pkl?) "
                "— cannot resolve flow column order; skipping row"
            )
            for ax in row_axes:
                _blank_axis(ax, 'flow_types\nmissing')
            continue

        flow_types = sema_model.flow_types  # e.g. ['realnvp', 'maf', 'rbig']

        # -- load AMF-VI model (optional; blank col if missing) --
        amfvi_model = None
        try:
            amfvi_pkl   = _load_pickle(amfvi_results_dir, ds)
            amfvi_model = amfvi_pkl['model']
            amfvi_model.eval()
        except FileNotFoundError:
            logger.error(
                f"AMF-VI pkl missing for '{ds}' in {amfvi_results_dir} "
                "— AMF-VI column will be blank"
            )
        except Exception as e:
            logger.error(f"Failed to load AMF-VI pkl for '{ds}': {e}")

        # -- load true samples (all splits: train + val + test) --
        try:
            split_data = get_split_data(ds)
            def _to_np(t):
                return t.detach().cpu().numpy() if hasattr(t, 'detach') else np.array(t)
            true_samples = np.concatenate([
                _to_np(split_data['train']),
                _to_np(split_data['val']),
                _to_np(split_data['test']),
            ], axis=0)
        except Exception as e:
            logger.error(f"Failed to load true samples for '{ds}': {e}")
            for ax in row_axes:
                _blank_axis(ax, 'data\nerror')
            continue

        # Shared grid bounds derived from true data
        x_min, x_max, y_min, y_max = _compute_grid_bounds(true_samples)
        true_stats = (
            float(true_samples[:, 0].mean()), float(true_samples[:, 0].std()),
            float(true_samples[:, 1].mean()), float(true_samples[:, 1].std()),
        )

        # -- col 0: True --
        _draw_contour(row_axes[0], true_samples,
                      x_min, x_max, y_min, y_max, n_grid, label=f'{ds}/True',
                      true_stats=true_stats)

        # -- cols 1,2,3: individual flows (RealNVP, MAF, RBIG) --
        individual_flows = ['realnvp', 'maf', 'rbig']
        for col_offset, flow_name in enumerate(individual_flows):
            col_idx = col_offset + 1  # cols 1,2,3
            ax = row_axes[col_idx]
            try:
                samples = _sample_flow(sema_model, flow_name, n_samples)
                _draw_contour(ax, samples, x_min, x_max, y_min, y_max, n_grid,
                              label=f'{ds}/{flow_name}', true_stats=true_stats)
            except ValueError as e:
                # flow not present in this model
                logger.error(f"Flow '{flow_name}' not found in sEMA model for '{ds}': {e}")
                _blank_axis(ax, f'{FLOW_DISPLAY_NAMES.get(flow_name, flow_name)}\nnot in model')
            except Exception as e:
                logger.error(f"Sampling failed for flow '{flow_name}' on '{ds}': {e}")
                _blank_axis(ax, 'sample\nerror')

        # -- col 4: AMF-VI mixture --
        ax = row_axes[4]
        if amfvi_model is not None:
            try:
                samples = _sample_mixture(amfvi_model, n_samples)
                _draw_contour(ax, samples, x_min, x_max, y_min, y_max, n_grid,
                              label=f'{ds}/AMF-VI', true_stats=true_stats)
            except Exception as e:
                logger.error(f"AMF-VI mixture sampling failed for '{ds}': {e}")
                _blank_axis(ax, 'sample\nerror')
        else:
            _blank_axis(ax, 'AMF-VI\nN/A')

        # -- col 5: AMF-VI-sEMA mixture --
        ax = row_axes[5]
        try:
            samples = _sample_mixture(sema_model, n_samples)
            _draw_contour(ax, samples, x_min, x_max, y_min, y_max, n_grid,
                          label=f'{ds}/sEMA', true_stats=true_stats)
        except Exception as e:
            logger.error(f"sEMA mixture sampling failed for '{ds}': {e}")
            _blank_axis(ax, 'sample\nerror')

    # ----- layout & save -----
    plt.tight_layout(h_pad=0.4, w_pad=0.2)
    try:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Qualitative contour plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save figure: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description='Qualitative density contour plots (paper Fig.3 style)'
    )
    parser.add_argument(
        'datasets', nargs='*', default=None,
        help=(
            'Dataset names to plot (space-separated). '
            f'Defaults to all canonical: {CANONICAL_DATASETS}. '
            'Example: banana x_shape BLR'
        )
    )
    parser.add_argument(
        '--sema', default=DEFAULT_SEMA_RESULTS_DIR,
        help='Directory containing AMF-VI-sEMA pkl files'
    )
    parser.add_argument(
        '--amfvi', default=DEFAULT_AMFVI_RESULTS_DIR,
        help='Directory containing AMF-VI pkl files (results_AMF_VI_C)'
    )
    parser.add_argument(
        '--save', default=None,
        help='Output PNG path (default: <sema_dir>/qualitative_contours.png)'
    )
    parser.add_argument(
        '--n_samples', type=int, default=N_MODEL_SAMPLES,
        help=f'Samples per model for KDE (default: {N_MODEL_SAMPLES})'
    )
    parser.add_argument(
        '--n_grid', type=int, default=N_GRID,
        help=f'KDE grid resolution (default: {N_GRID})'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    datasets = args.datasets if args.datasets else CANONICAL_DATASETS

    # Validate dataset names
    invalid = [d for d in datasets if d not in CANONICAL_DATASETS]
    if invalid:
        logger.error(f"Unrecognised dataset names: {invalid}. Valid: {CANONICAL_DATASETS}")
        sys.exit(1)

    logger.info(f"Plotting {len(datasets)} dataset(s): {datasets}")
    logger.info(f"sEMA dir : {args.sema}")
    logger.info(f"AMF-VI dir: {args.amfvi}")

    plot_qualitative_contours(
        datasets         = datasets,
        sema_results_dir = args.sema,
        amfvi_results_dir= args.amfvi,
        save_path        = args.save,
        n_samples        = args.n_samples,
        n_grid           = args.n_grid,
    )
