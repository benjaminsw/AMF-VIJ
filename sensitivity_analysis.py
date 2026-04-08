"""
File: sensitivity_analysis.py | Version: 1.1.0 | Date: 2026-03-14
Abbreviation: SENS-ANAL
Plan ID: IP-SENSITIVITY-v1.0

Hyperparameter sensitivity analysis for AMF-VI-sEMA (paper Fig. 5).
Loads frozen Stage 1 experts from SEMA_MBATCH_vis pickle, reruns Stage 2
only, sweeps tau / alpha / M independently, and plots final Neff, Volatility
V, and Churn C in a 3-column figure per dataset.

CHANGELOG:
- 1.1.0 (2026-03-14): Align dataset list with SEMA-MBATCH-vis v2.6.0
  * DATASETS: replaced stale keys (x_shaped, bimodal, multimodal, blr, bpr, weibull)
    with canonical keys (x_shape, bimodal_shared, multimodal-5, BLR, BPR, Weibull)
  * Added Real-GMM2 to DATASETS
  * No logic changes; purely dataset-scope alignment
- 1.0.0 (2026-03-12): Initial implementation
  * load_frozen_model(): loads pickle, resets weights to uniform before each run
  * compute_volatility(): V = (1/T-1) * sum ||pi^(t+1) - pi^(t)||_1
  * compute_churn(): C = count of argmax changes over epochs
  * compute_stabilisation_time(): epoch when Neff first settles within tol of final
  * run_stage2_sweep(): sweeps one param, returns (neff, V, C, stab_time) per value
  * plot_sensitivity(): 3-column dual-y-axis figure matching paper Fig.5 style
  * main(): iterates over datasets, saves to results/sensitivity/
"""

import os
import sys
import copy
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

# Required for unpickling — must be imported before pickle.load
from main.unit_test.SEMA_MBATCH_vis import SequentialAMFVI, train_sequential_amf_vi  # noqa: F401

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sweep ranges (matching paper Fig. 5)
# ---------------------------------------------------------------------------
TAU_VALUES   = [0.8, 1.0, 1.1, 1.2, 1.4, 1.6]
ALPHA_VALUES = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
M_VALUES     = [1,   2,   3,   5,   10,  15  ]

# Fixed defaults when sweeping another param
DEFAULT_TAU   = 1.1
DEFAULT_ALPHA = 0.9
DEFAULT_M     = 2

# Stage 2 epochs per sensitivity run
SWEEP_EPOCHS = 500

# Datasets to analyse
DATASETS = [
    'banana', 'x_shape', 'bimodal_shared', 'two_moons', 'rings',
    'multimodal-5', 'BLR', 'BPR', 'Weibull', 'Real-GMM2',
]


# ---------------------------------------------------------------------------
# 1. Load frozen model
# ---------------------------------------------------------------------------
def load_frozen_model(dataset_name: str, results_dir: str = None):
    """
    Load saved pickle from SEMA_MBATCH_vis and return model + training data.

    Args:
        dataset_name: e.g. 'multimodal'
        results_dir: override default results path

    Returns:
        model: SequentialAMFVI with frozen flows (weights reset to uniform)
        train_data: torch.Tensor used for Stage 2 fresh batches
        meta: dict with dataset/version metadata

    Raises:
        FileNotFoundError: if pickle does not exist — no silent fallback
    """
    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'
        )

    model_path = os.path.join(results_dir, f'trained_model_{dataset_name}.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pickle not found: {model_path}. "
            f"Run SEMA_MBATCH_vis.py for dataset '{dataset_name}' first."
        )

    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        checkpoint = pickle.load(f)

    model = checkpoint['model']

    # Ensure flows are in eval mode and frozen
    for flow in model.flows:
        flow.eval()
        for param in flow.parameters():
            param.requires_grad_(False)

    # Load fresh training data for Stage 2 batches
    try:
        from data.data_cache import get_split_data
        split_data = get_split_data(dataset_name)
        train_data = split_data['train']
        device = next(iter(model.flows[0].parameters()), torch.tensor(0.0)).device
        train_data = train_data.to(device)
    except Exception as e:
        logger.error(f"Failed to load training data for {dataset_name}: {e}")
        raise

    meta = {
        'dataset': checkpoint.get('dataset', dataset_name),
        'version': checkpoint.get('version', 'unknown'),
        'orig_tau': checkpoint.get('tau', DEFAULT_TAU),
        'orig_alpha': DEFAULT_ALPHA,
        'orig_M': checkpoint.get('M', DEFAULT_M),
    }

    logger.info(f"Model loaded — flows frozen, metadata: {meta}")
    return model, train_data, meta


# ---------------------------------------------------------------------------
# 2. Diagnostic metrics
# ---------------------------------------------------------------------------
def compute_volatility(weight_history: np.ndarray) -> float:
    """
    Weight volatility V = (1/T-1) * sum_t ||pi^(t+1) - pi^(t)||_1

    Args:
        weight_history: (T, K) array of weights over epochs

    Returns:
        Scalar volatility value
    """
    if len(weight_history) < 2:
        logger.warning("Weight history too short for volatility computation")
        return 0.0
    diffs = np.abs(np.diff(weight_history, axis=0)).sum(axis=1)  # (T-1,)
    return float(diffs.mean())


def compute_churn(weight_history: np.ndarray) -> int:
    """
    Gate churn C = number of times the dominant expert changes over epochs.

    Args:
        weight_history: (T, K) array of weights over epochs

    Returns:
        Integer churn count
    """
    if len(weight_history) < 2:
        logger.warning("Weight history too short for churn computation")
        return 0
    dominant = np.argmax(weight_history, axis=1)  # (T,)
    changes = np.sum(np.diff(dominant) != 0)
    return int(changes)


def compute_stabilisation_time(neff_history: np.ndarray, tol: float = 0.05) -> int:
    """
    Epoch at which Neff first stays within tol of its final value.

    Args:
        neff_history: (T,) array of Neff over epochs
        tol: tolerance band around final Neff

    Returns:
        Stabilisation epoch index (used for colour shading)
    """
    if len(neff_history) == 0:
        return 0
    final = neff_history[-1]
    for t, val in enumerate(neff_history):
        if abs(val - final) <= tol:
            return t
    return len(neff_history) - 1


# ---------------------------------------------------------------------------
# 3. Stage 2 sweep runner
# ---------------------------------------------------------------------------
def run_stage2_sweep(model, train_data: torch.Tensor,
                     param_name: str, param_values: list,
                     fixed_tau: float = DEFAULT_TAU,
                     fixed_alpha: float = DEFAULT_ALPHA,
                     fixed_M: int = DEFAULT_M,
                     epochs: int = SWEEP_EPOCHS):
    """
    Sweep one hyperparameter, rerunning Stage 2 for each value.

    Args:
        model: SequentialAMFVI with frozen flows
        train_data: training data tensor
        param_name: one of 'tau', 'alpha', 'M'
        param_values: list of values to sweep
        fixed_tau/alpha/M: held-constant defaults
        epochs: Stage 2 epochs per run

    Returns:
        results: dict {param_value: {'neff': float, 'V': float, 'C': int, 'stab': int}}
    """
    if param_name not in ('tau', 'alpha', 'M'):
        raise ValueError(f"param_name must be 'tau', 'alpha', or 'M', got '{param_name}'")

    results = {}
    K = len(model.flows)

    for val in param_values:
        logger.info(f"  Sweep {param_name}={val}")

        # Reset weights to uniform before each run
        with torch.no_grad():
            model.weights.data = torch.ones(K, device=model.weights.device) / K

        # Resolve which param is being swept
        tau   = val          if param_name == 'tau'   else fixed_tau
        alpha = val          if param_name == 'alpha' else fixed_alpha
        M     = int(val)     if param_name == 'M'     else fixed_M

        try:
            _, tracker = model.train_mixture_weights_moving_average(
                train_data, epochs=epochs, tau=tau, alpha=alpha, M=M
            )
        except Exception as e:
            logger.error(f"Stage 2 failed for {param_name}={val}: {e}")
            results[val] = {'neff': np.nan, 'V': np.nan, 'C': np.nan, 'stab': 0}
            continue

        weight_history = np.array(tracker.weights)   # (T, K)
        neff_history   = np.array(tracker.neff_history)  # (T,)

        final_neff = float(neff_history[-1]) if len(neff_history) > 0 else np.nan
        V          = compute_volatility(weight_history)
        C          = compute_churn(weight_history)
        stab       = compute_stabilisation_time(neff_history)

        results[val] = {'neff': final_neff, 'V': V, 'C': C, 'stab': stab}
        logger.info(f"    → Neff={final_neff:.3f}, V={V:.4f}, C={C}, stab_epoch={stab}")

    return results


# ---------------------------------------------------------------------------
# 4. Plot — 3-column figure matching paper Fig. 5
# ---------------------------------------------------------------------------
def plot_sensitivity(tau_results: dict, alpha_results: dict, M_results: dict,
                     dataset_name: str,
                     save_path: str = None,
                     n_experts: int = 3):
    """
    3-column sensitivity plot: tau | alpha | M.
    Left y-axis: Neff (solid). Right y-axis: Volatility V (dashed), Churn C (dotted).
    Colour shading encodes stabilisation time.

    Args:
        tau_results/alpha_results/M_results: dicts from run_stage2_sweep()
        dataset_name: used in title
        save_path: if provided, saves figure
        n_experts: K (for y-axis limits)
    """
    try:
        sweeps = [
            ('tau_results',   r'Temperature ($\tau$)',    tau_results),
            ('alpha_results', r'Smoothing Factor ($\alpha$)', alpha_results),
            ('M_results',     r'Ensemble Size $M$',       M_results),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Sensitivity Analysis — {dataset_name.title()}', fontsize=14)

        # Colour palette — 3 colour sets, one per column (matching paper)
        colour_sets = [
            {'neff': '#3A0CA3', 'V': '#E63946', 'C': '#2EC4B6'},  # blue/red/teal
            {'neff': '#7B2D8B', 'V': '#F4A261', 'C': '#48CAE4'},  # purple/orange/cyan
            {'neff': '#2D6A4F', 'V': '#95D5B2', 'C': '#F72585'},  # green/mint/pink
        ]

        for col, (_, xlabel, results) in enumerate(sweeps):
            ax_l = axes[col]
            ax_r = ax_l.twinx()
            colours = colour_sets[col]

            param_vals = sorted(results.keys())
            neff_vals  = [results[v]['neff'] for v in param_vals]
            V_vals     = [results[v]['V']    for v in param_vals]
            C_vals     = [results[v]['C']    for v in param_vals]
            stab_vals  = [results[v]['stab'] for v in param_vals]

            # Colour shading on markers: darker = longer stabilisation time
            max_stab = max(stab_vals) if max(stab_vals) > 0 else 1
            alphas   = [0.3 + 0.7 * (s / max_stab) for s in stab_vals]

            # Left axis: Neff (solid)
            ax_l.plot(param_vals, neff_vals, color=colours['neff'],
                      linestyle='-', linewidth=2, label=r'$N_{eff}$', zorder=3)
            for xv, yv, a in zip(param_vals, neff_vals, alphas):
                ax_l.scatter(xv, yv, color=colours['neff'], s=60,
                             alpha=a, zorder=4)

            # Right axis: Volatility V (dashed)
            ax_r.plot(param_vals, V_vals, color=colours['V'],
                      linestyle='--', linewidth=1.5, label=r'Volatility', zorder=2)
            for xv, yv, a in zip(param_vals, V_vals, alphas):
                ax_r.scatter(xv, yv, color=colours['V'], s=40,
                             alpha=a, marker='s', zorder=3)

            # Right axis: Churn C (dotted)
            ax_r.plot(param_vals, C_vals, color=colours['C'],
                      linestyle=':', linewidth=1.5, label=r'Churn', zorder=2)
            for xv, yv, a in zip(param_vals, C_vals, alphas):
                ax_r.scatter(xv, yv, color=colours['C'], s=40,
                             alpha=a, marker='o', zorder=3)

            # Labels and limits
            ax_l.set_xlabel(xlabel, fontsize=11)
            ax_l.set_ylabel(r'$N_{eff}$', fontsize=11)
            ax_r.set_ylabel(r'Vol / Churn', fontsize=11)
            ax_l.set_ylim(1.0, n_experts + 0.2)

            # Combined legend
            lines_l, labels_l = ax_l.get_legend_handles_labels()
            lines_r, labels_r = ax_r.get_legend_handles_labels()
            ax_l.legend(lines_l + lines_r, labels_l + labels_r,
                        fontsize=8, loc='lower right')

            ax_l.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sensitivity plot to {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Error plotting sensitivity for {dataset_name}: {e}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main(datasets=None, results_dir=None, epochs=SWEEP_EPOCHS):
    """
    Run full sensitivity analysis across datasets.

    Args:
        datasets: list of dataset names (defaults to DATASETS)
        results_dir: path to results dir containing trained_model_*.pkl
        epochs: Stage 2 epochs per sweep run
    """
    if datasets is None:
        datasets = DATASETS

    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'
        )

    output_dir = os.path.join(results_dir, 'sensitivity')
    os.makedirs(output_dir, exist_ok=True)

    for dataset_name in datasets:
        logger.info(f"=== Sensitivity analysis: {dataset_name} ===")

        try:
            model, train_data, meta = load_frozen_model(dataset_name, results_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        n_experts = len(model.flows)

        logger.info(f"  Sweeping tau {TAU_VALUES}")
        tau_results = run_stage2_sweep(
            model, train_data, 'tau', TAU_VALUES,
            fixed_tau=DEFAULT_TAU, fixed_alpha=DEFAULT_ALPHA, fixed_M=DEFAULT_M,
            epochs=epochs
        )

        logger.info(f"  Sweeping alpha {ALPHA_VALUES}")
        alpha_results = run_stage2_sweep(
            model, train_data, 'alpha', ALPHA_VALUES,
            fixed_tau=DEFAULT_TAU, fixed_alpha=DEFAULT_ALPHA, fixed_M=DEFAULT_M,
            epochs=epochs
        )

        logger.info(f"  Sweeping M {M_VALUES}")
        M_results = run_stage2_sweep(
            model, train_data, 'M', M_VALUES,
            fixed_tau=DEFAULT_TAU, fixed_alpha=DEFAULT_ALPHA, fixed_M=DEFAULT_M,
            epochs=epochs
        )

        save_path = os.path.join(output_dir, f'{dataset_name}_sensitivity.png')
        plot_sensitivity(
            tau_results, alpha_results, M_results,
            dataset_name=dataset_name,
            save_path=save_path,
            n_experts=n_experts
        )

        logger.info(f"  ✅ Done — {dataset_name} saved to {save_path}")

    logger.info("=== Sensitivity analysis complete ===")


if __name__ == '__main__':
    # Optionally pass dataset names as CLI args: python sensitivity_analysis.py multimodal rings
    datasets_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    main(datasets=datasets_arg)
