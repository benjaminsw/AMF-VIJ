"""
File: sensitivity_analysis.py | Version: 1.2.0 | Date: 2026-03-21
Abbreviation: SENS-ANAL
Plan ID: IP-SENS-JSON-v1.0

Hyperparameter sensitivity analysis for AMF-VI-sEMA (paper Fig. 5).
Loads frozen Stage 1 experts from SEMA_MBATCH_vis pickle, reruns Stage 2
only, sweeps tau / alpha / M independently, and plots final Neff, Volatility
V, Churn C, stabilisation time, final weights, and NLL per sweep value.

CHANGELOG:
- 1.2.0 (2026-03-21): IP-SENS-JSON-v1.0 — extended diagnostics + JSON export
  * load_frozen_model(): also returns test_data (split_data['test']) for NLL eval (C)
  * run_stage2_sweep(): adds test_data param; captures final_weights (K,) [A] and
    final_nll on test_data [C]; stab already present [E]
  * plot_sensitivity(): adds stab time as 4th line dash-dot gray on extra right axis [E]
  * plot_weight_distribution(): new — stacked bar of final pi_k per param value [A]
  * plot_nll_sensitivity(): new — NLL vs param, 3-column [C]
  * plot_cross_dataset_heatmap(): new — datasets x param values, colour = Neff [D]
  * main(): per-dataset JSON export all fields; post-loop cross-dataset JSON + heatmap [D]
- 1.1.1 (2026-03-18): Fix pickle deserialization — register 'SEMA_MBATCH_vis' in sys.modules
  * Added sys.modules['SEMA_MBATCH_vis'] = sys.modules['main.unit_test.SEMA_MBATCH_vis']
  * Resolves ModuleNotFoundError: No module named 'SEMA_MBATCH_vis' on pickle.load
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
import main.unit_test.SEMA_MBATCH_vis
sys.modules['SEMA_MBATCH_vis'] = sys.modules['main.unit_test.SEMA_MBATCH_vis']

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
    Load saved pickle from SEMA_MBATCH_vis and return model + data splits.

    Args:
        dataset_name: e.g. 'multimodal'
        results_dir: override default results path

    Returns:
        model:      SequentialAMFVI with frozen flows (weights reset to uniform)
        train_data: torch.Tensor — Stage 2 fresh batches
        test_data:  torch.Tensor — held-out split for NLL evaluation (C) [v1.2.0]
        meta:       dict with dataset/version metadata

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

    # Load train + test splits
    try:
        from data.data_cache import get_split_data
        split_data = get_split_data(dataset_name)
        device = next(iter(model.flows[0].parameters()), torch.tensor(0.0)).device
        train_data = split_data['train'].to(device)
        test_data  = split_data['test'].to(device)   # v1.2.0: held-out for NLL eval (C)
    except Exception as e:
        logger.error(f"Failed to load data splits for {dataset_name}: {e}")
        raise

    meta = {
        'dataset':    checkpoint.get('dataset', dataset_name),
        'version':    checkpoint.get('version', 'unknown'),
        'orig_tau':   checkpoint.get('tau',   DEFAULT_TAU),
        'orig_alpha': DEFAULT_ALPHA,
        'orig_M':     checkpoint.get('M',     DEFAULT_M),
    }

    logger.info(f"Model loaded — flows frozen, metadata: {meta}")
    return model, train_data, test_data, meta


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
                     test_data: torch.Tensor = None,
                     fixed_tau: float = DEFAULT_TAU,
                     fixed_alpha: float = DEFAULT_ALPHA,
                     fixed_M: int = DEFAULT_M,
                     epochs: int = SWEEP_EPOCHS):
    """
    Sweep one hyperparameter, rerunning Stage 2 for each value.

    Args:
        model:        SequentialAMFVI with frozen flows
        train_data:   training data tensor (Stage 2 fresh batches)
        param_name:   one of 'tau', 'alpha', 'M'
        param_values: list of values to sweep
        test_data:    held-out tensor for NLL evaluation (C) [v1.2.0]
        fixed_tau/alpha/M: held-constant defaults
        epochs:       Stage 2 epochs per run

    Returns:
        results: dict {param_value: {
            'neff':          float,
            'V':             float,
            'C':             int,
            'stab':          int,
            'final_weights': list[float],  # (K,) [A] v1.2.0
            'final_nll':     float,        # on test_data [C] v1.2.0
        }}
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
        tau   = val      if param_name == 'tau'   else fixed_tau
        alpha = val      if param_name == 'alpha' else fixed_alpha
        M     = int(val) if param_name == 'M'     else fixed_M

        try:
            _, tracker = model.train_mixture_weights_moving_average(
                train_data, epochs=epochs, tau=tau, alpha=alpha, M=M
            )
        except Exception as e:
            logger.error(f"Stage 2 failed for {param_name}={val}: {e}")
            results[val] = {
                'neff': np.nan, 'V': np.nan, 'C': np.nan, 'stab': 0,
                'final_weights': [np.nan] * K, 'final_nll': np.nan,
            }
            continue

        weight_history = np.array(tracker.weights)      # (T, K)
        neff_history   = np.array(tracker.neff_history) # (T,)

        final_neff     = float(neff_history[-1]) if len(neff_history) > 0 else np.nan
        V              = compute_volatility(weight_history)
        C              = compute_churn(weight_history)
        stab           = compute_stabilisation_time(neff_history)
        final_weights  = weight_history[-1].tolist() if len(weight_history) > 0 else [np.nan] * K  # A

        # C: final NLL on held-out test_data
        final_nll = np.nan
        if test_data is not None:
            try:
                with torch.no_grad():
                    flow_preds = []
                    for flow in model.flows:
                        lp = flow.log_prob(test_data)          # (N,)
                        lp = torch.clamp(lp, min=-1e6)
                        flow_preds.append(lp)
                    flow_preds = torch.stack(flow_preds, dim=1)  # (N, K)
                    log_pi = torch.log(model.weights.data.clamp(min=1e-8))  # (K,)
                    mix_lp = torch.logsumexp(flow_preds + log_pi, dim=1)   # (N,)
                    final_nll = float(-mix_lp.mean().item())
            except Exception as e:
                logger.error(f"NLL computation failed for {param_name}={val}: {e}")

        results[val] = {
            'neff':          final_neff,
            'V':             V,
            'C':             C,
            'stab':          stab,
            'final_weights': final_weights,  # A v1.2.0
            'final_nll':     final_nll,      # C v1.2.0
        }
        logger.info(
            f"    → Neff={final_neff:.3f}, V={V:.4f}, C={C}, "
            f"stab={stab}, NLL={final_nll:.4f}, weights={[f'{w:.3f}' for w in final_weights]}"
        )

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
    Left y-axis:        Neff (solid).
    Right y-axis:       Volatility V (dashed), Churn C (dotted).
    Far-right y-axis:   Stabilisation time (dash-dot gray) [E v1.2.0]
    Colour shading encodes stabilisation time.

    Args:
        tau_results/alpha_results/M_results: dicts from run_stage2_sweep()
        dataset_name: used in title
        save_path: if provided, saves figure
        n_experts: K (for y-axis limits)
    """
    try:
        sweeps = [
            ('tau_results',   r'Temperature ($\tau$)',       tau_results),
            ('alpha_results', r'Smoothing Factor ($\alpha$)', alpha_results),
            ('M_results',     r'Ensemble Size $M$',           M_results),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Sensitivity Analysis — {dataset_name.title()}', fontsize=14)

        colour_sets = [
            {'neff': '#3A0CA3', 'V': '#E63946', 'C': '#2EC4B6', 'stab': '#888888'},
            {'neff': '#7B2D8B', 'V': '#F4A261', 'C': '#48CAE4', 'stab': '#888888'},
            {'neff': '#2D6A4F', 'V': '#95D5B2', 'C': '#F72585', 'stab': '#888888'},
        ]

        for col, (_, xlabel, results) in enumerate(sweeps):
            ax_l  = axes[col]
            ax_r  = ax_l.twinx()
            # v1.2.0: third axis for stabilisation time [E]
            ax_r2 = ax_l.twinx()
            ax_r2.spines['right'].set_position(('outward', 52))
            colours = colour_sets[col]

            param_vals = sorted(results.keys())
            neff_vals  = [results[v]['neff'] for v in param_vals]
            V_vals     = [results[v]['V']    for v in param_vals]
            C_vals     = [results[v]['C']    for v in param_vals]
            stab_vals  = [results[v]['stab'] for v in param_vals]

            # Marker shading: darker = longer stabilisation
            max_stab = max(stab_vals) if max(stab_vals) > 0 else 1
            alphas   = [0.3 + 0.7 * (s / max_stab) for s in stab_vals]

            # Left axis: Neff (solid)
            ax_l.plot(param_vals, neff_vals, color=colours['neff'],
                      linestyle='-', linewidth=2, label=r'$N_{eff}$', zorder=3)
            for xv, yv, a in zip(param_vals, neff_vals, alphas):
                ax_l.scatter(xv, yv, color=colours['neff'], s=60, alpha=a, zorder=4)

            # Right axis: Volatility V (dashed)
            ax_r.plot(param_vals, V_vals, color=colours['V'],
                      linestyle='--', linewidth=1.5, label='Volatility', zorder=2)
            for xv, yv, a in zip(param_vals, V_vals, alphas):
                ax_r.scatter(xv, yv, color=colours['V'], s=40, alpha=a, marker='s', zorder=3)

            # Right axis: Churn C (dotted)
            ax_r.plot(param_vals, C_vals, color=colours['C'],
                      linestyle=':', linewidth=1.5, label='Churn', zorder=2)
            for xv, yv, a in zip(param_vals, C_vals, alphas):
                ax_r.scatter(xv, yv, color=colours['C'], s=40, alpha=a, marker='o', zorder=3)

            # Far-right axis: Stabilisation time (dash-dot gray) [E v1.2.0]
            ax_r2.plot(param_vals, stab_vals, color=colours['stab'],
                       linestyle='-.', linewidth=1.2, label='Stab. epoch', zorder=1)
            ax_r2.set_ylabel('Stab. epoch', fontsize=9, color=colours['stab'])
            ax_r2.tick_params(axis='y', labelcolor=colours['stab'], labelsize=8)

            # Labels and limits
            ax_l.set_xlabel(xlabel, fontsize=11)
            ax_l.set_ylabel(r'$N_{eff}$', fontsize=11)
            ax_r.set_ylabel('Vol / Churn', fontsize=11)
            ax_l.set_ylim(1.0, n_experts + 0.2)

            # Combined legend from all three axes
            lines_l,  labels_l  = ax_l.get_legend_handles_labels()
            lines_r,  labels_r  = ax_r.get_legend_handles_labels()
            lines_r2, labels_r2 = ax_r2.get_legend_handles_labels()
            ax_l.legend(lines_l + lines_r + lines_r2,
                        labels_l + labels_r + labels_r2,
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
# 5. Plot — stacked bar of final weights per param value [A v1.2.0]
# ---------------------------------------------------------------------------
def plot_weight_distribution(tau_results: dict, alpha_results: dict, M_results: dict,
                              expert_names: list, dataset_name: str,
                              save_path: str = None):
    """
    Stacked bar chart of final mixture weights π_k across sweep values.
    One subplot per param (tau | alpha | M).

    Args:
        tau_results/alpha_results/M_results: dicts from run_stage2_sweep()
        expert_names: list of flow type strings e.g. ['realnvp', 'maf', 'rbig']
        dataset_name: used in title
        save_path: if provided, saves figure
    """
    try:
        EXPERT_COLORS = {'realnvp': '#4C72B0', 'maf': '#DD8452', 'rbig': '#55A868'}
        default_colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

        sweeps = [
            (r'Temperature ($\tau$)',       tau_results),
            (r'Smoothing Factor ($\alpha$)', alpha_results),
            (r'Ensemble Size $M$',           M_results),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.suptitle(f'Final Weight Distribution — {dataset_name.title()}', fontsize=13)

        for col, (xlabel, results) in enumerate(sweeps):
            ax = axes[col]
            param_vals = sorted(results.keys())
            K = len(expert_names)
            bottoms = np.zeros(len(param_vals))

            for k, name in enumerate(expert_names):
                weights_k = []
                for v in param_vals:
                    fw = results[v].get('final_weights', [np.nan] * K)
                    weights_k.append(fw[k] if not np.isnan(fw[k]) else 0.0)
                color = EXPERT_COLORS.get(name.lower(), default_colors[k % len(default_colors)])
                ax.bar(range(len(param_vals)), weights_k, bottom=bottoms,
                       color=color, label=name.upper(), alpha=0.85)
                bottoms += np.array(weights_k)

            ax.set_xticks(range(len(param_vals)))
            ax.set_xticklabels([str(v) for v in param_vals], fontsize=9)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(r'$\pi_k$', fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.grid(axis='y', alpha=0.3)
            if col == 0:
                ax.legend(fontsize=8, loc='upper right')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved weight distribution plot to {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Error plotting weight distribution for {dataset_name}: {e}")


# ---------------------------------------------------------------------------
# 6. Plot — NLL vs param value [C v1.2.0]
# ---------------------------------------------------------------------------
def plot_nll_sensitivity(tau_results: dict, alpha_results: dict, M_results: dict,
                          dataset_name: str, save_path: str = None):
    """
    3-column NLL vs hyperparameter plot on held-out test_data.
    Horizontal dashed line = NLL at default hyperparameters for reference.

    Args:
        tau_results/alpha_results/M_results: dicts from run_stage2_sweep()
        dataset_name: used in title
        save_path: if provided, saves figure
    """
    try:
        sweeps = [
            (r'Temperature ($\tau$)',       DEFAULT_TAU,   tau_results),
            (r'Smoothing Factor ($\alpha$)', DEFAULT_ALPHA, alpha_results),
            (r'Ensemble Size $M$',           DEFAULT_M,     M_results),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle(f'NLL Sensitivity (test set) — {dataset_name.title()}', fontsize=13)

        colours = ['#3A0CA3', '#7B2D8B', '#2D6A4F']

        for col, (xlabel, default_val, results) in enumerate(sweeps):
            ax = axes[col]
            param_vals = sorted(results.keys())
            nll_vals   = [results[v].get('final_nll', np.nan) for v in param_vals]

            # Reference NLL at default value
            ref_nll = results.get(default_val, {}).get('final_nll', np.nan)

            ax.plot(param_vals, nll_vals, color=colours[col],
                    linestyle='-', linewidth=2, marker='o', markersize=6)

            if not np.isnan(ref_nll):
                ax.axhline(ref_nll, color='gray', linestyle='--',
                           linewidth=1.2, label=f'default ({default_val})')
                ax.legend(fontsize=8)

            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('NLL (test)', fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved NLL sensitivity plot to {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Error plotting NLL sensitivity for {dataset_name}: {e}")


# ---------------------------------------------------------------------------
# 7. Plot — cross-dataset Neff heatmap [D v1.2.0]
# ---------------------------------------------------------------------------
def plot_cross_dataset_heatmap(cross_dataset: dict, param: str,
                                param_values: list, save_path: str = None):
    """
    Heatmap: rows = datasets, cols = param values, colour = final Neff.
    One call per param (tau, alpha, M).

    Args:
        cross_dataset: dict {dataset_name: {param_str: neff_float}}
        param:         'tau', 'alpha', or 'M'
        param_values:  ordered list of sweep values for x-axis
        save_path:     if provided, saves figure
    """
    try:
        datasets = list(cross_dataset.keys())
        matrix   = np.full((len(datasets), len(param_values)), np.nan)

        for i, ds in enumerate(datasets):
            for j, pv in enumerate(param_values):
                matrix[i, j] = cross_dataset[ds].get(str(pv), np.nan)

        fig, ax = plt.subplots(figsize=(max(8, len(param_values) * 1.2),
                                        max(4, len(datasets) * 0.6)))
        im = ax.imshow(matrix, aspect='auto', cmap='viridis',
                       vmin=1.0, vmax=3.0, interpolation='nearest')
        plt.colorbar(im, ax=ax, label=r'$N_{eff}$')

        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels([str(v) for v in param_values], fontsize=9)
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets, fontsize=9)
        ax.set_xlabel(param, fontsize=11)
        ax.set_title(f'Cross-dataset $N_{{eff}}$ — sweep {param}', fontsize=12)

        # Annotate cells with Neff value
        for i in range(len(datasets)):
            for j in range(len(param_values)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=7, color='white' if val < 2.0 else 'black')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cross-dataset heatmap to {save_path}")

        plt.close()

    except Exception as e:
        logger.error(f"Error plotting cross-dataset heatmap for param={param}: {e}")


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------
def main(datasets=None, results_dir=None, epochs=SWEEP_EPOCHS):
    """
    Run full sensitivity analysis across datasets.

    Args:
        datasets:    list of dataset names (defaults to DATASETS)
        results_dir: path to results dir containing trained_model_*.pkl
        epochs:      Stage 2 epochs per sweep run
    """
    import json

    if datasets is None:
        datasets = DATASETS

    if results_dir is None:
        results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results'
        )

    output_dir = os.path.join(results_dir, 'sensitivity')
    os.makedirs(output_dir, exist_ok=True)

    # v1.2.0: accumulate Neff per dataset per param for cross-dataset heatmap [D]
    cross_tau   = {}  # {dataset: {str(val): neff}}
    cross_alpha = {}
    cross_M     = {}

    for dataset_name in datasets:
        logger.info(f"=== Sensitivity analysis: {dataset_name} ===")

        try:
            model, train_data, test_data, meta = load_frozen_model(dataset_name, results_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        n_experts    = len(model.flows)
        expert_names = getattr(model, 'flow_types', [f'expert_{k}' for k in range(n_experts)])

        logger.info(f"  Sweeping tau {TAU_VALUES}")
        tau_results = run_stage2_sweep(
            model, train_data, 'tau', TAU_VALUES,
            test_data=test_data,
            fixed_tau=DEFAULT_TAU, fixed_alpha=DEFAULT_ALPHA, fixed_M=DEFAULT_M,
            epochs=epochs
        )

        logger.info(f"  Sweeping alpha {ALPHA_VALUES}")
        alpha_results = run_stage2_sweep(
            model, train_data, 'alpha', ALPHA_VALUES,
            test_data=test_data,
            fixed_tau=DEFAULT_TAU, fixed_alpha=DEFAULT_ALPHA, fixed_M=DEFAULT_M,
            epochs=epochs
        )

        logger.info(f"  Sweeping M {M_VALUES}")
        M_results = run_stage2_sweep(
            model, train_data, 'M', M_VALUES,
            test_data=test_data,
            fixed_tau=DEFAULT_TAU, fixed_alpha=DEFAULT_ALPHA, fixed_M=DEFAULT_M,
            epochs=epochs
        )

        # ── plots ──────────────────────────────────────────────────────────
        try:
            plot_sensitivity(
                tau_results, alpha_results, M_results,
                dataset_name=dataset_name,
                save_path=os.path.join(output_dir, f'{dataset_name}_sensitivity.png'),
                n_experts=n_experts
            )
        except Exception as e:
            logger.error(f"plot_sensitivity failed for {dataset_name}: {e}")

        try:
            plot_weight_distribution(                                       # A v1.2.0
                tau_results, alpha_results, M_results,
                expert_names=expert_names,
                dataset_name=dataset_name,
                save_path=os.path.join(output_dir, f'{dataset_name}_weight_dist.png'),
            )
        except Exception as e:
            logger.error(f"plot_weight_distribution failed for {dataset_name}: {e}")

        try:
            plot_nll_sensitivity(                                           # C v1.2.0
                tau_results, alpha_results, M_results,
                dataset_name=dataset_name,
                save_path=os.path.join(output_dir, f'{dataset_name}_nll_sensitivity.png'),
            )
        except Exception as e:
            logger.error(f"plot_nll_sensitivity failed for {dataset_name}: {e}")

        # ── per-dataset JSON export ─────────────────────────────────────────
        def _serialise(results_dict):
            """Convert sweep results to JSON-safe dict."""
            out = {}
            for k, v in results_dict.items():
                entry = {}
                for field, val in v.items():
                    if isinstance(val, float) and np.isnan(val):
                        entry[field] = None
                    elif isinstance(val, list):
                        entry[field] = [None if (isinstance(x, float) and np.isnan(x)) else x
                                        for x in val]
                    else:
                        entry[field] = val
                out[str(k)] = entry
            return out

        sens_record = {
            'metadata': {
                'dataset':       dataset_name,
                'sweep_epochs':  epochs,
                'defaults':      {'tau': DEFAULT_TAU, 'alpha': DEFAULT_ALPHA, 'M': DEFAULT_M},
                'sweep_ranges':  {'tau': TAU_VALUES, 'alpha': ALPHA_VALUES, 'M': M_VALUES},
                'n_experts':     n_experts,
                'expert_names':  expert_names,
                'model_version': meta.get('version', 'unknown'),
                'nll_eval_split': 'test',
            },
            'tau_results':   _serialise(tau_results),
            'alpha_results': _serialise(alpha_results),
            'M_results':     _serialise(M_results),
        }

        json_path = os.path.join(output_dir, f'{dataset_name}_sensitivity.json')
        try:
            with open(json_path, 'w') as f:
                json.dump(sens_record, f, indent=2)
            logger.info(f"  Sensitivity JSON saved to {json_path}")
        except Exception as je:
            logger.error(f"Failed to save sensitivity JSON for {dataset_name}: {je}")

        # ── accumulate cross-dataset Neff [D] ──────────────────────────────
        cross_tau[dataset_name]   = {str(k): v['neff'] for k, v in tau_results.items()}
        cross_alpha[dataset_name] = {str(k): v['neff'] for k, v in alpha_results.items()}
        cross_M[dataset_name]     = {str(k): v['neff'] for k, v in M_results.items()}

        logger.info(f"  ✅ Done — {dataset_name}")

    # ── cross-dataset JSON + heatmaps [D v1.2.0] ───────────────────────────
    cross_record = {
        'metadata': {
            'datasets':     list(cross_tau.keys()),
            'sweep_ranges': {'tau': TAU_VALUES, 'alpha': ALPHA_VALUES, 'M': M_VALUES},
            'defaults':     {'tau': DEFAULT_TAU, 'alpha': DEFAULT_ALPHA, 'M': DEFAULT_M},
        },
        'tau':   cross_tau,
        'alpha': cross_alpha,
        'M':     cross_M,
    }

    cross_json_path = os.path.join(output_dir, 'cross_dataset_neff_summary.json')
    try:
        with open(cross_json_path, 'w') as f:
            json.dump(cross_record, f, indent=2)
        logger.info(f"Cross-dataset JSON saved to {cross_json_path}")
    except Exception as je:
        logger.error(f"Failed to save cross-dataset JSON: {je}")

    for param, param_vals, cross_data in [
        ('tau',   TAU_VALUES,   cross_tau),
        ('alpha', ALPHA_VALUES, cross_alpha),
        ('M',     M_VALUES,     cross_M),
    ]:
        try:
            plot_cross_dataset_heatmap(
                cross_data, param=param, param_values=param_vals,
                save_path=os.path.join(output_dir, f'cross_dataset_neff_{param}.png'),
            )
        except Exception as e:
            logger.error(f"plot_cross_dataset_heatmap failed for param={param}: {e}")

    logger.info("=== Sensitivity analysis complete ===")


if __name__ == '__main__':
    # Optionally pass dataset names as CLI args: python sensitivity_analysis.py multimodal rings
    datasets_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    main(datasets=datasets_arg)
