"""
File: cross_seed_robustness.py | Version: 1.3.0 | Date: 2026-03-21
Abbreviation: CROSSSEED-ROB
Plan ID: IP-CROSSSEED-ROB-v1.1

Cross-seed robustness analysis for AMF-VI-sEMA (paper Fig. 6).
Runs full pipeline (Stage 1 + Stage 2 + eval) R times with different seeds,
computes std and CV across seeds for NLL, KL, W2, MMD-u, and plots
two-panel grouped bar chart matching paper Fig. 6.

Imports train_sequential_amf_vi from SEMA_MBATCH_vis and
evaluate_single_dataset_comprehensive from eval_10_iters_mbatch.
No reimplementation of existing logic.

CHANGELOG:
- 1.3.0 (2026-03-21): Output directory and JSON tracking (IP-CROSSSEED-ROB-v1.1)
  * output_dir changed from main/results/ to main/results/robustness/
  * save_plot_data_json(): writes std, cv, mean, per_seed values to JSON
  * all_raw dict added in main() to carry (results_array, valid_seeds) per dataset
- 1.2.0 (2026-03-13): Update output directory
  * RESULTS_DIR changed from _ROOT_DIR/results to _MAIN_DIR/results
  * output_dir removed cross_seed subdirectory; saves directly to main/results/
  * CSV and PNG now written to /home/benjamin/Documents/AMF-VIJ/main/results/
- 1.0.0 (2026-03-12): Initial implementation
  * set_all_seeds(): sets torch/numpy/random/cuda seeds simultaneously
  * run_single_seed(): train Stage1+2 -> save pickle -> eval -> extract metrics
  * run_cross_seed(): loops over R seeds, returns (R,4) results array
  * compute_std_cv(): std and CV across seeds per metric
  * save_seed_results_csv(): raw per-seed metrics to CSV
  * plot_cross_seed_robustness(): two-panel grouped bar chart (Fig.6 style)
  * main(): CLI-driven, one or all datasets, R=10 seeds by default
"""

import os
import sys
import csv
import json
import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup — resolves AMF-VIJ root from unit_test/ location
# ---------------------------------------------------------------------------
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))   # .../main/unit_test
_MAIN_DIR  = os.path.dirname(_THIS_DIR)                   # .../main
_ROOT_DIR  = os.path.dirname(_MAIN_DIR)                   # .../AMF-VIJ

for _p in [_THIS_DIR, _MAIN_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
# Imports from existing files
# ---------------------------------------------------------------------------
try:
    from SEMA_MBATCH_vis import train_sequential_amf_vi
except ImportError as e:
    logger.error(f"Failed to import train_sequential_amf_vi from SEMA_MBATCH_vis: {e}")
    raise

try:
    from eval_10_iters_mbatch import evaluate_single_dataset_comprehensive
except ImportError as e:
    logger.error(f"Failed to import evaluate_single_dataset_comprehensive from eval_10_iters_mbatch: {e}")
    raise

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_SEEDS = list(range(2025, 2035))   # R=10: 2025–2034
DEFAULT_FLOW_TYPES = ['realnvp', 'maf', 'rbig']
#RESULTS_DIR = os.path.join(_ROOT_DIR, 'results')
RESULTS_DIR = os.path.join(_MAIN_DIR, 'results')

METRIC_KEYS = ['NLL', 'KL', 'W2', 'MMD_u']
METRIC_EVAL_KEYS = {
    'NLL':   'nll_mean',
    'KL':    'kl_divergence_mean',
    'W2':    'full_wasserstein_mean',
    'MMD_u': 'gaussian_mmd_unbiased_mean',
}
METRIC_COLORS = {
    'NLL':   '#4361EE',   # blue
    'KL':    '#E63946',   # red
    'W2':    '#2EC4B6',   # green
    'MMD_u': '#9B2226',   # purple
}

DATASETS = [
        'banana',
        'x_shape',
        'bimodal_shared',
        #'bimodal_different',
        #'multimodal',
        'two_moons',
        'rings',
        "multimodal-5",
        "BLR",
        "BPR",
        "Weibull",
        "Real-GMM2",
        #"Old-Faithful",
        #"Iris-3Class",
    ]


# ---------------------------------------------------------------------------
# 1. Seed setter
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int):
    """
    Set all random seeds for full reproducibility.

    Args:
        seed: integer seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Seeds set to {seed}")


# ---------------------------------------------------------------------------
# 2. Single seed run
# ---------------------------------------------------------------------------
def run_single_seed(dataset_name: str, seed: int,
                    flow_types=None,
                    n_samples: int = 100_000,
                    tau: float = 1.1,
                    beta: float = 1e-5,
                    M: int = 2) -> dict:
    """
    Run full pipeline (Stage 1 + Stage 2 + eval) for one seed.

    Relies on train_sequential_amf_vi() saving pickle to results/ automatically,
    and evaluate_single_dataset_comprehensive() loading it back.

    Args:
        dataset_name: e.g. 'multimodal'
        seed: random seed for this run
        flow_types: list of flow type strings
        n_samples: samples for training
        tau/beta/M: sEMA hyperparameters

    Returns:
        dict with keys NLL, KL, W2, MMD_u (float each)

    Raises:
        RuntimeError: if training or eval returns None
    """
    if flow_types is None:
        flow_types = DEFAULT_FLOW_TYPES

    set_all_seeds(seed)

    logger.info(f"  [seed={seed}] Starting Stage 1 + Stage 2 for {dataset_name}")
    try:
        train_sequential_amf_vi(
            dataset_name=dataset_name,
            flow_types=flow_types,
            show_plots=False,
            save_plots=False,
            n_samples=n_samples,
            tau=tau,
            beta=beta,
            M=M,
        )
    except Exception as e:
        logger.error(f"  [seed={seed}] Training failed for {dataset_name}: {e}")
        raise RuntimeError(f"Training failed: {e}") from e

    logger.info(f"  [seed={seed}] Starting eval for {dataset_name}")
    try:
        eval_results = evaluate_single_dataset_comprehensive(dataset_name)
    except Exception as e:
        logger.error(f"  [seed={seed}] Eval failed for {dataset_name}: {e}")
        raise RuntimeError(f"Eval failed: {e}") from e

    if eval_results is None:
        raise RuntimeError(f"evaluate_single_dataset_comprehensive returned None for {dataset_name} seed={seed}")

    mixture_metrics = eval_results.get('mixture_metrics', {})
    if not mixture_metrics:
        raise RuntimeError(f"mixture_metrics empty for {dataset_name} seed={seed}")

    # Extract mean values for each metric
    metrics = {}
    for key, eval_key in METRIC_EVAL_KEYS.items():
        val = mixture_metrics.get(eval_key)
        if val is None:
            logger.error(f"  [seed={seed}] Metric '{eval_key}' missing for {dataset_name}")
            raise RuntimeError(f"Missing metric {eval_key}")
        metrics[key] = float(val)

    logger.info(
        f"  [seed={seed}] {dataset_name}: "
        f"NLL={metrics['NLL']:.4f} KL={metrics['KL']:.4f} "
        f"W2={metrics['W2']:.4f} MMD_u={metrics['MMD_u']:.4f}"
    )
    return metrics


# ---------------------------------------------------------------------------
# 3. Cross-seed runner
# ---------------------------------------------------------------------------
def run_cross_seed(dataset_name: str,
                   seeds=None,
                   flow_types=None,
                   n_samples: int = 100_000,
                   tau: float = 1.1,
                   beta: float = 1e-5,
                   M: int = 2):
    """
    Run pipeline for all seeds, collect (R, 4) results array.

    Args:
        dataset_name: dataset to evaluate
        seeds: list of seed integers (default: DEFAULT_SEEDS)
        flow_types/n_samples/tau/beta/M: passed to run_single_seed

    Returns:
        results_array: np.ndarray shape (R, 4) — rows=seeds, cols=metrics
        valid_seeds: list of seeds that completed successfully
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS

    rows = []
    valid_seeds = []

    for seed in seeds:
        try:
            metrics = run_single_seed(
                dataset_name, seed,
                flow_types=flow_types,
                n_samples=n_samples,
                tau=tau, beta=beta, M=M
            )
            rows.append([metrics[k] for k in METRIC_KEYS])
            valid_seeds.append(seed)
        except Exception as e:
            logger.error(f"  Seed {seed} failed for {dataset_name} — skipping: {e}")
            continue

    if len(rows) < 2:
        raise RuntimeError(
            f"Only {len(rows)} seeds succeeded for {dataset_name} — need at least 2 for std/CV"
        )

    results_array = np.array(rows)   # (R, 4)
    logger.info(
        f"  {dataset_name}: {len(valid_seeds)}/{len(seeds)} seeds succeeded"
    )
    return results_array, valid_seeds


# ---------------------------------------------------------------------------
# 4. Std and CV computation
# ---------------------------------------------------------------------------
def compute_std_cv(results_array: np.ndarray) -> dict:
    """
    Compute std and CV across seeds per metric.

    Args:
        results_array: (R, 4) array

    Returns:
        dict with 'std' and 'cv' as arrays of shape (4,), 'metrics' as list
    """
    std = results_array.std(axis=0)                          # (4,)
    mean_abs = np.abs(results_array.mean(axis=0))            # (4,)
    cv = np.where(mean_abs > 1e-10, std / mean_abs, np.nan) # (4,) — guard /0

    return {
        'std':     std,
        'cv':      cv,
        'metrics': METRIC_KEYS,
        'mean':    results_array.mean(axis=0),
    }


# ---------------------------------------------------------------------------
# 5. CSV saver
# ---------------------------------------------------------------------------
def save_seed_results_csv(results_array: np.ndarray,
                          valid_seeds: list,
                          dataset_name: str,
                          save_dir: str):
    """
    Save raw per-seed metric values to CSV.

    Args:
        results_array: (R, 4) array
        valid_seeds: list of seed integers matching rows
        dataset_name: used in filename
        save_dir: output directory
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{dataset_name}_seed_results.csv')

    try:
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seed'] + METRIC_KEYS)
            for seed, row in zip(valid_seeds, results_array):
                writer.writerow([seed] + [f'{v:.6f}' for v in row])
        logger.info(f"Saved seed results CSV to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save CSV for {dataset_name}: {e}")


# ---------------------------------------------------------------------------
# 6. JSON tracker
# ---------------------------------------------------------------------------
def save_plot_data_json(all_std: dict,
                        all_cv: dict,
                        all_raw: dict,
                        completed_datasets: list,
                        seeds: list,
                        save_dir: str):
    """
    Save all values used for plots to JSON for further analysis.

    Writes std, cv, mean, and per_seed values for every completed dataset.

    Args:
        all_std: {dataset: np.ndarray(4,)} std per metric
        all_cv:  {dataset: np.ndarray(4,)} CV per metric
        all_raw: {dataset: (results_array (R,4), valid_seeds list)}
        completed_datasets: ordered list of dataset names
        seeds: full seed list used (for metadata)
        save_dir: output directory
    """
    payload = {
        "metadata": {
            "version":      "1.3.0",
            "plan_id":      "IP-CROSSSEED-ROB-v1.1",
            "generated_at": datetime.now().isoformat(),
            "seeds":        seeds,
            "metrics":      METRIC_KEYS,
        },
        "datasets": {}
    }

    for dataset in completed_datasets:
        results_array, valid_seeds = all_raw[dataset]
        std  = all_std[dataset]
        cv   = all_cv[dataset]
        mean = results_array.mean(axis=0)

        per_seed = {}
        for seed, row in zip(valid_seeds, results_array):
            per_seed[str(seed)] = {
                k: float(v) for k, v in zip(METRIC_KEYS, row)
            }

        payload["datasets"][dataset] = {
            "std":      {k: float(v) for k, v in zip(METRIC_KEYS, std)},
            "cv":       {k: float(v) for k, v in zip(METRIC_KEYS, cv)},
            "mean":     {k: float(v) for k, v in zip(METRIC_KEYS, mean)},
            "per_seed": per_seed,
        }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "cross_seed_plot_data.json")
    try:
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved plot data JSON to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot data JSON: {e}")


# ---------------------------------------------------------------------------
# 7. Plot
# ---------------------------------------------------------------------------
def plot_cross_seed_robustness(all_std: dict,
                               all_cv: dict,
                               datasets: list,
                               save_dir: str):
    """
    Two-panel grouped bar chart matching paper Fig. 6.

    Panel (a): std per metric grouped by dataset
    Panel (b): CV per metric grouped by dataset
    4 coloured bars per dataset: NLL/KL/W2/MMD-u

    Args:
        all_std: {dataset_name: array(4,)} std values
        all_cv:  {dataset_name: array(4,)} CV values
        datasets: ordered list of dataset names for x-axis
        save_dir: output directory
    """
    try:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Cross-Seed Robustness — AMF-VI-sEMA', fontsize=14)

        n_metrics  = len(METRIC_KEYS)
        n_datasets = len(datasets)
        bar_width  = 0.18
        x = np.arange(n_datasets)

        dataset_labels = [d.replace('_', ' ').title() for d in datasets]

        for ax_idx, (ax, data_dict, ylabel, panel_label) in enumerate(zip(
            axes,
            [all_std, all_cv],
            ['Standard Deviation', 'Variance'],
            ['(a) Standard deviation across random seeds.',
             '(b) Coefficient of variation (CV) across seeds.']
        )):
            for m_idx, metric in enumerate(METRIC_KEYS):
                vals = [
                    data_dict[d][m_idx] if d in data_dict else np.nan
                    for d in datasets
                ]
                offset = (m_idx - (n_metrics - 1) / 2) * bar_width
                ax.bar(
                    x + offset, vals,
                    width=bar_width,
                    color=METRIC_COLORS[metric],
                    label=metric if ax_idx == 0 else None,
                    zorder=2
                )

            ax.set_xticks(x)
            ax.set_xticklabels(dataset_labels, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_xlabel(panel_label, fontsize=10, labelpad=8)
            ax.grid(True, axis='y', alpha=0.3, zorder=1)
            ax.set_xlim(-0.5, n_datasets - 0.5)

        # Single legend at top from first panel
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS[m])
            for m in METRIC_KEYS
        ]
        fig.legend(
            handles, METRIC_KEYS,
            loc='upper center', ncol=4,
            fontsize=10, bbox_to_anchor=(0.5, 0.98)
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'cross_seed_robustness.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved cross-seed robustness plot to {save_path}")
        plt.close()

    except Exception as e:
        logger.error(f"Error plotting cross-seed robustness: {e}")


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------
def main(datasets=None, seeds=None, flow_types=None,
         n_samples=100_000, tau=1.1, beta=1e-5, M=2):
    """
    Run cross-seed robustness analysis for one or all datasets.

    Args:
        datasets: list of dataset names (default: all 9)
        seeds: list of seed integers (default: 2025–2034, R=10)
        flow_types/n_samples/tau/beta/M: training hyperparameters
    """
    if datasets is None:
        datasets = DATASETS
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if flow_types is None:
        flow_types = DEFAULT_FLOW_TYPES

    output_dir = os.path.join(RESULTS_DIR, 'robustness')
    os.makedirs(output_dir, exist_ok=True)

    all_std = {}
    all_cv  = {}
    all_raw = {}   # {dataset: (results_array, valid_seeds)} for JSON
    completed_datasets = []

    for dataset_name in datasets:
        logger.info(f"=== Cross-seed robustness: {dataset_name} ({len(seeds)} seeds) ===")

        try:
            results_array, valid_seeds = run_cross_seed(
                dataset_name, seeds=seeds, flow_types=flow_types,
                n_samples=n_samples, tau=tau, beta=beta, M=M
            )
        except RuntimeError as e:
            logger.error(f"Skipping {dataset_name}: {e}")
            continue

        stats = compute_std_cv(results_array)
        all_std[dataset_name] = stats['std']
        all_cv[dataset_name]  = stats['cv']
        all_raw[dataset_name] = (results_array, valid_seeds)

        save_seed_results_csv(results_array, valid_seeds, dataset_name, output_dir)

        # Log summary
        for m, s, c in zip(METRIC_KEYS, stats['std'], stats['cv']):
            logger.info(f"  {dataset_name} {m}: std={s:.4f}, CV={c:.4f}")

        completed_datasets.append(dataset_name)
        logger.info(f"  ✅ {dataset_name} done")

    if len(completed_datasets) < 1:
        logger.error("No datasets completed — no plot generated")
        return

    plot_cross_seed_robustness(all_std, all_cv, completed_datasets, output_dir)
    save_plot_data_json(all_std, all_cv, all_raw, completed_datasets, seeds, output_dir)
    logger.info("=== Cross-seed robustness analysis complete ===")


if __name__ == '__main__':
    # CLI: python cross_seed_robustness.py [dataset1 dataset2 ...]
    datasets_arg = sys.argv[1:] if len(sys.argv) > 1 else None
    main(datasets=datasets_arg)
