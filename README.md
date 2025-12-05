# AMF-VI: Adaptive Mixture of Flows for Variational Inference

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of Sequential Adaptive Mixture of Flows for Variational Inference with learned mixture weights.

## 🎯 Overview

AMF-VI combines multiple heterogeneous normalizing flows (RealNVP, MAF, RBIG) using a **two-stage sequential training** approach:

1. **Stage 1**: Train individual flows independently on target distribution
2. **Stage 2**: Learn optimal mixture weights via moving average

This avoids gradient conflicts and allows each flow to specialize on different aspects of the target.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Train Models

```python
from main.threeflows_amf_vi_weights_log import train_sequential_amf_vi

# Train on multimodal dataset with 3 flows
model, flow_losses, weight_losses = train_sequential_amf_vi(
    dataset_name='multimodal',
    flow_types=['realnvp', 'maf', 'rbig'],
    show_plots=True,
    save_plots=True
)
```

### Evaluate Models

```python
from main.eval_10_iters import comprehensive_evaluation

# Evaluate with statistical robustness (10 iterations)
results = comprehensive_evaluation(n_iterations=10)
```

## 📊 Features

- **10+ Flow Architectures**: RealNVP, MAF, RBIG, IAF, Gaussianization, NAF, Glow, NICE, Spline, TAN
- **7 Benchmark Datasets**: banana, x_shape, bimodal_shared, bimodal_different, multimodal, two_moons, rings
- **Sequential Training**: Two-stage approach prevents gradient interference
- **Adaptive Weights**: Moving average weight learning based on flow performance
- **Statistical Evaluation**: 5 metrics (NLL, KL, Wasserstein, MMD×2) with mean ± std over 10 iterations
- **Automatic Visualization**: Training curves, samples, and comparisons

## 📂 Project Structure

```
amf-vi/
├── amf_vi/
│   ├── flows/              # Normalizing flow implementations
│   │   ├── realnvp.py
│   │   ├── maf.py
│   │   ├── rbig.py
│   │   └── ...
│   ├── kde_kl_divergence.py
│   └── loss.py
├── data/
│   ├── data_generator.py  # Synthetic dataset generation
│   └── visualize_data.py
├── main/
│   ├── threeflows_amf_vi_weights_log.py  # Training script
│   ├── eval_10_iters.py                   # Evaluation script
│   └── amf_vi_visualization.py
├── results/                # Output directory (auto-created)
└── requirements.txt
```

## 🔧 Usage

### Train All Datasets

```bash
python main/threeflows_amf_vi_weights_log.py
```

**Outputs:**
- `results/trained_model_{dataset}.pkl` - Trained models
- `results/sequential_amf_vi_results_{dataset}.png` - Visualizations

### Evaluate All Models

```bash
python main/eval_10_iters.py
```

**Outputs:**
- `results/comprehensive_evaluation_10_iterations.csv` - Metrics table

## 📈 Results Format

### Training Output

```
🚀 Sequential AMF-VI Experiment on multimodal
============================================================
Stage 1: Training flows independently...
  Flow 1/3: RealNVP - Final loss: 0.8234
  Flow 2/3: MAF - Final loss: 0.7891
  Flow 3/3: RBIG - Final loss: 0.8012

Stage 2: Learning mixture weights...
  Final weights: [0.412, 0.358, 0.230]
```

### Evaluation Output (CSV)

| dataset | model | nll_mean | nll_std | kl_divergence_mean | ... |
|---------|-------|----------|---------|-------------------|-----|
| multimodal | MIXTURE | 1.2346 | 0.0235 | 0.5679 | ... |
| multimodal | REALNVP | 1.4568 | 0.0346 | 0.7890 | ... |

## 📖 Documentation

For detailed documentation:

- **[Training Script README](README_threeflows_amf_vi_weights_log.md)** - Complete training guide
- **[Evaluation Script README](README_eval_10_iters.md)** - Evaluation methodology
- **[Comparison Summary](COMPARISON_SUMMARY.md)** - Training vs Evaluation

## 🛠️ Available Flow Types

| Flow | Parametric | Best For |
|------|------------|----------|
| `realnvp` | ✅ | General purpose, stable |
| `maf` | ✅ | Autoregressive modeling |
| `rbig` | ❌ | Non-parametric, rotation-based |


## 🎨 Visualization Examples

Training script generates comprehensive visualizations:

```
Row 1: [Target Data] [AMF-VI Samples] [Training Losses]
Row 2: [RealNVP Samples] [MAF Samples] [RBIG Samples]
```

## ⚙️ Configuration

### Custom Flow Selection

```python
# Use 5 different flow types
custom_flows = ['realnvp', 'maf', 'rbig']

model, _, _ = train_sequential_amf_vi(
    dataset_name='two_moons',
    flow_types=custom_flows
)
```

### Custom Training Parameters

```python
# Edit in threeflows_amf_vi_weights_log.py:
train_epochs = 10000      # Epochs per flow (default: 1000)
weight_epochs = 10000     # Weight learning epochs (default: 500)
alpha = 0.9               # Moving average decay
```

## 📊 Evaluation Metrics

1. **NLL (Negative Log-Likelihood)**: Predictive quality
2. **KL Divergence**: Distribution mismatch (KDE-based)
3. **Wasserstein Distance**: Geometric distance (optimal transport)
4. **Gaussian MMD (Unbiased)**: Kernel-based two-sample test
5. **Gaussian MMD (Biased)**: Alternative estimator

All metrics reported as **mean ± std** over 10 iterations.

## 🔬 Research Workflow

```bash
# Step 1: Train models (~90 min for 7 datasets)
python main/threeflows_amf_vi_weights_log.py

# Step 2: Evaluate models (~70 min for 7 datasets)
python main/eval_10_iters.py

# Step 3: Analyze results
# Open results/comprehensive_evaluation_10_iterations.csv
```

## 📦 Dependencies

```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
POT>=0.8.0  # For Wasserstein distance
```

## 🐛 Troubleshooting

### "No models could be evaluated"
**Solution:** Run training script first to generate model files

### High standard deviations in evaluation
**Solution:** Increase iterations in `eval_10_iters.py`:
```python
results = comprehensive_evaluation(n_iterations=20)
```

### NaN/Inf during training
**Solution:** Already handled - automatic fallback to finite values with logging

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional flow architectures
- More benchmark datasets
- Parallel evaluation
- Advanced weight learning methods

## 📄 License

This project is licensed under the MIT License.

## 📚 Citation

If you use this code in your research:

```bibtex
@software{amfvi2025,
  title={AMF-VI: Adaptive Mixture of Flows for Variational Inference},
  author={Benjamin Wiriyapong},
  year={2025},
  url={https://github.com/benjaminsw/amf-vi}
}
```

## 🔗 Related Work

- **Normalizing Flows**: [arXiv:1908.09257](https://arxiv.org/abs/1908.09257)
- **RealNVP**: [arXiv:1605.08803](https://arxiv.org/abs/1605.08803)
- **MAF**: [arXiv:1705.07057](https://arxiv.org/abs/1705.07057)
- **RBIG**: [arXiv:1602.00229](https://arxiv.org/abs/1602.00229)

---

**Version:** 1.0 | **Last Updated:** 2024-12-01 | **Python:** 3.7+ | **PyTorch:** 1.9+
