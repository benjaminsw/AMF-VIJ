# Flow Performance Testing

This directory contains implementations of various normalizing flows and a comprehensive testing script to evaluate their performance.

## Implemented Flows

1. **RealNVP Flow** - Coupling-based flow with efficient forward/inverse computation
2. **Planar Flow** - Simple planar transformations
3. **Radial Flow** - Radial transformations from a center point
4. **MAF (Masked Autoregressive Flow)** - Autoregressive flow with efficient forward pass
5. **IAF (Inverse Autoregressive Flow)** - Autoregressive flow with efficient inverse pass
6. **Gaussianization Flow** - Based on the paper "Gaussianization Flows" with trainable kernel layers and Householder reflections

## Quick Start

### Running the Flow Test

```bash
# From the project root directory
python main/test_flows.py
```

This will:
- Test all flow types with different numbers of layers (2, 4, 6, 8)
- Train each configuration on synthetic 2D data
- Evaluate training success, final loss, parameter count, and training time
- Generate comparison plots saved to `results/flow_comparison.png`
- Print comprehensive results and recommendations

### Key Features of the Test

- **Automatic flow comparison**: Tests multiple flow architectures
- **Performance metrics**: Loss, training time, parameter efficiency
- **Robustness testing**: Checks sampling and log probability computation
- **Visual analysis**: Generates detailed comparison plots
- **Best configuration finder**: Identifies optimal layer counts for each flow

### Expected Output

The test will print:
1. **Individual flow results** - Training progress for each flow/layer combination
2. **Summary table** - Compact overview of all results
3. **Best configurations** - Optimal settings for each flow type
4. **Recommendations** - Top-performing flows for the dataset

### Gaussianization Flow Highlights

The Gaussianization Flow implementation includes:
- **Trainable kernel layers** using mixture of logistic distributions
- **Householder reflection layers** for efficient orthogonal transformations
- **Data-dependent initialization** for better convergence
- **Numerical stability** improvements for 2D applications

### Customization

You can modify the test by:
- Changing the dataset: Edit `dataset_name` in `main()`
- Adjusting layer configurations: Modify `layer_configs`
- Tuning flow parameters: Update the `flow_configs` dictionary
- Adding new flows: Implement in `amf_vi/flows/` and add to test

### Available Datasets

- `two_moons` - Two crescent shapes
- `multimodal` - Multiple Gaussian clusters  
- `rings` - Concentric circles
- `banana` - Banana-shaped distribution
- `x_shape` - X-shaped mixture of Gaussians

## Results Interpretation

- **Final Loss**: Lower is better (< 1.0 is very good)
- **Success Rate**: Percentage of layer configs that trained successfully
- **Efficiency**: Parameter count vs performance trade-off
- **Training Time**: Computational efficiency

The test provides a comprehensive evaluation to help choose the best flow architecture for your specific use case.