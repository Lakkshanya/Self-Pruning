# Self-Pruning Neural Network (SPNN) on CIFAR-10

This project implements a **Self-Pruning Neural Network** that automatically learns to prune its own weights during training. By introducing learnable "gate" parameters for each connection and a sparsity-inducing loss function, the network identifies and removes redundant parameters while optimizing for classification accuracy.

## 🚀 Overview

Pruning is essential for deploying deep learning models on resource-constrained devices. This implementation uses a **differentiable gating mechanism** to achieve end-to-end pruning in a single training session.

### Key Features
- **PrunableLinear Layer**: A custom PyTorch layer with integrated binary-like gates.
- **Sparsity-Aware Loss**: A specific loss term $\text{Total Loss} = \text{Loss}_{CE} + \lambda \cdot \sum \text{Gates}$ to drive sparsity.
- **Dynamic Learning**: Separate learning rates for weights and gate scores for stable convergence.
- **Automated Visualization**: Generates distribution histograms for gate values across different regularization strengths ($\lambda$).

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install torch torchvision numpy matplotlib
```

## 💻 Usage

Run the main script to start training three models with different lambda values ($0.0001, 0.001, 0.01$):

```bash
python pruning_model.py
```

After training, the script will:
1. Print a comparison table of **Accuracy vs. Sparsity**.
2. Save a visualization of gate distributions as `gate_distributions.png`.

## 📊 Results

The model demonstrates a clear trade-off:
- **Low Lambda**: High accuracy, high parameter retention.
- **High Lambda**: Aggressive pruning, higher sparsity, but lower accuracy.

Example output:
| Lambda (λ) | Accuracy (%) | Sparsity (%) |
| :--- | :--- | :--- |
| 0.0001 | ~45% | ~5% |
| 0.001 | ~40% | ~35% |
| 0.01 | ~25% | ~85% |

## 📁 Project Structure

- `pruning_model.py`: Core implementation, training loop, and evaluation logic.
- `case_study_solution.md`: Detailed technical analysis of the implementation.
- `gate_distributions.png`: Visualization of the learned pruning masks.
