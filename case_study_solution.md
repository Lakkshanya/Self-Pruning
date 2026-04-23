# Case Study: Implementation of Self-Pruning Neural Networks (SPNN)

## 1. Introduction
Modern deep learning models are often over-parameterized, leading to high computational costs and memory requirements. **Pruning** is a technique used to remove redundant connections (weights) from a neural network without significantly compromising its accuracy. 

This case study explores the implementation of a **Self-Pruning Neural Network (SPNN)** on the CIFAR-10 dataset using PyTorch. Unlike traditional pruning which is often a post-processing step, SPNN learns which weights to prune during the training process itself.

## 2. Problem Statement
The objective is to design a neural network that automatically identifies and removes unnecessary parameters during training. We aim to achieve high **sparsity** (percentage of zero weights) while maintaining competitive performance.

## 3. Methodology

### 3.1 Gating Mechanism
A custom layer, `PrunableLinear`, is implemented to replace standard `nn.Linear` layers. For every weight $W_{ij}$, we introduce a learnable **gate score** $s_{ij}$.
- The gate value is calculated using a sigmoid function: $g_{ij} = \sigma(s_{ij})$.
- The effective weight used in the forward pass is: $W'_{ij} = W_{ij} \cdot g_{ij}$.

### 3.2 Sparsity-Aware Loss Function
To encourage pruning, the loss function is augmented with a penalty term that drives the gate values towards zero:
$$\text{Total Loss} = \text{CrossEntropyLoss} + \lambda \cdot \sum \sigma(\text{gate\_scores})$$
Where:
- $\lambda$ (Lambda) is the regularization strength.
- The penalty induces sparsity by forcing the network to justify the "cost" of keeping each connection active.

### 3.3 Training Strategy
- **Optimization**: Two different learning rates are used—low for weights (0.001) and higher for gate scores (0.01)—to allow the pruning mask to evolve quickly.
- **Dataset**: CIFAR-10, normalized and flattened for a Multi-Layer Perceptron (MLP) architecture.

## 4. Implementation Details

The architecture consists of:
1. **Input Layer**: 3072 features (from $32 \times 32 \times 3$ images).
2. **Hidden Layer 1**: 256 units with ReLU activation and prunable weights.
3. **Hidden Layer 2**: 128 units with ReLU activation and prunable weights.
4. **Output Layer**: 10 units representing CIFAR-10 classes.




## 5. Experimental Results

We tested the model with three different $\lambda$ values to observe the trade-off between performance and sparsity.

| Lambda ($\lambda$) | Accuracy (%) | Sparsity (%) | Observations |
| :--- | :--- | :--- | :--- |
| 0.0001 | High | Low | Model retains most weights; focus is on accuracy. |
| 0.001 | Medium-High | Medium | A balanced trade-off where redundant weights start disappearing. |
| 0.01 | Lower | High | Aggressive pruning occurs, removing a significant portion of the network. |

### Visualization
The training script generates a distribution of gate values. As $\lambda$ increases, the histogram shows a significant peak near $0.0$, indicating that most connections have been effectively "turned off" by the gating mechanism.

## 6. Conclusion
The Self-Pruning Neural Network successfully demonstrates that sparsity can be learned end-to-end. By applying a differentiable mask and a sparsity-aware penalty, we can effectively compress a model during training. This approach is highly beneficial for deploying large-scale models on edge devices with limited memory and compute power.
