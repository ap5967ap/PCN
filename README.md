# Point Cloud Completion using PCN

This repository contains the implementation of the Point Completion Network (PCN) for point cloud completion tasks. The project is based on the paper [**PCN: Point Completion Network**](https://arxiv.org/pdf/1808.00671) and has been trained and tested on the [**ShapeNet**](https://drive.google.com/file/d/1OvvRyx02-C_DkzYiJ5stpin0mnXydHQ7/view?pli=1) dataset.

## Overview

Point cloud completion aims to predict the complete shape of an object given its partial point cloud. This project uses PyTorch to implement the PCN architecture and replicates the results from the original paper with slight variations in metrics.

---

## Dataset

The **ShapeNet** dataset was used for training and testing the model. Testing was done specifically on the "car" and "bus" categories for seen and unseen datasets respectively. 

---

## Results

The model achieved competitive accuracy with the original paper's results using Chamfer Distance (CD) and Earth Mover's Distance (EMD) as evaluation metrics. Below are the obtained metrics:

| Dataset       | Our CD | Our EMD | Paper CD | Paper EMD |
|---------------|------------------------|------------------------------|----------|-----------|
| Car (Seen)    | 0.009                 | 0.016                        | 0.008    | 0.015     |
| Bus (Unseen)    | 0.010                 | 0.020                        | 0.009    | 0.009     |

---

## Implementation Details

### Architecture

The PCN architecture is divided into two main components: the **Encoder** and the **Decoder**.

#### Encoder
The encoder extracts a global feature vector from the partial point cloud input:
- **Input**: A partial point cloud with shape `(N, 3)` where `N` is the number of points.
- **Feature Extraction**:
  - Uses shared Multi-Layer Perceptrons (MLPs) with point-wise operations.
  - Applies a **max pooling layer** to aggregate the point-wise features into a global feature vector of fixed size (e.g., 1024 dimensions).

#### Decoder
The decoder generates the completed point cloud in two stages:
1. **Coarse Prediction**:
   - Fully connected layers decode the global feature vector into a coarse point cloud with a fixed number of points (e.g., 1024 points).
   - This stage provides a rough approximation of the completed shape.
2. **Dense Prediction**:
   - Uses folding-based methods to refine the coarse prediction into a dense point cloud (e.g., 16,384 points).
   - Employs two MLP-based folding layers to map the global feature and 2D grid into a detailed 3D point cloud.

#### Summary of Network Flow
1. **Input**: Partial point cloud → Encoder (Global Feature)
2. **Stage 1**: Global Feature → Coarse Point Cloud
3. **Stage 2**: Coarse Point Cloud + Folding Layers → Dense Point Cloud
4. **Output**: Completed point cloud.

### Loss Function

The loss function is defined as:

```python
loss1 = cd_loss_L1(coarse_pred, c)
loss2 = cd_loss_L1(dense_pred, c)
loss = loss1 + alpha * loss2
```

- `cd_loss_L1`: Chamfer Distance with L1 norm.
- `c`: Ground truth complete point cloud.
- `alpha`: Weighting parameter for dense prediction loss.

#### Alpha Scheduling
The value of `alpha` changes dynamically during training:
- `train_step < 10,000`: `alpha = 0.01`
- `10,000 ≤ train_step < 20,000`: `alpha = 0.1`
- `20,000 ≤ train_step < 50,000`: `alpha = 0.5`
- `train_step ≥ 50,000`: `alpha = 1.0`

### Hyperparameters

- **Learning Rate**: 0.0001
- **Epochs**: 100
- **Batch Size**: 32
- **Number of Workers**: 8




## Acknowledgments

- **Paper**: [PCN: Point Completion Network](https://arxiv.org/abs/1808.00671)
- **Dataset**: ShapeNet

---


## Contributors
- [Aditya Priyadarshi](https://github.com/ap5967ap)
- [Ananthakrishna K](https://github.com/Ananthakrishna-K-13)