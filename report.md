
---

# 🧾 8. `report.md` (clean + human)

```markdown
# Self-Pruning Neural Network Report

## Overview

This project explores a neural network that can prune its own weights during training. Each weight is associated with a learnable gate that determines its importance. If the gate value becomes very small, the corresponding weight is effectively removed.

---

## Why L1 Regularization Encourages Sparsity

An L1 penalty was applied to the gate values. Since the gate outputs lie between 0 and 1, minimizing their sum pushes many of them toward zero. This results in a sparse network where only important connections remain active.

---

## Results

| Lambda | Test Accuracy (%) | Sparsity (%) |
|--------|------------------|-------------|
| 1e-05  | 56.78            | 68.79       |
| 5e-05  | 56.43            | 96.30       |
| 1e-04  | 55.73            | 99.09       |

---

## Observations

At higher values of λ, the model achieves extremely high sparsity (up to 99%), meaning that only a very small subset of weights remains active.

Despite this aggressive pruning, the model retains competitive accuracy, indicating that the original network contained significant redundancy.

This highlights the effectiveness of the gating mechanism in identifying and removing less important connections.
---

## Conclusion

The experiment shows that a neural network can learn to prune itself during training using a simple gating mechanism. This approach can reduce model size while maintaining acceptable performance. 