# Self-Pruning Neural Network

This project implements a neural network that dynamically prunes its own weights during training using learnable gating mechanisms.

## Key Features
- Custom PyTorch layer (PrunableLinear)
- No use of torch.nn.Linear
- Differentiable pruning via sigmoid gates
- L1-based sparsity regularization
- Trade-off analysis between sparsity and accuracy

## How to Run

```bash
pip install -r requirements.txt
python main.py