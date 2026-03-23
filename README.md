# CSI 4140 Group Project: CIFAR-10 DNN

Anton Sakhanovych and Gavin D'Hondt

## How to Run

Requires Python 3.13 and [uv](https://github.com/astral-sh/uv).

```bash
uv run main
```

This trains the CNN on CIFAR-10 (downloaded automatically to `./data/`) and saves the best checkpoint. Training and test metrics are printed each epoch.

To run the ablation experiments:

```bash
uv run python run_ablation.py
```

To run tests:

```bash
uv run pytest
```

## Code Structure

`src/main.py` is the entry point. It builds the model, sets up the data loaders and optimizer, and calls the training loop in `src/train.py`, which handles epoch iteration, evaluation, and best-checkpoint saving.

The model is defined in `src/models/cifar10_cnn.py` and composed from components in `src/network/`. That directory contains all the from-scratch implementations: layers (convolutional, fully connected, dropout, flatten), activations (ReLU, Sigmoid, Softmax), optimizers (SGD, SGD+Momentum, Adam), learning rate schedulers (cosine decay, step decay), L2 regularization, and cross-entropy loss. None of these use `nn.Module` or `torch.optim`.

`src/data/cifar10.py` handles CIFAR-10 loading, per-channel normalization, and training augmentation (random flip + crop).

`tests/` contains unit tests and numerical gradient checks for all layers. `run_ablation.py` is the CLI runner for the ablation experiments, and `experiments/ablation.ipynb` is the Colab notebook version. The report and presentation are in `docs/`.
