import torch
from typing import Sequence
from src.network.layer import Layer


class NeuralNetwork:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
        self.training = True

    def to(self, device: str) -> "NeuralNetwork":
        """FIX: model params (nn.Parameter) are created on CPU. DataLoader tensors
        are moved to GPU via .to(device) in the training loop, causing device
        mismatch errors. This moves all params to the target device."""
        for layer in self.layers:
            for _, param in layer.parameters().items():
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train(self) -> None:
        self.training = True
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self) -> None:
        self.training = False
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False

    def parameters(self) -> list[Layer]:
        return self.layers
