import torch
from typing import Sequence
from src.network.layer import Layer


class NeuralNetwork:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
        self.training = True

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
