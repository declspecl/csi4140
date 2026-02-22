import torch
from typing import Sequence
from src.network import Propagatable
from src.network.layer import Layer


class NeuralNetwork(Propagatable):
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input
        for layer in self.layers:
            output = layer.forward(output)

        return output


    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return grad
