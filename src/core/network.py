from typing import Sequence

import torch

from .protocols import Propagatable


class NeuralNetwork(Propagatable):
    def __init__(self, propagatables: Sequence[Propagatable]) -> None:
        self.propagatables = propagatables

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        for p in self.propagatables:
            x = p.forward(x)
        return x

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        grad = grad_output
        for p in reversed(self.propagatables):
            grad = p.backward(grad)
        return grad
