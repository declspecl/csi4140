import torch

from src.core.protocols import Propagatable


class Identity(Propagatable):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output


class Sigmoid(Propagatable):
    def __init__(self):
        self._cached_output = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = 1 / (1 + torch.exp(-input))
        self._cached_output = output
        return output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self._cached_output is None:
            raise ValueError("Must call forward() before backward()")

        a = self._cached_output
        grad_input = grad_output * a * (1 - a)
        return grad_input
