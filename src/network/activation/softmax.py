import torch

from src.network import Propagatable


class Softmax(Propagatable):
    def __init__(self):
        self._cached_output = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_max = input.max(dim=0, keepdim=True)[0]
        input_shifted = input - input_max
        exp_input = torch.exp(input_shifted)
        output = exp_input / exp_input.sum(dim=0, keepdim=True)

        self._cached_output = output
        return output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self._cached_output is None:
            raise ValueError("Must call forward() before backward()")

        y = self._cached_output
        sum_term = (grad_output * y).sum(dim=0, keepdim=True)
        grad_input = y * (grad_output - sum_term)

        return grad_input
