import torch

from src.network import Propagatable


class ReLU(Propagatable):
    def __init__(self):
        self._cached_input = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._cached_input = input
        return (input + torch.abs(input)) / 2

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self._cached_input is None:
            raise ValueError("Must call forward() before backward()")
        assert grad_output.dtype == self._cached_input.dtype, (
            f"dtype mismatch: grad_output={grad_output.dtype}, input={self._cached_input.dtype}"
        )

        return grad_output * (self._cached_input > 0).to(grad_output.dtype)
