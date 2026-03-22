import torch
from src.network import Propagatable


class Identity(Propagatable):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output
