import torch
from typing import Dict
from src.network import ParameterType
from src.network.layer import Layer


class Dropout(Layer):
    def __init__(self, p: float = 0.5):
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
        self.training = True
        self._mask = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return input
        # Inverted dropout: scale by 1/(1-p) so expected value is preserved
        self._mask = (torch.rand_like(input) > self.p).to(input.dtype) / (1.0 - self.p)
        return input * self._mask

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0 or self._mask is None:
            return grad_output
        return grad_output * self._mask

    def parameters(self) -> Dict[ParameterType, torch.nn.Parameter]:
        return {}
