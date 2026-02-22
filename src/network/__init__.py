import torch
from enum import Enum
from typing import Protocol


class Propagatable(Protocol):
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor: ...


class ParameterType(Enum):
    WEIGHT = "weight"
    BIAS = "bias"
