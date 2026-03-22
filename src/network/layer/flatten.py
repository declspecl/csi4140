from typing import Dict

import torch
import torch.nn as nn

from src.network import ParameterType
from src.network.layer import Layer


class Flatten(Layer):
    def __init__(self):
        self._input_shape = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._input_shape = input.shape
        N = input.shape[0]
        return input.reshape(N, -1).t()

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self._input_shape is None:
            raise ValueError("Must call forward() before backward()")
        return grad_output.t().reshape(self._input_shape)

    def parameters(self) -> Dict[ParameterType, nn.Parameter]:
        return {}
