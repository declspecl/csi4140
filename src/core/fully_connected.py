from typing import Dict

import torch
import torch.nn as nn

from .protocols import Layer, ParameterType


class FullyConnected(Layer):
    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features

        std = torch.sqrt(torch.tensor(2.0 / in_features))
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias = nn.Parameter(torch.zeros(out_features, 1))

        self._input_cache = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._input_cache = input
        z = torch.mm(self.weight, input) + self.bias
        return z

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self._input_cache is None:
            raise ValueError("Must call forward() before backward()")

        x = self._input_cache

        self.weight.grad = torch.mm(grad_output, x.t())
        self.bias.grad = grad_output.sum(dim=1, keepdim=True)
        grad_input = torch.mm(self.weight.t(), grad_output)

        return grad_input

    def parameters(self) -> Dict[ParameterType, nn.Parameter]:
        return {ParameterType.WEIGHT: self.weight, ParameterType.BIAS: self.bias}
