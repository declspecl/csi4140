import torch
import torch.nn as nn
from typing import Dict
from src.network import ParameterType
from src.network.layer import Layer
from src.network.activation import Activation


class FullyConnected(Layer):
    def __init__(self, in_features: int, out_features: int, activation: Activation):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        std = torch.sqrt(torch.tensor(2.0 / in_features))
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.bias = nn.Parameter(torch.zeros(out_features, 1))

        self._input_cache = None


    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._input_cache = input
        z = torch.mm(self.weight, input) + self.bias
        a = self.activation.forward(z)

        return a


    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self._input_cache is None:
            raise ValueError("Must call forward() before backward()")

        grad_z = self.activation.backward(grad_output)
        self.weight.grad = torch.mm(grad_z, self._input_cache.t())
        self.bias.grad = grad_z.sum(dim=1, keepdim=True)
        grad_a = torch.mm(self.weight.t(), grad_z)

        return grad_a


    def parameters(self) -> Dict[ParameterType, nn.Parameter]:
        return {
            ParameterType.WEIGHT: self.weight,
            ParameterType.BIAS: self.bias
        }
