import torch
import torch.nn as nn
from typing import Dict

from src.network import ParameterType
from src.network.activation import Activation


class ActivationLayer:
    # Adapter pattern (https://refactoring.guru/design-patterns/adapter)
    # Wraps Activation to satisfy the Layer protocol (adds parameters())
    def __init__(self, activation: Activation):
        self._activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._activation.forward(input)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        return self._activation.backward(grad_output)

    def parameters(self) -> Dict[ParameterType, nn.Parameter]:
        return {}
