import torch
from . import Layer
from .activation import Activation


class FullyConnected(Layer):
    def __init__(self, in_features: int, out_features: int, activation: Activation):
        self.weight = torch.randn(in_features, out_features) * (2.0 / in_features) ** 0.5
        self.bias = torch.zeros(out_features)
        self.activation = activation


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), -1)

        return self.activation.forward(inputs @ self.weight + self.bias)

