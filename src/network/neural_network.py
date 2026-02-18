import torch
from typing import List
from .layer import Layer


class NeuralNetwork:
    def __init__(self, layers: List[Layer]):
        self.layers = layers


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)

        return inputs
