import torch
from . import Activation

class Identity(Activation):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

