import torch
from typing import Protocol


class Activation(Protocol):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

