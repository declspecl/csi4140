import torch
from typing import Protocol, runtime_checkable


@runtime_checkable
class Layer(Protocol):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        ...

