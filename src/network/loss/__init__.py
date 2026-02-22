import torch
from typing import Protocol


class Loss(Protocol):
    def calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...
    def calculate_gradient(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...
