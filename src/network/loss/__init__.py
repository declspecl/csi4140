from typing import Protocol

import torch


class Loss(Protocol):
    def calculate_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...
    def calculate_gradient(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...
