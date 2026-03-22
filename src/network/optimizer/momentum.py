from typing import Optional, Sequence

import torch
import torch.nn as nn

from src.network.layer import Layer
from src.network.optimizer.base import BaseOptimizer
from src.network.regularizer import Regularizer


class Momentum(BaseOptimizer):
    def __init__(
        self,
        layers: Sequence[Layer],
        learning_rate: float,
        beta: float = 0.9,
        regularizer: Optional[Regularizer] = None,
    ):
        super().__init__(layers, learning_rate, regularizer)
        self.beta = beta
        self.velocity = {id(param): torch.zeros_like(param.data) for _, param in self.params}

    def _update(self, param: nn.Parameter, grad: torch.Tensor) -> None:
        pid = id(param)
        # FIX: lazy device move + in-place ops (same rationale as Adam)
        if self.velocity[pid].device != param.device:
            self.velocity[pid] = self.velocity[pid].to(param.device)
        self.velocity[pid].mul_(self.beta).add_(grad, alpha=1.0 - self.beta)
        param.data.add_(self.velocity[pid], alpha=-self.learning_rate)
