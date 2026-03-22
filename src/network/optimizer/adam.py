from typing import Optional, Sequence

import torch
import torch.nn as nn

from src.network.layer import Layer
from src.network.optimizer.base import BaseOptimizer
from src.network.regularizer import Regularizer


class Adam(BaseOptimizer):
    def __init__(
        self,
        layers: Sequence[Layer],
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        regularizer: Optional[Regularizer] = None,
    ):
        super().__init__(layers, learning_rate, regularizer)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = {id(param): torch.zeros_like(param.data) for _, param in self.params}
        self.v = {id(param): torch.zeros_like(param.data) for _, param in self.params}

    def step(self) -> None:
        self.t += 1
        super().step()

    def _update(self, param: nn.Parameter, grad: torch.Tensor) -> None:
        pid = id(param)
        # FIX: optimizer state (m, v) is created on CPU at init time. When params
        # are moved to GPU via model.to(device), the state must follow. We lazily
        # move on first use rather than requiring a separate to(device) call.
        if self.m[pid].device != param.device:
            self.m[pid] = self.m[pid].to(param.device)
            self.v[pid] = self.v[pid].to(param.device)

        # FIX: use in-place ops (mul_, add_, addcmul_, addcdiv_) instead of
        # creating new tensors. The original code allocated ~6 intermediates per
        # parameter per step, which compounded across all params to cause OOM.
        self.m[pid].mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
        self.v[pid].mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

        bias1 = 1.0 - self.beta1**self.t
        bias2 = 1.0 - self.beta2**self.t

        step_size = self.learning_rate / bias1
        denom = (self.v[pid] / bias2).sqrt_().add_(self.epsilon)
        param.data.addcdiv_(self.m[pid], denom, value=-step_size)
