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
        m = self.beta1 * self.m[id(param)] + (1.0 - self.beta1) * grad
        v = self.beta2 * self.v[id(param)] + (1.0 - self.beta2) * grad**2
        self.m[id(param)] = m
        self.v[id(param)] = v

        m_hat = m / (1.0 - self.beta1**self.t)
        v_hat = v / (1.0 - self.beta2**self.t)

        param.data -= self.learning_rate * m_hat / (torch.sqrt(v_hat) + self.epsilon)
