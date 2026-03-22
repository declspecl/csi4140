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
        v = self.beta * self.velocity[id(param)] + (1.0 - self.beta) * grad
        self.velocity[id(param)] = v
        param.data -= self.learning_rate * v
