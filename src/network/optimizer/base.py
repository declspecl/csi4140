from typing import Optional, Sequence

import torch
import torch.nn as nn

from src.network import ParameterType
from src.network.layer import Layer
from src.network.optimizer import Optimizer
from src.network.regularizer import Regularizer


class BaseOptimizer(Optimizer):
    def __init__(self, layers: Sequence[Layer], learning_rate: float, regularizer: Optional[Regularizer] = None):
        self.params = [(param_type, param) for layer in layers for param_type, param in layer.parameters().items()]
        self._learning_rate = learning_rate
        self.regularizer = regularizer

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"Learning rate must be positive, got {value}")
        self._learning_rate = value

    def zero_grad(self) -> None:
        for _, param in self.params:
            param.grad = None

    def step(self) -> None:
        for param_type, param in self.params:
            if param.grad is None:
                continue
            grad = (
                self.regularizer.apply(param) if self.regularizer and param_type == ParameterType.WEIGHT else param.grad
            )
            self._update(param, grad)

    def _update(self, param: nn.Parameter, grad: torch.Tensor) -> None:
        raise NotImplementedError
