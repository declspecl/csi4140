import torch.nn as nn
from typing import Sequence, Optional
from src.network.layer import Layer
from src.network.optimizer.base import BaseOptimizer
from src.network.regularizer import Regularizer


class SGD(BaseOptimizer):
    def __init__(self, layers: Sequence[Layer], learning_rate: float, regularizer: Optional[Regularizer] = None):
        super().__init__(layers, learning_rate, regularizer)

    def _update(self, param: nn.Parameter, grad) -> None:
        param.data -= self.learning_rate * grad
