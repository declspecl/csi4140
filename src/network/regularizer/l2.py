import torch
import torch.nn as nn

from src.network.regularizer import Regularizer


class L2(Regularizer):
    def __init__(self, lambda_: float):
        self.lambda_ = lambda_

    def apply(self, param: nn.Parameter) -> torch.Tensor:
        assert param.grad is not None
        return param.grad + 2.0 * self.lambda_ * param.data
