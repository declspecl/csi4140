import torch
import torch.nn as nn
from typing import Protocol


class Regularizer(Protocol):
    def apply(self, param: nn.Parameter) -> torch.Tensor: ...
