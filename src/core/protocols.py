from enum import Enum
from typing import Dict, List, Protocol

import torch
import torch.nn as nn


class ParameterType(Enum):
    WEIGHT = "weight"
    BIAS = "bias"


class Propagatable(Protocol):
    def forward(self, input: torch.Tensor) -> torch.Tensor: ...
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor: ...


class Activation(Propagatable, Protocol):
    pass


class Layer(Propagatable, Protocol):
    def parameters(self) -> Dict[ParameterType, nn.Parameter]: ...


class Loss(Protocol):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...
    def backward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...


class Regularizer(Protocol):
    def compute_penalty(self, parameters: List[nn.Parameter]) -> torch.Tensor: ...
    def compute_gradient(self, parameter: nn.Parameter) -> torch.Tensor: ...


class Updater(Protocol):
    def step(self, parameters: Dict[ParameterType, List[nn.Parameter]]) -> None: ...
