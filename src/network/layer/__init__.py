import torch.nn as nn
from typing import Protocol, Dict
from src.network import ParameterType, Propagatable
from src.network.activation import Activation


class Layer(Propagatable, Protocol):
    def parameters(self) -> Dict[ParameterType, nn.Parameter]: ...
