from typing import Protocol
from src.network import Propagatable


class Activation(Propagatable, Protocol):
    pass
