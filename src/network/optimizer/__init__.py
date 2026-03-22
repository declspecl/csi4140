from typing import Protocol


class Optimizer(Protocol):
    learning_rate: float

    def step(self) -> None: ...
