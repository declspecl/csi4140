from typing import Protocol


class Scheduler(Protocol):
    def step(self) -> None: ...
