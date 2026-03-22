import math
from src.network.optimizer import Optimizer
from src.network.scheduler import Scheduler


class CosineDecay(Scheduler):
    def __init__(self, optimizer: Optimizer, epochs: int, lr_min: float = 0.0):
        self.optimizer = optimizer
        self.lr_initial = optimizer.learning_rate
        self.lr_min = lr_min
        self.epochs = epochs
        self.t = 0

    def step(self) -> None:
        self.t += 1
        self.optimizer.learning_rate = (
            self.lr_min + 0.5 * (self.lr_initial - self.lr_min) * (1.0 + math.cos(math.pi * self.t / self.epochs))
        )
