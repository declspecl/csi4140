from src.network.optimizer import Optimizer
from src.network.scheduler import Scheduler


class StepDecay(Scheduler):
    def __init__(self, optimizer: Optimizer, step_size: int, decay_factor: float = 0.1, lr_min: float = 0.0):
        self.optimizer = optimizer
        self.lr_initial = optimizer.learning_rate
        self.step_size = step_size
        self.decay_factor = decay_factor
        self.lr_min = lr_min
        self.t = 0

    def step(self) -> None:
        self.t += 1
        if self.t % self.step_size == 0:
            new_lr = self.optimizer.learning_rate * self.decay_factor
            self.optimizer.learning_rate = max(new_lr, self.lr_min)
