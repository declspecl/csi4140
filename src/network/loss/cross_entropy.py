import torch
from src.network.loss import Loss


class CrossEntropy(Loss):
    def calculate_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = probs.shape[1]

        loss = -torch.log(probs[targets, range(batch_size)]).mean()

        return loss

    def calculate_gradient(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = probs.shape[1]

        grad = torch.zeros_like(probs)
        grad[targets, range(batch_size)] = -1 / probs[targets, range(batch_size)]
        grad = grad / batch_size

        return grad
