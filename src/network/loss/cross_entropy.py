import torch
from src.network.loss import Loss


class CrossEntropy(Loss):
    def calculate_loss(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = probs.shape[1]

        # FIX: clamp to avoid log(0) = -inf when softmax output is near zero
        loss = -torch.log(probs[targets, range(batch_size)].clamp(min=1e-12)).mean()

        return loss

    def calculate_gradient(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = probs.shape[1]

        grad = torch.zeros_like(probs)
        grad[targets, range(batch_size)] = -1 / probs[targets, range(batch_size)].clamp(min=1e-12)
        grad = grad / batch_size

        return grad
