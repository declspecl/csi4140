import torch
from src.network.loss import Loss


class CrossEntropy(Loss):
    def calculate_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            logits: Raw unnormalized scores. Shape: (n_classes, n_samples)
            ```
            [
                [2.3, -1.2, 0.8],
                [0.6, 3.1, -0.5],
                [-1.1, 0.7, 2.1]
            ]
            ```
            targets: Class indices. Shape: (n_samples,)
            ```
            [1, 2, 0]
            ```
        """
        batch_size = logits.shape[1]

        # Softmax
        logits_max = logits.max(dim=0, keepdim=True)[0]
        logits_shifted = logits - logits_max
        log_sum_exp = torch.log(torch.exp(logits_shifted).sum(dim=0, keepdim=True))
        log_probs = logits_shifted - log_sum_exp

        # Cross-entropy
        loss = -log_probs[targets, range(batch_size)].mean()

        return loss


    def calculate_gradient(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = logits.shape[1]

        logits_max = logits.max(dim=0, keepdim=True)[0]
        logits_shifted = logits - logits_max
        exp_logits = torch.exp(logits_shifted)
        softmax = exp_logits / exp_logits.sum(dim=0, keepdim=True)

        grad = softmax.clone()
        grad[targets, range(batch_size)] -= 1
        grad = grad / batch_size

        return grad
