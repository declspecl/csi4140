import os
import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

from src.train import train_epoch, evaluate, train


class FixedClassifier(nn.Module):
    """Always predicts class 0 regardless of input."""

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))
        with torch.no_grad():
            self.bias[0] = 10.0

    def forward(self, x):
        return self.bias.unsqueeze(0).expand(x.shape[0], -1)


def make_loader(batch_size: int = 4, num_batches: int = 1, all_class_0: bool = True):
    batches = []
    for _ in range(num_batches):
        images = torch.randn(batch_size, 4)
        labels = torch.zeros(batch_size, dtype=torch.long) if all_class_0 else torch.arange(batch_size) % 3
        batches.append((images, labels))
    return batches


class TestTrainEpoch:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(0)
        model = FixedClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = make_loader(batch_size=4)
        return model, criterion, optimizer, loader

    def test_returns_avg_loss_and_accuracy(self, setup):
        model, criterion, optimizer, loader = setup
        avg_loss, acc = train_epoch(model, loader, criterion, optimizer, "cpu")
        assert avg_loss > 0
        assert acc == pytest.approx(1.0)

    def test_partial_accuracy(self):
        model = FixedClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # 4 samples: 2 with label 0 (correct), 2 with label 1 (wrong)
        images = torch.randn(4, 4)
        labels = torch.tensor([0, 0, 1, 1])
        loader = [(images, labels)]
        _, acc = train_epoch(model, loader, criterion, optimizer, "cpu")
        assert acc == pytest.approx(0.5)

    def test_sets_training_mode(self, setup):
        model, criterion, optimizer, loader = setup
        model.eval()
        train_epoch(model, loader, criterion, optimizer, "cpu")
        assert model.training is True

    def test_optimizer_zero_grad_and_step_called(self, setup):
        model, criterion, _, loader = setup
        real_optim = torch.optim.SGD(model.parameters(), lr=0.01)
        mock_optim = MagicMock(wraps=real_optim)
        train_epoch(model, loader, criterion, mock_optim, "cpu")
        assert mock_optim.zero_grad.call_count == 1
        assert mock_optim.step.call_count == 1

    def test_iteration_losses_appended(self, setup):
        model, criterion, optimizer, loader = setup
        losses = []
        train_epoch(model, loader, criterion, optimizer, "cpu", iteration_losses=losses)
        assert len(losses) == 1
        assert losses[0] > 0

    def test_multiple_batches(self):
        model = FixedClassifier()
        criterion = nn.CrossEntropyLoss()
        real_optim = torch.optim.SGD(model.parameters(), lr=0.01)
        mock_optim = MagicMock(wraps=real_optim)
        loader = make_loader(batch_size=2, num_batches=3)
        avg_loss, acc = train_epoch(model, loader, criterion, mock_optim, "cpu")
        assert avg_loss > 0
        assert acc == pytest.approx(1.0)
        assert mock_optim.step.call_count == 3

    def test_empty_loader(self):
        model = FixedClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        avg_loss, acc = train_epoch(model, [], criterion, optimizer, "cpu")
        assert avg_loss == 0.0
        assert acc == 0.0


class TestEvaluate:
    @pytest.fixture
    def setup(self):
        model = FixedClassifier()
        criterion = nn.CrossEntropyLoss()
        loader = make_loader(batch_size=4)
        return model, criterion, loader

    def test_returns_avg_loss_and_accuracy(self, setup):
        model, criterion, loader = setup
        avg_loss, acc = evaluate(model, loader, criterion, "cpu")
        assert avg_loss > 0
        assert acc == pytest.approx(1.0)

    def test_sets_eval_mode(self, setup):
        model, criterion, loader = setup
        model.train()
        evaluate(model, loader, criterion, "cpu")
        assert model.training is False

    def test_no_gradients_computed(self, setup):
        model, criterion, loader = setup
        evaluate(model, loader, criterion, "cpu")
        for p in model.parameters():
            assert p.grad is None

    def test_empty_loader(self):
        model = FixedClassifier()
        criterion = nn.CrossEntropyLoss()
        avg_loss, acc = evaluate(model, [], criterion, "cpu")
        assert avg_loss == 0.0
        assert acc == 0.0

    def test_multiple_batches(self):
        model = FixedClassifier()
        criterion = nn.CrossEntropyLoss()
        loader = make_loader(batch_size=2, num_batches=3)
        avg_loss, acc = evaluate(model, loader, criterion, "cpu")
        assert avg_loss > 0
        assert acc == pytest.approx(1.0)


class TestTrain:
    @pytest.fixture
    def setup(self, tmp_path):
        torch.manual_seed(0)
        model = FixedClassifier()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = make_loader(batch_size=2)
        cp = str(tmp_path / "best.pt")
        return model, criterion, optimizer, loader, cp

    def test_returns_history_keys(self, setup):
        model, criterion, optimizer, loader, cp = setup
        history = train(model, loader, loader, criterion, optimizer, epochs=2, checkpoint_path=cp)
        assert set(history.keys()) == {"train_loss", "train_acc", "test_loss", "test_acc", "iteration_loss"}

    def test_history_length(self, setup):
        model, criterion, optimizer, loader, cp = setup
        history = train(model, loader, loader, criterion, optimizer, epochs=3, checkpoint_path=cp)
        for key in ("train_loss", "train_acc", "test_loss", "test_acc"):
            assert len(history[key]) == 3

    def test_checkpoint_saved(self, setup):
        model, criterion, optimizer, loader, cp = setup
        train(model, loader, loader, criterion, optimizer, epochs=1, checkpoint_path=cp)
        assert os.path.exists(cp)

    def test_checkpoint_is_state_dict(self, setup):
        model, criterion, optimizer, loader, cp = setup
        train(model, loader, loader, criterion, optimizer, epochs=1, checkpoint_path=cp)
        state = torch.load(cp, weights_only=True)
        assert isinstance(state, dict)
        for key in state:
            assert isinstance(key, str)

    def test_scheduler_stepped(self, setup):
        model, criterion, optimizer, loader, cp = setup
        scheduler = MagicMock()
        train(model, loader, loader, criterion, optimizer, scheduler=scheduler, epochs=4, checkpoint_path=cp)
        assert scheduler.step.call_count == 4

    def test_no_scheduler(self, setup):
        model, criterion, optimizer, loader, cp = setup
        history = train(model, loader, loader, criterion, optimizer, scheduler=None, epochs=1, checkpoint_path=cp)
        assert len(history["train_loss"]) == 1

    def test_prints_epoch_summary(self, setup, capsys):
        model, criterion, optimizer, loader, cp = setup
        train(model, loader, loader, criterion, optimizer, epochs=2, checkpoint_path=cp)
        out = capsys.readouterr().out
        assert "Epoch 1/2" in out
        assert "Epoch 2/2" in out
        assert "train_loss" in out
        assert "test_acc" in out
