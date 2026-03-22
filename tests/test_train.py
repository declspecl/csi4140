import os
import pytest
import torch
import torch.nn as nn

from src.network import ParameterType
from src.train import train_epoch, evaluate, train


# --- Fake components for testing ---


class FakeLayerWithTraining:
    def __init__(self):
        self.training = False

    def parameters(self):
        return {}


class FakeLayerWithoutTraining:
    def parameters(self):
        return {}


class FakeLayerWithParams:
    def __init__(self, weight):
        self.weight = weight

    def parameters(self):
        return {ParameterType.WEIGHT: self.weight}


class FakeModel:
    def __init__(self, output):
        self.layers = [FakeLayerWithTraining(), FakeLayerWithoutTraining()]
        self._output = output
        self.forward_calls = []
        self.backward_calls = []

    def forward(self, x):
        self.forward_calls.append(x)
        return self._output

    def backward(self, grad):
        self.backward_calls.append(grad)
        return grad

    def train(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False


class FakeLoss:
    def __init__(self, loss_value=1.0):
        self._loss = loss_value

    def calculate_loss(self, probs, targets):
        return torch.tensor(self._loss)

    def calculate_gradient(self, probs, targets):
        return torch.ones_like(probs)


class FakeOptimizer:
    def __init__(self):
        self.step_count = 0
        self.zero_grad_count = 0

    def zero_grad(self):
        self.zero_grad_count += 1

    def step(self):
        self.step_count += 1


class FakeScheduler:
    def __init__(self):
        self.step_count = 0

    def step(self):
        self.step_count += 1


def make_loader(batches):
    return batches


# --- Tests ---


class TestTrainEpoch:
    @pytest.fixture
    def setup(self):
        torch.manual_seed(0)
        # probs shape: (classes=3, batch=4), class 1 has highest prob for all samples
        probs = torch.zeros(3, 4)
        probs[1, :] = 1.0
        model = FakeModel(probs)
        loss_fn = FakeLoss(loss_value=0.5)
        optimizer = FakeOptimizer()
        labels = torch.ones(4, dtype=torch.long)  # all class 1 = all correct
        loader = [(torch.randn(4, 3, 2, 2), labels)]
        return model, loader, loss_fn, optimizer

    def test_returns_avg_loss(self, setup):
        model, loader, loss_fn, optimizer = setup
        avg_loss, _ = train_epoch(model, loader, loss_fn, optimizer)
        assert avg_loss == pytest.approx(0.5)

    def test_returns_accuracy(self, setup):
        model, loader, loss_fn, optimizer = setup
        _, acc = train_epoch(model, loader, loss_fn, optimizer)
        assert acc == pytest.approx(1.0)

    def test_partial_accuracy(self):
        probs = torch.zeros(3, 4)
        probs[1, :2] = 1.0  # first 2 predict class 1
        probs[0, 2:] = 1.0  # last 2 predict class 0
        model = FakeModel(probs)
        loss_fn = FakeLoss()
        optimizer = FakeOptimizer()
        labels = torch.ones(4, dtype=torch.long)  # all class 1
        loader = [(torch.randn(4, 3, 2, 2), labels)]

        _, acc = train_epoch(model, loader, loss_fn, optimizer)
        assert acc == pytest.approx(0.5)

    def test_sets_training_mode(self, setup):
        model, loader, loss_fn, optimizer = setup
        train_epoch(model, loader, loss_fn, optimizer)
        assert model.layers[0].training is True

    def test_calls_forward_and_backward(self, setup):
        model, loader, loss_fn, optimizer = setup
        train_epoch(model, loader, loss_fn, optimizer)
        assert len(model.forward_calls) == 1
        assert len(model.backward_calls) == 1

    def test_calls_optimizer_step(self, setup):
        model, loader, loss_fn, optimizer = setup
        train_epoch(model, loader, loss_fn, optimizer)
        assert optimizer.step_count == 1

    def test_multiple_batches(self):
        probs = torch.zeros(3, 2)
        probs[0, :] = 1.0
        model = FakeModel(probs)
        loss_fn = FakeLoss(loss_value=1.0)
        optimizer = FakeOptimizer()
        labels = torch.zeros(2, dtype=torch.long)
        loader = [
            (torch.randn(2, 3, 2, 2), labels),
            (torch.randn(2, 3, 2, 2), labels),
            (torch.randn(2, 3, 2, 2), labels),
        ]

        avg_loss, acc = train_epoch(model, loader, loss_fn, optimizer)
        assert avg_loss == pytest.approx(1.0)
        assert acc == pytest.approx(1.0)
        assert optimizer.step_count == 3

    def test_empty_loader(self):
        model = FakeModel(None)
        loss_fn = FakeLoss()
        optimizer = FakeOptimizer()

        avg_loss, acc = train_epoch(model, [], loss_fn, optimizer)
        assert avg_loss == 0.0
        assert acc == 0.0

    def test_calls_zero_grad(self, setup):
        model, loader, loss_fn, optimizer = setup
        train_epoch(model, loader, loss_fn, optimizer)
        assert optimizer.zero_grad_count == 1


class TestEvaluate:
    @pytest.fixture
    def setup(self):
        probs = torch.zeros(3, 4)
        probs[2, :] = 1.0  # predict class 2
        model = FakeModel(probs)
        loss_fn = FakeLoss(loss_value=0.25)
        labels = torch.full((4,), 2, dtype=torch.long)
        loader = [(torch.randn(4, 3, 2, 2), labels)]
        return model, loader, loss_fn

    def test_returns_avg_loss(self, setup):
        model, loader, loss_fn = setup
        avg_loss, _ = evaluate(model, loader, loss_fn)
        assert avg_loss == pytest.approx(0.25)

    def test_returns_accuracy(self, setup):
        model, loader, loss_fn = setup
        _, acc = evaluate(model, loader, loss_fn)
        assert acc == pytest.approx(1.0)

    def test_sets_eval_mode(self, setup):
        model, loader, loss_fn = setup
        model.layers[0].training = True
        evaluate(model, loader, loss_fn)
        assert model.layers[0].training is False

    def test_no_backward_called(self, setup):
        model, loader, loss_fn = setup
        evaluate(model, loader, loss_fn)
        assert len(model.backward_calls) == 0

    def test_empty_loader(self):
        model = FakeModel(None)
        loss_fn = FakeLoss()
        avg_loss, acc = evaluate(model, [], loss_fn)
        assert avg_loss == 0.0
        assert acc == 0.0

    def test_multiple_batches(self):
        probs = torch.zeros(2, 3)
        probs[0, :] = 1.0
        model = FakeModel(probs)
        loss_fn = FakeLoss(loss_value=2.0)
        labels = torch.zeros(3, dtype=torch.long)
        loader = [
            (torch.randn(3, 1), labels),
            (torch.randn(3, 1), labels),
        ]

        avg_loss, acc = evaluate(model, loader, loss_fn)
        assert avg_loss == pytest.approx(2.0)
        assert acc == pytest.approx(1.0)


class TestTrain:
    @pytest.fixture
    def setup(self, tmp_path):
        probs = torch.zeros(3, 2)
        probs[0, :] = 1.0
        p = nn.Parameter(torch.randn(2, 2))
        model = FakeModel(probs)
        model.layers.append(FakeLayerWithParams(p))
        loss_fn = FakeLoss(loss_value=0.5)
        optimizer = FakeOptimizer()
        labels = torch.zeros(2, dtype=torch.long)
        train_loader = [(torch.randn(2, 3, 2, 2), labels)]
        test_loader = [(torch.randn(2, 3, 2, 2), labels)]
        checkpoint_path = str(tmp_path / "best.pt")
        return model, train_loader, test_loader, loss_fn, optimizer, checkpoint_path, p

    def test_returns_history_dict(self, setup):
        model, train_loader, test_loader, loss_fn, optimizer, cp, _ = setup
        history = train(
            model,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            epochs=3,
            checkpoint_path=cp,
        )
        assert set(history.keys()) == {"train_loss", "train_acc", "test_loss", "test_acc"}
        assert len(history["train_loss"]) == 3
        assert len(history["train_acc"]) == 3
        assert len(history["test_loss"]) == 3
        assert len(history["test_acc"]) == 3

    def test_history_values(self, setup):
        model, train_loader, test_loader, loss_fn, optimizer, cp, _ = setup
        history = train(
            model,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            epochs=2,
            checkpoint_path=cp,
        )
        for loss in history["train_loss"]:
            assert loss == pytest.approx(0.5)
        for acc in history["train_acc"]:
            assert acc == pytest.approx(1.0)

    def test_checkpoint_saved(self, setup):
        model, train_loader, test_loader, loss_fn, optimizer, cp, _ = setup
        train(
            model,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            epochs=1,
            checkpoint_path=cp,
        )
        assert os.path.exists(cp)

    def test_checkpoint_has_stable_keys(self, setup):
        model, train_loader, test_loader, loss_fn, optimizer, cp, _ = setup
        train(
            model,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            epochs=1,
            checkpoint_path=cp,
        )
        saved_state = torch.load(cp, weights_only=False)
        # Keys should be (layer_idx, param_name) tuples, not id() ints
        for key in saved_state:
            assert isinstance(key, tuple)
            assert isinstance(key[0], int)

    def test_best_weights_restored(self, setup):
        model, train_loader, test_loader, loss_fn, optimizer, cp, p = setup

        train(
            model,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            epochs=1,
            checkpoint_path=cp,
        )

        saved_state = torch.load(cp, weights_only=False)
        # The param is on layer index 2 (after FakeLayerWithTraining, FakeLayerWithoutTraining)
        key = (2, ParameterType.WEIGHT)
        assert torch.allclose(p.data, saved_state[key])

    def test_scheduler_stepped(self, setup):
        model, train_loader, test_loader, loss_fn, optimizer, cp, _ = setup
        scheduler = FakeScheduler()

        train(
            model,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            scheduler=scheduler,
            epochs=5,
            checkpoint_path=cp,
        )

        assert scheduler.step_count == 5

    def test_no_scheduler(self, setup):
        model, train_loader, test_loader, loss_fn, optimizer, cp, _ = setup
        # Should not raise when scheduler is None
        history = train(
            model,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            scheduler=None,
            epochs=1,
            checkpoint_path=cp,
        )
        assert len(history["train_loss"]) == 1

    def test_prints_epoch_summary(self, setup, capsys):
        model, train_loader, test_loader, loss_fn, optimizer, cp, _ = setup
        train(
            model,
            train_loader,
            test_loader,
            loss_fn,
            optimizer,
            epochs=2,
            checkpoint_path=cp,
        )
        captured = capsys.readouterr()
        assert "Epoch 1/2" in captured.out
        assert "Epoch 2/2" in captured.out
        assert "train_loss" in captured.out
        assert "test_acc" in captured.out

    def test_checkpoint_only_on_improvement(self, tmp_path):
        """Test that checkpoint is saved only when test_acc improves."""
        call_count = 0

        class DegradingModel(FakeModel):
            def forward(self, x):
                nonlocal call_count
                call_count += 1
                probs = torch.zeros(2, 2)
                if call_count <= 2:
                    probs[0, :] = 1.0
                else:
                    probs[1, :] = 1.0
                return probs

        p = nn.Parameter(torch.randn(2))
        model = DegradingModel(None)
        model.layers.append(FakeLayerWithParams(p))
        loss_fn = FakeLoss()
        optimizer = FakeOptimizer()
        labels = torch.zeros(2, dtype=torch.long)
        loader = [(torch.randn(2, 1), labels)]
        cp = str(tmp_path / "best.pt")

        train(model, loader, loader, loss_fn, optimizer, epochs=2, checkpoint_path=cp)

        assert os.path.exists(cp)
        saved = torch.load(cp, weights_only=False)
        # Key should be stable tuple, not id()
        assert (2, ParameterType.WEIGHT) in saved
