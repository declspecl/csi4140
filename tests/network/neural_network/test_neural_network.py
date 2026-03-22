import pytest
import torch

from src.network.activation.identity import Identity
from src.network.activation.softmax import Softmax
from src.network.layer.dropout import Dropout
from src.network.layer.fully_connected import FullyConnected
from src.network.neural_network.neural_network import NeuralNetwork


@pytest.fixture
def simple_network():
    torch.manual_seed(0)
    return NeuralNetwork(
        [
            FullyConnected(4, 3, Identity()),
            FullyConnected(3, 2, Softmax()),
        ]
    )


@pytest.fixture
def network_with_dropout():
    torch.manual_seed(0)
    return NeuralNetwork(
        [
            FullyConnected(4, 3, Identity()),
            Dropout(p=0.5),
            FullyConnected(3, 2, Softmax()),
        ]
    )


class TestInit:
    def test_layers_stored(self, simple_network):
        assert len(simple_network.layers) == 2

    def test_training_default_true(self, simple_network):
        assert simple_network.training is True


class TestForward:
    def test_output_shape(self, simple_network):
        x = torch.randn(4, 5)
        out = simple_network.forward(x)
        assert out.shape == (2, 5)

    def test_output_sums_to_one(self, simple_network):
        x = torch.randn(4, 5)
        out = simple_network.forward(x)
        assert torch.allclose(out.sum(dim=0), torch.ones(5), atol=1e-5)

    def test_passes_through_all_layers(self, simple_network):
        x = torch.randn(4, 1)
        out = simple_network.forward(x)
        # Manually compute: FC1 -> FC2 with softmax
        z1 = simple_network.layers[0].weight @ x + simple_network.layers[0].bias
        z2 = simple_network.layers[1].weight @ z1 + simple_network.layers[1].bias
        expected = torch.softmax(z2, dim=0)
        assert torch.allclose(out, expected, atol=1e-5)


class TestBackward:
    def test_returns_gradient(self, simple_network):
        x = torch.randn(4, 5)
        out = simple_network.forward(x)
        grad = torch.randn_like(out)
        grad_in = simple_network.backward(grad)
        assert grad_in.shape == x.shape

    def test_gradients_populated(self, simple_network):
        x = torch.randn(4, 5)
        out = simple_network.forward(x)
        simple_network.backward(torch.randn_like(out))
        for layer in simple_network.layers:
            for param in layer.parameters().values():
                assert param.grad is not None

    def test_gradients_finite(self, simple_network):
        x = torch.randn(4, 5)
        out = simple_network.forward(x)
        simple_network.backward(torch.randn_like(out))
        for layer in simple_network.layers:
            for param in layer.parameters().values():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all()


class TestTrain:
    def test_sets_training_true(self, network_with_dropout):
        network_with_dropout.eval()
        network_with_dropout.train()
        assert network_with_dropout.training is True

    def test_propagates_to_dropout(self, network_with_dropout):
        network_with_dropout.eval()
        network_with_dropout.train()
        dropout = network_with_dropout.layers[1]
        assert dropout.training is True

    def test_does_not_affect_non_dropout_layers(self, network_with_dropout):
        network_with_dropout.train()
        fc = network_with_dropout.layers[0]
        assert not hasattr(fc, "training") or fc.training is True


class TestEval:
    def test_sets_training_false(self, network_with_dropout):
        network_with_dropout.eval()
        assert network_with_dropout.training is False

    def test_propagates_to_dropout(self, network_with_dropout):
        network_with_dropout.eval()
        dropout = network_with_dropout.layers[1]
        assert dropout.training is False

    def test_dropout_inactive_in_eval(self, network_with_dropout):
        torch.manual_seed(42)
        x = torch.randn(4, 10)
        network_with_dropout.eval()
        out1 = network_with_dropout.forward(x)
        out2 = network_with_dropout.forward(x)
        assert torch.allclose(out1, out2)

    def test_multiple_dropout_layers(self):
        torch.manual_seed(0)
        net = NeuralNetwork(
            [
                FullyConnected(4, 3, Identity()),
                Dropout(p=0.3),
                FullyConnected(3, 3, Identity()),
                Dropout(p=0.5),
                FullyConnected(3, 2, Softmax()),
            ]
        )
        net.eval()
        assert net.layers[1].training is False
        assert net.layers[3].training is False
        net.train()
        assert net.layers[1].training is True
        assert net.layers[3].training is True


class TestParameters:
    def test_returns_layers(self, simple_network):
        params = simple_network.parameters()
        assert params is simple_network.layers

    def test_return_type_is_list_like(self, simple_network):
        params = simple_network.parameters()
        assert hasattr(params, "__iter__")
        assert hasattr(params, "__len__")
