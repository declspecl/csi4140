import pytest
import torch

from src.network.activation.softmax import Softmax
from src.network.layer.convolutional import Convolutional
from src.network.layer.flatten import Flatten
from src.network.layer.fully_connected import FullyConnected
from src.network.neural_network import NeuralNetwork


@pytest.fixture
def network():
    torch.manual_seed(0)
    return NeuralNetwork(
        [
            Convolutional(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1),
            Flatten(),
            FullyConnected(4 * 8 * 8, 10, Softmax()),
        ]
    )


@pytest.fixture
def input():
    torch.manual_seed(1)
    return torch.randn(2, 3, 8, 8)


class TestConvFlattenFC:
    def test_forward_output_shape(self, network, input):
        out = network.forward(input)
        assert out.shape == (10, 2)

    def test_forward_output_sums_to_one(self, network, input):
        out = network.forward(input)
        assert torch.allclose(out.sum(dim=0), torch.ones(2), atol=1e-5)

    def test_backward_grad_input_shape(self, network, input):
        out = network.forward(input)
        grad = torch.randn_like(out)
        grad_in = network.backward(grad)
        assert grad_in.shape == input.shape

    def test_conv_weight_grad_populated(self, network, input):
        out = network.forward(input)
        network.backward(torch.randn_like(out))
        conv = network.layers[0]
        assert conv.weight.grad is not None
        assert conv.bias.grad is not None

    def test_fc_weight_grad_populated(self, network, input):
        out = network.forward(input)
        network.backward(torch.randn_like(out))
        fc = network.layers[2]
        assert fc.weight.grad is not None
        assert fc.bias.grad is not None

    def test_gradients_are_finite(self, network, input):
        out = network.forward(input)
        network.backward(torch.randn_like(out))
        for layer in network.layers:
            for param in layer.parameters().values():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all()
