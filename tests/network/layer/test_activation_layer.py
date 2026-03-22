import pytest
import torch

from src.network.layer.activation_layer import ActivationLayer
from src.network.activation.relu import ReLU
from src.network.activation.softmax import Softmax


@pytest.fixture
def relu_layer():
    return ActivationLayer(ReLU())


@pytest.fixture
def input():
    torch.manual_seed(0)
    return torch.randn(2, 3, 4, 4)


class TestForward:
    def test_output_shape(self, relu_layer, input):
        out = relu_layer.forward(input)
        assert out.shape == input.shape

    def test_relu_zeros_negatives(self, relu_layer):
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        out = relu_layer.forward(x)
        expected = torch.tensor([0.0, 0.0, 1.0, 2.0])
        assert torch.allclose(out, expected)

    def test_delegates_to_activation(self):
        layer = ActivationLayer(Softmax())
        x = torch.tensor([[1.0], [2.0], [3.0]])
        out = layer.forward(x)
        assert torch.allclose(out.sum(dim=0), torch.tensor([1.0]))


class TestBackward:
    def test_grad_shape(self, relu_layer, input):
        relu_layer.forward(input)
        grad_out = torch.randn_like(input)
        grad_in = relu_layer.backward(grad_out)
        assert grad_in.shape == input.shape

    def test_relu_grad_masks_negatives(self, relu_layer):
        x = torch.tensor([-1.0, 0.0, 1.0, 2.0])
        relu_layer.forward(x)
        grad_out = torch.ones(4)
        grad_in = relu_layer.backward(grad_out)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0])
        assert torch.allclose(grad_in, expected)


class TestParameters:
    def test_returns_empty_dict(self, relu_layer):
        assert relu_layer.parameters() == {}
