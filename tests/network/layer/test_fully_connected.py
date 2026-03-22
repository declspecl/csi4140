import pytest
import torch

from src.network.activation.identity import Identity
from src.network.activation.relu import ReLU
from src.network.layer.fully_connected import FullyConnected


@pytest.fixture
def fc():
    torch.manual_seed(0)
    return FullyConnected(in_features=4, out_features=3, activation=Identity())


@pytest.fixture
def input():
    torch.manual_seed(1)
    return torch.randn(4, 8)  # (in_features, batch_size)


class TestForward:
    def test_output_shape(self, fc, input):
        out = fc.forward(input)
        assert out.shape == (3, 8)

    def test_output_shape_single_sample(self, fc):
        x = torch.randn(4, 1)
        out = fc.forward(x)
        assert out.shape == (3, 1)

    def test_bias_only(self):
        torch.manual_seed(0)
        fc = FullyConnected(in_features=2, out_features=3, activation=Identity())
        fc.weight.data.zero_()
        fc.bias.data = torch.tensor([[1.0], [2.0], [3.0]])
        x = torch.randn(2, 5)
        out = fc.forward(x)
        expected = torch.tensor([[1.0], [2.0], [3.0]]).expand(3, 5)
        assert torch.allclose(out, expected)

    def test_activation_applied(self):
        torch.manual_seed(0)
        fc_identity = FullyConnected(in_features=4, out_features=3, activation=Identity())
        fc_relu = FullyConnected(in_features=4, out_features=3, activation=ReLU())
        fc_relu.weight.data = fc_identity.weight.data.clone()
        fc_relu.bias.data = fc_identity.bias.data.clone()

        x = torch.randn(4, 8)
        out_identity = fc_identity.forward(x)
        out_relu = fc_relu.forward(x)
        assert torch.allclose(out_relu, out_identity.clamp(min=0))

    def test_input_cached(self, fc, input):
        fc.forward(input)
        assert fc._input_cache is input


class TestBackward:
    def test_grad_input_shape(self, fc, input):
        fc.forward(input)
        grad_out = torch.randn(3, 8)
        grad_in = fc.backward(grad_out)
        assert grad_in.shape == input.shape

    def test_grad_weight_shape(self, fc, input):
        fc.forward(input)
        fc.backward(torch.randn(3, 8))
        assert fc.weight.grad.shape == fc.weight.shape

    def test_grad_bias_shape(self, fc, input):
        fc.forward(input)
        fc.backward(torch.randn(3, 8))
        assert fc.bias.grad.shape == fc.bias.shape

    def test_backward_before_forward_raises(self, fc):
        with pytest.raises(ValueError, match="forward"):
            fc.backward(torch.randn(3, 8))

    def test_grad_weight_numerical(self):
        """Verify grad_weight via finite differences."""
        torch.manual_seed(42)
        fc = FullyConnected(in_features=3, out_features=2, activation=Identity())
        x = torch.randn(3, 4)
        eps = 1e-3

        fc.forward(x)
        grad_out = torch.ones(2, 4)
        fc.backward(grad_out)
        assert fc.weight.grad is not None
        grad_w_analytical = fc.weight.grad.clone()

        grad_w_numerical = torch.zeros_like(fc.weight)
        for o in range(fc.weight.shape[0]):
            for i in range(fc.weight.shape[1]):
                w_plus = fc.weight.data.clone()
                w_plus[o, i] += eps
                w_minus = fc.weight.data.clone()
                w_minus[o, i] -= eps

                fc.weight.data = w_plus
                f_plus = fc.forward(x).sum()
                fc.weight.data = w_minus
                f_minus = fc.forward(x).sum()
                grad_w_numerical[o, i] = (f_plus - f_minus) / (2 * eps)

        assert torch.allclose(grad_w_analytical, grad_w_numerical, atol=1e-3)

    def test_grad_input_numerical(self):
        """Verify grad_input via finite differences."""
        torch.manual_seed(42)
        fc = FullyConnected(in_features=3, out_features=2, activation=ReLU())
        x = torch.randn(3, 4)
        eps = 1e-3

        fc.forward(x)
        grad_out = torch.ones(2, 4)
        grad_in_analytical = fc.backward(grad_out)

        grad_in_numerical = torch.zeros_like(x)
        for i in range(x.shape[0]):
            for n in range(x.shape[1]):
                x_plus = x.clone()
                x_plus[i, n] += eps
                x_minus = x.clone()
                x_minus[i, n] -= eps
                f_plus = fc.forward(x_plus).sum()
                f_minus = fc.forward(x_minus).sum()
                grad_in_numerical[i, n] = (f_plus - f_minus) / (2 * eps)

        assert torch.allclose(grad_in_analytical, grad_in_numerical, atol=1e-3)

    def test_grad_bias_value(self, fc, input):
        fc.forward(input)
        grad_out = torch.randn(3, 8)
        fc.backward(grad_out)
        expected = grad_out.sum(dim=1, keepdim=True)
        assert fc.bias.grad is not None
        assert torch.allclose(fc.bias.grad, expected)
