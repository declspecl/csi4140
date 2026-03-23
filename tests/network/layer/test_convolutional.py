import pytest
import torch

from src.network.layer.convolutional import Convolutional


@pytest.fixture
def conv():
    torch.manual_seed(0)
    return Convolutional(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=0)


@pytest.fixture
def input():
    torch.manual_seed(1)
    return torch.randn(2, 3, 8, 8)


class TestForward:
    def test_output_shape_no_padding(self, conv, input):
        out = conv.forward(input)
        assert out.shape == (2, 4, 6, 6)

    def test_output_shape_with_padding(self, input):
        torch.manual_seed(0)
        conv = Convolutional(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=1)
        out = conv.forward(input)
        assert out.shape == (2, 4, 8, 8)

    def test_output_shape_with_stride(self, input):
        torch.manual_seed(0)
        conv = Convolutional(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)
        out = conv.forward(input)
        assert out.shape == (2, 4, 3, 3)

    def test_requires_4d_input(self, conv):
        with pytest.raises(ValueError, match="4D"):
            conv.forward(torch.randn(3, 8, 8))

    def test_bias_effect(self):
        torch.manual_seed(0)
        conv = Convolutional(in_channels=1, out_channels=1, kernel_size=1)
        x = torch.randn(1, 1, 4, 4)
        conv.weight.data.zero_()
        conv.bias.data.fill_(5.0)
        out = conv.forward(x)
        assert torch.allclose(out, torch.full_like(out, 5.0))


class TestBackward:
    def test_grad_input_shape(self, conv, input):
        conv.forward(input)
        grad_out = torch.randn(2, 4, 6, 6)
        grad_in = conv.backward(grad_out)
        assert grad_in.shape == input.shape

    def test_grad_weight_shape(self, conv, input):
        conv.forward(input)
        conv.backward(torch.randn(2, 4, 6, 6))
        assert conv.weight.grad.shape == conv.weight.shape

    def test_grad_bias_shape(self, conv, input):
        conv.forward(input)
        conv.backward(torch.randn(2, 4, 6, 6))
        assert conv.bias.grad.shape == conv.bias.shape

    def test_grad_bias_value(self, conv, input):
        conv.forward(input)
        grad_out = torch.randn(2, 4, 6, 6)
        conv.backward(grad_out)
        expected = grad_out.sum(dim=(0, 2, 3))
        assert torch.allclose(conv.bias.grad, expected)

    def test_backward_before_forward_raises(self, conv):
        with pytest.raises(ValueError, match="forward"):
            conv.backward(torch.randn(2, 4, 6, 6))

    def test_grad_input_numerical(self):
        torch.manual_seed(42)
        conv = Convolutional(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0)
        x = torch.randn(1, 2, 5, 5)
        eps = 1e-3

        out = conv.forward(x)
        grad_out = torch.ones_like(out)
        grad_in_analytical = conv.backward(grad_out)

        grad_in_numerical = torch.zeros_like(x)
        for n in range(x.shape[0]):
            for c in range(x.shape[1]):
                for h in range(x.shape[2]):
                    for w in range(x.shape[3]):
                        x_plus = x.clone()
                        x_plus[n, c, h, w] += eps
                        x_minus = x.clone()
                        x_minus[n, c, h, w] -= eps
                        f_plus = conv.forward(x_plus).sum()
                        f_minus = conv.forward(x_minus).sum()
                        grad_in_numerical[n, c, h, w] = (f_plus - f_minus) / (2 * eps)

        assert torch.allclose(grad_in_analytical, grad_in_numerical, atol=1e-3)

    def test_grad_weight_numerical(self):
        torch.manual_seed(42)
        conv = Convolutional(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0)
        x = torch.randn(1, 2, 5, 5)
        eps = 1e-3

        conv.forward(x)
        grad_out = torch.ones(1, 2, 3, 3)
        conv.backward(grad_out)
        assert conv.weight.grad is not None
        grad_w_analytical = conv.weight.grad.clone()

        grad_w_numerical = torch.zeros_like(conv.weight)
        for o in range(conv.weight.shape[0]):
            for c in range(conv.weight.shape[1]):
                for kh in range(conv.weight.shape[2]):
                    for kw in range(conv.weight.shape[3]):
                        w_plus = conv.weight.data.clone()
                        w_plus[o, c, kh, kw] += eps
                        w_minus = conv.weight.data.clone()
                        w_minus[o, c, kh, kw] -= eps

                        conv.weight.data = w_plus
                        f_plus = conv.forward(x).sum()
                        conv.weight.data = w_minus
                        f_minus = conv.forward(x).sum()
                        grad_w_numerical[o, c, kh, kw] = (f_plus - f_minus) / (2 * eps)

        assert torch.allclose(grad_w_analytical, grad_w_numerical, atol=1e-3)
