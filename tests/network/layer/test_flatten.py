import pytest
import torch

from src.network.layer.flatten import Flatten


@pytest.fixture
def flatten():
    return Flatten()


@pytest.fixture
def input():
    torch.manual_seed(0)
    return torch.randn(4, 3, 8, 8)


class TestForward:
    def test_output_shape(self, flatten, input):
        out = flatten.forward(input)
        assert out.shape == (3 * 8 * 8, 4)

    def test_batch_last(self, flatten, input):
        out = flatten.forward(input)
        assert out.shape[1] == input.shape[0]

    def test_features_first(self, flatten, input):
        out = flatten.forward(input)
        assert out.shape[0] == 3 * 8 * 8

    def test_values_preserved(self, flatten):
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
        out = flatten.forward(x)
        assert torch.allclose(out, x.reshape(2, -1).t())


class TestBackward:
    def test_grad_shape_restored(self, flatten, input):
        flatten.forward(input)
        grad_out = torch.randn(3 * 8 * 8, 4)
        grad_in = flatten.backward(grad_out)
        assert grad_in.shape == input.shape

    def test_backward_before_forward_raises(self, flatten):
        with pytest.raises(ValueError, match="forward"):
            flatten.backward(torch.randn(12, 4))

    def test_roundtrip(self, flatten, input):
        out = flatten.forward(input)
        grad_in = flatten.backward(out)
        assert grad_in.shape == input.shape
        assert torch.allclose(grad_in, input)
