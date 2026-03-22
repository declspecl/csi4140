import torch

from src.models.cifar10_cnn import build_cifar10_cnn
from src.network.layer.convolutional import Convolutional
from src.network.layer.flatten import Flatten
from src.network.layer.fully_connected import FullyConnected
from src.network.layer.activation_layer import ActivationLayer


class TestBuildCifar10Cnn:
    def test_output_shape(self):
        model = build_cifar10_cnn()
        x = torch.randn(4, 3, 32, 32)
        out = model.forward(x)
        assert out.shape == (10, 4)

    def test_output_shape_single_sample(self):
        model = build_cifar10_cnn()
        x = torch.randn(1, 3, 32, 32)
        out = model.forward(x)
        assert out.shape == (10, 1)

    def test_output_sums_to_one(self):
        model = build_cifar10_cnn()
        x = torch.randn(2, 3, 32, 32)
        out = model.forward(x)
        sums = out.sum(dim=0)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_output_non_negative(self):
        model = build_cifar10_cnn()
        x = torch.randn(2, 3, 32, 32)
        out = model.forward(x)
        assert (out >= 0).all()

    def test_has_convolutional_layer(self):
        model = build_cifar10_cnn()
        conv_layers = [l for l in model.layers if isinstance(l, Convolutional)]
        assert len(conv_layers) >= 1

    def test_has_fully_connected_layer(self):
        model = build_cifar10_cnn()
        fc_layers = [l for l in model.layers if isinstance(l, FullyConnected)]
        assert len(fc_layers) >= 1

    def test_has_relu_activation(self):
        model = build_cifar10_cnn()
        relu_layers = [l for l in model.layers if isinstance(l, ActivationLayer)]
        assert len(relu_layers) >= 1

    def test_has_flatten_layer(self):
        model = build_cifar10_cnn()
        flatten_layers = [l for l in model.layers if isinstance(l, Flatten)]
        assert len(flatten_layers) == 1

    def test_backward_runs(self):
        model = build_cifar10_cnn()
        x = torch.randn(2, 3, 32, 32)
        out = model.forward(x)
        grad = torch.randn_like(out)
        grad_in = model.backward(grad)
        assert grad_in.shape == x.shape
