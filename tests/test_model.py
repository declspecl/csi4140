import torch
import torch.nn as nn

from src.model import CIFAR10CNN


class TestCIFAR10CNN:
    def test_output_shape(self):
        model = CIFAR10CNN()
        out = model(torch.randn(4, 3, 32, 32))
        assert out.shape == (4, 10)

    def test_output_shape_single_sample(self):
        model = CIFAR10CNN()
        out = model(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, 10)

    def test_outputs_logits_not_probs(self):
        # Logits — no softmax applied, values can be negative or > 1
        model = CIFAR10CNN()
        out = model(torch.randn(2, 3, 32, 32))
        assert not torch.allclose(out.sum(dim=1), torch.ones(2))

    def test_has_three_conv_layers(self):
        model = CIFAR10CNN()
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        assert len(convs) == 3

    def test_conv_channels(self):
        model = CIFAR10CNN()
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        assert (convs[0].in_channels, convs[0].out_channels) == (3, 64)
        assert (convs[1].in_channels, convs[1].out_channels) == (64, 128)
        assert (convs[2].in_channels, convs[2].out_channels) == (128, 256)

    def test_has_two_linear_layers(self):
        model = CIFAR10CNN()
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 2

    def test_linear_dims(self):
        model = CIFAR10CNN()
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        assert (linears[0].in_features, linears[0].out_features) == (256 * 8 * 8, 512)
        assert (linears[1].in_features, linears[1].out_features) == (512, 10)

    def test_backward_runs(self):
        model = CIFAR10CNN()
        x = torch.randn(2, 3, 32, 32)
        loss = model(x).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
