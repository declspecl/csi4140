from src.network.neural_network import NeuralNetwork
from src.network.layer.convolutional import Convolutional
from src.network.layer.flatten import Flatten
from src.network.layer.fully_connected import FullyConnected
from src.network.layer.activation_layer import ActivationLayer
from src.network.activation.relu import ReLU
from src.network.activation.softmax import Softmax


def build_cifar10_cnn() -> NeuralNetwork:
    # Strided convolutions for downsampling instead of MaxPool (https://arxiv.org/abs/1412.6806)
    layers = [
        # (N, 3, 32, 32) -> (N, 64, 32, 32)
        Convolutional(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        ActivationLayer(ReLU()),
        # (N, 64, 32, 32) -> (N, 128, 16, 16) — stride=2 downsamples spatially
        Convolutional(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
        ActivationLayer(ReLU()),
        # (N, 128, 16, 16) -> (N, 256, 8, 8) — stride=2 downsamples spatially
        Convolutional(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
        ActivationLayer(ReLU()),
        # (N, 256, 8, 8) -> (16384, N)
        Flatten(),
        # (16384, N) -> (512, N)
        FullyConnected(256 * 8 * 8, 512, ReLU()),
        # (512, N) -> (10, N)
        FullyConnected(512, 10, Softmax()),
    ]
    return NeuralNetwork(layers)
