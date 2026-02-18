import torch
from src.network.neural_network import NeuralNetwork
from src.network.layer.fully_connected import FullyConnected
from src.network.layer.activation.identity import Identity


def main():
    model = NeuralNetwork([
        FullyConnected(3072, 512, activation=Identity()),
        FullyConnected(512, 256, activation=Identity()),
        FullyConnected(256, 10, activation=Identity()),
    ])

    inputs = torch.randn(4, 3, 32, 32)
    output = model.forward(inputs)

    print(output)


if __name__ == "__main__":
    main()
