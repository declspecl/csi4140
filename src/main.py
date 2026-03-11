import torch
from src.network.activation.identity import Identity
from src.network.activation.sigmoid import Sigmoid
from src.network.activation.softmax import Softmax
from src.network.layer.fully_connected import FullyConnected
from src.network.loss.cross_entropy import CrossEntropy
from src.network.neural_network import NeuralNetwork


def main():
    torch.manual_seed(32)

    X = torch.rand(5, 100, dtype=torch.float32)
    y = torch.randint(0, 3, (100,), dtype=torch.long)

    n_x, m = X.shape
    n_h1 = 5
    n_h2 = 4
    n_output = 3
    learning_rate = 0.01
    iterations = 1000

    network = NeuralNetwork([
        FullyConnected(n_x, n_h1, Sigmoid()),
        FullyConnected(n_h1, n_h2, Sigmoid()),
        FullyConnected(n_h2, n_output, Softmax()),
    ])

    loss_fn = CrossEntropy()
    cost_values = []

    for iteration in range(iterations):
        output = network.forward(X)
        loss = loss_fn.calculate_loss(output, y)

        cost_values.append(loss.item())

        grad_output = loss_fn.calculate_gradient(output, y)
        network.backward(grad_output)

        params = [param for layer in network.layers for param in layer.parameters().values()]
        for param in params:
            if param.grad is not None:
                param.data -= learning_rate * param.grad

    print(f"Final cost value: {cost_values[-1]:.6f}")


if __name__ == "__main__":
    main()
