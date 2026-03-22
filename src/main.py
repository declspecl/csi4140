import torch
from src.network.activation.sigmoid import Sigmoid
from src.network.activation.softmax import Softmax
from src.network.layer.dropout import Dropout
from src.network.layer.fully_connected import FullyConnected
from src.network.loss.cross_entropy import CrossEntropy
from src.network.neural_network import NeuralNetwork
from src.network.optimizer.sgd import SGD
from src.network.regularizer.l2 import L2


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
        Dropout(p=0.5),
        FullyConnected(n_h1, n_h2, Sigmoid()),
        Dropout(p=0.5),
        FullyConnected(n_h2, n_output, Softmax()),
    ])

    loss_fn = CrossEntropy()
    optimizer = SGD(network.layers, learning_rate=learning_rate, regularizer=L2(lambda_=0.01))
    cost_values = []

    for iteration in range(iterations):
        output = network.forward(X)
        loss = loss_fn.calculate_loss(output, y)

        cost_values.append(loss.item())

        grad_output = loss_fn.calculate_gradient(output, y)
        network.backward(grad_output)

        optimizer.step()

    print(f"Final cost value: {cost_values[-1]:.6f}")


if __name__ == "__main__":
    main()
