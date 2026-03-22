import torch

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.cifar10_cnn import build_cifar10_cnn
from src.network.loss.cross_entropy import CrossEntropy
from src.network.optimizer.adam import Adam
from src.network.scheduler.cosine import CosineDecay
from src.train import train
from src.utils.visualization import plot_all


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    epochs = 50
    learning_rate = 0.001

    train_loader, test_loader = get_cifar10_dataloaders(batch_size=128)
    model = build_cifar10_cnn()
    model.to(device)
    loss_fn = CrossEntropy()
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)
    scheduler = CosineDecay(optimizer, epochs=epochs)

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=device,
        checkpoint_path="best_model.pt",
    )

    print(f"\nBest test accuracy: {max(history['test_acc']):.4f}")
    plot_all(history)


if __name__ == "__main__":
    main()
