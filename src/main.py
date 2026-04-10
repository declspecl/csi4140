import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.model import CIFAR10CNN
from src.train import train
from src.plots import plot_all


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=128)
    model = CIFAR10CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.8, 0.99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=50,
        device=device,
        checkpoint_path="best_model.pt",
    )

    print(f"\nBest test accuracy: {max(history['test_acc']):.4f}")
    plot_all(history)


if __name__ == "__main__":
    main()
