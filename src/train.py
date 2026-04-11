import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import TypedDict


class TrainingHistory(TypedDict):
    train_loss: list[float]
    train_acc: list[float]
    test_loss: list[float]
    test_acc: list[float]
    iteration_loss: list[float]


def train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: Optimizer,
    device: str,
    iteration_losses: list[float] | None = None,
    verbose: bool = True,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits: Tensor = model(images)
        loss: Tensor = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

        if iteration_losses is not None:
            iteration_losses.append(loss.item())

        if verbose and i % 50 == 0:
            print(f"    Batch {i}/{len(loader)} - loss: {loss.item():.4f}", flush=True)

    return (total_loss / len(loader) if loader else 0.0), (correct / total if total else 0.0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits: Tensor = model(images)
        total_loss += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return (total_loss / len(loader) if loader else 0.0), (correct / total if total else 0.0)


def train(
    model: nn.Module,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    test_loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    epochs: int = 50,
    device: str = "cpu",
    checkpoint_path: str = "best_model.pt",
    verbose: bool = True,
) -> TrainingHistory:
    history: TrainingHistory = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "iteration_loss": [],
    }
    best_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, history["iteration_loss"], verbose
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} - "
            f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), checkpoint_path)

        if scheduler is not None:
            scheduler.step()

    return history
