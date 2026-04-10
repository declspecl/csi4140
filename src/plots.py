import os
import matplotlib.pyplot as plt

from src.train import TrainingHistory


def _save(fig_path: str) -> None:
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


def plot_all(history: TrainingHistory, save_dir: str = "results/plots") -> None:
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["test_acc"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    _save(os.path.join(save_dir, "accuracy.png"))

    plt.figure()
    plt.plot(history["iteration_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss over Iterations")
    plt.grid(True)
    _save(os.path.join(save_dir, "loss.png"))
