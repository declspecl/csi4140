import os
import matplotlib.pyplot as plt


def plot_training_accuracy(history: dict, save_path: str = "results/plots/train_accuracy.png") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(history["train_acc"]) + 1), history["train_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy over Epochs")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_test_accuracy(history: dict, save_path: str = "results/plots/test_accuracy.png") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.plot(range(1, len(history["test_acc"]) + 1), history["test_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy over Epochs")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_cost(history: dict, save_path: str = "results/plots/cost.png") -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure()
    plt.plot(history["iteration_loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost over Iterations")
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_all(history: dict, save_dir: str = "results/plots") -> None:
    plot_training_accuracy(history, os.path.join(save_dir, "train_accuracy.png"))
    plot_test_accuracy(history, os.path.join(save_dir, "test_accuracy.png"))
    plot_cost(history, os.path.join(save_dir, "cost.png"))
