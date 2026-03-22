import json
import os
import torch
import matplotlib.pyplot as plt
import argparse

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.cifar10_cnn import build_cifar10_cnn
from src.network.layer.convolutional import Convolutional
from src.network.layer.activation_layer import ActivationLayer
from src.network.layer.flatten import Flatten
from src.network.layer.fully_connected import FullyConnected
from src.network.layer.dropout import Dropout
from src.network.activation.relu import ReLU
from src.network.activation.softmax import Softmax
from src.network.neural_network import NeuralNetwork
from src.network.loss.cross_entropy import CrossEntropy
from src.network.optimizer.adam import Adam
from src.network.optimizer.momentum import Momentum
from src.network.scheduler.cosine import CosineDecay
from src.network.scheduler.step import StepDecay
from src.network.regularizer.l2 import L2
from src.train import train
from src.network.activation.sigmoid import Sigmoid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ABLATION_EPOCHS = int(os.getenv("ABLATION_EPOCHS", 15))
BATCH_SIZE = 128
RESULTS_DIR = "results/ablation"

def run_experiment(name, model, optimizer, scheduler=None, epochs=ABLATION_EPOCHS):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")

    model.to(DEVICE)
    loss_fn = CrossEntropy()
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=BATCH_SIZE, num_workers=2)

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        device=DEVICE,
        checkpoint_path=os.path.join(RESULTS_DIR, f"{name.replace(' ', '_')}.pt"),
    )
    return history

def build_model_with_dropout(p=0.5):
    return NeuralNetwork(
        [
            Convolutional(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            ActivationLayer(ReLU()),
            Dropout(p=p),
            Convolutional(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            ActivationLayer(ReLU()),
            Dropout(p=p),
            Convolutional(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            ActivationLayer(ReLU()),
            Dropout(p=p),
            Flatten(),
            FullyConnected(128 * 8 * 8, 512, ReLU()),
            FullyConnected(512, 10, Softmax()),
        ]
    )

def plot_comparison(results, title, ylabel, key, save_path):
    plt.figure(figsize=(8, 5))
    for label, history in results.items():
        plt.plot(range(1, len(history[key]) + 1), history[key], label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

def save_results(experiment_name, results):
    path = os.path.join(RESULTS_DIR, f"{experiment_name}.json")
    serializable = {}
    for label, history in results.items():
        serializable[label] = {k: v for k, v in history.items() if k != "iteration_loss"}
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved: {path}")

def experiment_lr_decay():
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 1: Learning Rate Decay Algorithms")
    print("=" * 60)

    results = {}

    model = build_cifar10_cnn()
    opt = Adam(model.parameters(), learning_rate=0.001)
    sched = CosineDecay(opt, epochs=ABLATION_EPOCHS)
    results["Cosine Decay"] = run_experiment("LR_cosine", model, opt, sched)

    model = build_cifar10_cnn()
    opt = Adam(model.parameters(), learning_rate=0.001)
    sched = StepDecay(opt, step_size=5, decay_factor=0.5)
    results["Step Decay (×0.5 every 5)"] = run_experiment("LR_step", model, opt, sched)

    model = build_cifar10_cnn()
    opt = Adam(model.parameters(), learning_rate=0.001)
    results["No Decay"] = run_experiment("LR_none", model, opt, None)

    plot_comparison(
        results,
        "Effect of LR Decay on Test Accuracy",
        "Test Accuracy",
        "test_acc",
        os.path.join(RESULTS_DIR, "exp1_lr_decay.png"),
    )
    plot_comparison(
        results,
        "Effect of LR Decay on Training Loss",
        "Training Loss",
        "train_loss",
        os.path.join(RESULTS_DIR, "exp1_lr_decay_loss.png"),
    )
    save_results("exp1_lr_decay", results)
    return results

def experiment_regularization():
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 2: Regularization Methods")
    print("=" * 60)

    results = {}

    model = build_cifar10_cnn()
    opt = Adam(model.parameters(), learning_rate=0.001)
    results["None"] = run_experiment("Reg_none", model, opt)

    model = build_cifar10_cnn()
    opt = Adam(model.parameters(), learning_rate=0.001, regularizer=L2(lambda_=0.001))
    results["L2 (λ=0.001)"] = run_experiment("Reg_l2", model, opt)

    model = build_model_with_dropout(p=0.3)
    opt = Adam(model.parameters(), learning_rate=0.001)
    results["Dropout (p=0.3)"] = run_experiment("Reg_dropout", model, opt)

    model = build_model_with_dropout(p=0.3)
    opt = Adam(model.parameters(), learning_rate=0.001, regularizer=L2(lambda_=0.001))
    results["L2 + Dropout"] = run_experiment("Reg_both", model, opt)

    plot_comparison(
        results,
        "Effect of Regularization on Test Accuracy",
        "Test Accuracy",
        "test_acc",
        os.path.join(RESULTS_DIR, "exp2_regularization.png"),
    )
    plot_comparison(
        results,
        "Effect of Regularization on Training Loss",
        "Training Loss",
        "train_loss",
        os.path.join(RESULTS_DIR, "exp2_regularization_loss.png"),
    )
    save_results("exp2_regularization", results)
    return results

def experiment_l2_lambda():
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 3: L2 Regularization Lambda")
    print("=" * 60)

    results = {}

    for lam in [0.0001, 0.001, 0.01, 0.1]:
        model = build_cifar10_cnn()
        opt = Adam(model.parameters(), learning_rate=0.001, regularizer=L2(lambda_=lam))
        results[f"λ={lam}"] = run_experiment(f"L2_lambda_{lam}", model, opt)

    plot_comparison(
        results,
        "Effect of L2 Lambda on Test Accuracy",
        "Test Accuracy",
        "test_acc",
        os.path.join(RESULTS_DIR, "exp3_l2_lambda.png"),
    )
    plot_comparison(
        results,
        "Effect of L2 Lambda on Training Loss",
        "Training Loss",
        "train_loss",
        os.path.join(RESULTS_DIR, "exp3_l2_lambda_loss.png"),
    )
    save_results("exp3_l2_lambda", results)
    return results

def experiment_optimizer():
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 4: Optimization Algorithms & Learning Rates")
    print("=" * 60)

    results = {}

    model = build_cifar10_cnn()
    opt = Adam(model.parameters(), learning_rate=0.001)
    results["Adam (lr=0.001)"] = run_experiment("Opt_adam", model, opt)

    for lr in [0.1, 0.05, 0.01]:
        model = build_cifar10_cnn()
        opt = Momentum(model.parameters(), learning_rate=lr, beta=0.9)
        results[f"SGD+Momentum (lr={lr})"] = run_experiment(f"Opt_momentum_lr_{lr}", model, opt)

    plot_comparison(
        results,
        "Effect of Optimizer/LR on Test Accuracy",
        "Test Accuracy",
        "test_acc",
        os.path.join(RESULTS_DIR, "exp4_optimizer.png"),
    )
    plot_comparison(
        results,
        "Effect of Optimizer/LR on Training Loss",
        "Training Loss",
        "train_loss",
        os.path.join(RESULTS_DIR, "exp4_optimizer_loss.png"),
    )
    save_results("exp4_optimizer", results)
    return results

def experiment_adam_betas():
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 5: Adam Beta Parameters")
    print("=" * 60)

    results = {}

    configs = [
        ("β1=0.9, β2=0.999 (default)", 0.9, 0.999),
        ("β1=0.95, β2=0.999", 0.95, 0.999),
        ("β1=0.8, β2=0.999", 0.8, 0.999),
        ("β1=0.9, β2=0.99", 0.9, 0.99),
        ("β1=0.9, β2=0.9", 0.9, 0.9),
    ]

    for label, b1, b2 in configs:
        model = build_cifar10_cnn()
        opt = Adam(model.parameters(), learning_rate=0.001, beta1=b1, beta2=b2)
        safe_name = f"Adam_b1_{b1}_b2_{b2}"
        results[label] = run_experiment(safe_name, model, opt)

    plot_comparison(
        results,
        "Effect of Adam Betas on Test Accuracy",
        "Test Accuracy",
        "test_acc",
        os.path.join(RESULTS_DIR, "exp5_adam_betas.png"),
    )
    plot_comparison(
        results,
        "Effect of Adam Betas on Training Loss",
        "Training Loss",
        "train_loss",
        os.path.join(RESULTS_DIR, "exp5_adam_betas_loss.png"),
    )
    save_results("exp5_adam_betas", results)
    return results

def run_experiment_aug(name, model, optimizer, use_aug, epochs=ABLATION_EPOCHS):
    print(f"\n{'=' * 60}")
    print(f"  {name} (Augmentation: {use_aug})")
    print(f"{'=' * 60}")

    model.to(DEVICE)
    loss_fn = CrossEntropy()
    train_loader, test_loader = get_cifar10_dataloaders(batch_size=BATCH_SIZE, num_workers=2, use_augmentation=use_aug)

    history = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        device=DEVICE,
        checkpoint_path=os.path.join(RESULTS_DIR, f"{name.replace(' ', '_')}.pt"),
    )
    return history

def experiment_augmentation():
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 6: Data Augmentation")
    print("=" * 60)

    results = {}

    model = build_cifar10_cnn()
    opt = Adam(model.parameters(), learning_rate=0.001)
    results["With Augmentation"] = run_experiment_aug("Aug_yes", model, opt, use_aug=True)

    model = build_cifar10_cnn()
    opt = Adam(model.parameters(), learning_rate=0.001)
    results["No Augmentation"] = run_experiment_aug("Aug_no", model, opt, use_aug=False)

    plot_comparison(
        results,
        "Effect of Data Augmentation on Test Accuracy",
        "Test Accuracy",
        "test_acc",
        os.path.join(RESULTS_DIR, "exp6_augmentation.png"),
    )
    plot_comparison(
        results,
        "Effect of Data Augmentation on Training Loss",
        "Training Loss",
        "train_loss",
        os.path.join(RESULTS_DIR, "exp6_augmentation_loss.png"),
    )
    save_results("exp6_augmentation", results)
    return results

def build_custom_cnn(filters=[32, 64, 128]):
    f1, f2, f3 = filters
    return NeuralNetwork(
        [
            Convolutional(3, f1, 3, 1, 1),
            ActivationLayer(ReLU()),
            Convolutional(f1, f2, 3, 2, 1),
            ActivationLayer(ReLU()),
            Convolutional(f2, f3, 3, 2, 1),
            ActivationLayer(ReLU()),
            Flatten(),
            FullyConnected(f3 * 8 * 8, 512, ReLU()),
            FullyConnected(512, 10, Softmax()),
        ]
    )

def experiment_width():
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 7: Architecture Width (Filters)")
    print("=" * 60)

    results = {}

    widths = {
        "Slim (16-32-64)": [16, 32, 64],
        "Standard (32-64-128)": [32, 64, 128],
        "Wide (64-128-256)": [64, 128, 256],
    }

    for label, filters in widths.items():
        model = build_custom_cnn(filters)
        opt = Adam(model.parameters(), learning_rate=0.001)
        results[label] = run_experiment(f"Width_{label.split()[0]}", model, opt)

    plot_comparison(
        results,
        "Effect of Model Width on Test Accuracy",
        "Test Accuracy",
        "test_acc",
        os.path.join(RESULTS_DIR, "exp7_width.png"),
    )
    plot_comparison(
        results,
        "Effect of Model Width on Training Loss",
        "Training Loss",
        "train_loss",
        os.path.join(RESULTS_DIR, "exp7_width_loss.png"),
    )
    save_results("exp7_width", results)
    return results

def build_cnn_with_activation(activation_class):
    return NeuralNetwork(
        [
            Convolutional(3, 32, 3, 1, 1),
            ActivationLayer(activation_class()),
            Convolutional(32, 64, 3, 2, 1),
            ActivationLayer(activation_class()),
            Convolutional(64, 128, 3, 2, 1),
            ActivationLayer(activation_class()),
            Flatten(),
            FullyConnected(128 * 8 * 8, 512, activation_class()),
            FullyConnected(512, 10, Softmax()),
        ]
    )

def experiment_activation():
    print("\n\n" + "=" * 60)
    print("  EXPERIMENT 8: Activation Functions")
    print("=" * 60)

    results = {}

    model = build_cnn_with_activation(ReLU)
    opt = Adam(model.parameters(), learning_rate=0.001)
    results["ReLU"] = run_experiment("Act_relu", model, opt)

    model = build_cnn_with_activation(Sigmoid)
    opt = Adam(model.parameters(), learning_rate=0.001)
    results["Sigmoid"] = run_experiment("Act_sigmoid", model, opt)

    plot_comparison(
        results,
        "Effect of Activation on Test Accuracy",
        "Test Accuracy",
        "test_acc",
        os.path.join(RESULTS_DIR, "exp8_activation.png"),
    )
    plot_comparison(
        results,
        "Effect of Activation on Training Loss",
        "Training Loss",
        "train_loss",
        os.path.join(RESULTS_DIR, "exp8_activation_loss.png"),
    )
    save_results("exp8_activation", results)
    return results

def main():
    parser = argparse.ArgumentParser(description="Run specific ablation experiments.")
    parser.add_argument(
        "--exp", type=int, choices=range(1, 9), help="Experiment number to run (1-8). If omitted, runs all."
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Ablation epochs: {ABLATION_EPOCHS}")

    exps = {
        1: experiment_lr_decay,
        2: experiment_regularization,
        3: experiment_l2_lambda,
        4: experiment_optimizer,
        5: experiment_adam_betas,
        6: experiment_augmentation,
        7: experiment_width,
        8: experiment_activation,
    }

    if args.exp:
        print(f"Running Experiment {args.exp}...")
        exps[args.exp]()
    else:
        print("Running ALL experiments...")
        for func in exps.values():
            func()

    print("\n\nDone!")

if __name__ == "__main__":
    main()
