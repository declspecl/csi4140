import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.model import CIFAR10CNN
from src.prune import apply_global_pruning, apply_pruning, compute_flops, count_parameters, remove_masks
from src.train import evaluate, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOTS_DIR = "results/plots"
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 10
FIXED_SPARSITY = 0.7


def _load_model(checkpoint: str) -> nn.Module:
    model = CIFAR10CNN().to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    return model


def _finetune(model: nn.Module, train_loader, test_loader, epochs: int, tag: str) -> None:
    if epochs == 0:
        return
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
    train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=DEVICE,
        checkpoint_path=f"/tmp/prune_{tag}.pt",
        verbose=False,
    )


def _save_plot(filename: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def _print_table(headers: list[str], rows: list[list]) -> None:
    col_widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))


def run_sparsity_sweep(checkpoint: str, train_loader, test_loader) -> None:
    print("\n" + "=" * 60)
    print("Aspect 1: Sparsity Sweep")
    print("=" * 60)

    sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    criterion = nn.CrossEntropyLoss()
    flops_base = compute_flops(_load_model(checkpoint))
    rows: list[dict] = []

    for sparsity in sparsity_levels:
        print(f"\n  Sparsity {sparsity:.0%} ...", flush=True)
        model = _load_model(checkpoint)

        if sparsity > 0.0:
            apply_pruning(model, amount=sparsity, scope="all")
            remove_masks(model)
            _finetune(model, train_loader, test_loader, FINETUNE_EPOCHS, f"sparsity_{sparsity:.2f}")

        _, train_acc = evaluate(model, train_loader, criterion, DEVICE)
        _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        total, nonzero = count_parameters(model)

        rows.append({
            "sparsity": sparsity,
            "nonzero": nonzero,
            "total": total,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "flops": flops_base,
            "eff_flops": int(flops_base * (1 - sparsity)),
        })
        print(f"    Nonzero: {nonzero:,} / {total:,} ({nonzero/total:.1%})")
        print(f"    Train acc: {train_acc:.4f}  Test acc: {test_acc:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot([r["nonzero"] for r in rows], [r["test_acc"] for r in rows], marker="o", linewidth=2)
    for r in rows:
        plt.annotate(f"{r['sparsity']:.0%}", (r["nonzero"], r["test_acc"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)
    plt.xlabel("Remaining Parameters (non-zero)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Remaining Parameters (L1 Pruning + Fine-tuning)")
    plt.grid(True)
    plt.tight_layout()
    _save_plot("prune_sparsity_sweep.png")

    print("\nSummary:")
    _print_table(
        ["Sparsity", "Nonzero Params", "Total Params", "Train Acc", "Test Acc", "Eff. FLOPs"],
        [[f"{r['sparsity']:.0%}", f"{r['nonzero']:,}", f"{r['total']:,}",
          f"{r['train_acc']:.4f}", f"{r['test_acc']:.4f}", f"{r['eff_flops']:,}"] for r in rows],
    )


def run_finetune_ablation(checkpoint: str, train_loader, test_loader) -> None:
    print("\n" + "=" * 60)
    print("Aspect 2: Fine-tuning Epochs Ablation (sparsity=70%)")
    print("=" * 60)

    epoch_options = [0, 5, 10, 20]
    criterion = nn.CrossEntropyLoss()
    rows: list[dict] = []

    for epochs in epoch_options:
        print(f"\n  Fine-tune epochs: {epochs} ...", flush=True)
        model = _load_model(checkpoint)
        apply_pruning(model, amount=FIXED_SPARSITY, scope="all")
        remove_masks(model)
        _finetune(model, train_loader, test_loader, epochs, f"ft_{epochs}")

        _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        rows.append({"epochs": epochs, "test_acc": test_acc})
        print(f"    Test acc: {test_acc:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot([r["epochs"] for r in rows], [r["test_acc"] for r in rows], marker="o", linewidth=2)
    plt.xlabel("Fine-tuning Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy Recovery vs Fine-tuning Epochs (sparsity=70%)")
    plt.xticks([r["epochs"] for r in rows])
    plt.grid(True)
    plt.tight_layout()
    _save_plot("prune_finetune_ablation.png")

    print("\nSummary:")
    _print_table(["Fine-tune Epochs", "Test Acc"],
                 [[r["epochs"], f"{r['test_acc']:.4f}"] for r in rows])


def run_scope_ablation(checkpoint: str, train_loader, test_loader) -> None:
    print("\n" + "=" * 60)
    print("Aspect 3: Pruning Scope Ablation (sparsity=70%)")
    print("=" * 60)

    scopes = [("all", "All layers"), ("conv", "Conv only"), ("fc", "FC only")]
    criterion = nn.CrossEntropyLoss()
    rows: list[dict] = []

    for scope, label in scopes:
        print(f"\n  Scope: {label} ...", flush=True)
        model = _load_model(checkpoint)
        apply_pruning(model, amount=FIXED_SPARSITY, scope=scope)
        remove_masks(model)
        _finetune(model, train_loader, test_loader, FINETUNE_EPOCHS, f"scope_{scope}")

        _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        _, nonzero = count_parameters(model)
        rows.append({"label": label, "nonzero": nonzero, "test_acc": test_acc})
        print(f"    Nonzero: {nonzero:,}  Test acc: {test_acc:.4f}")

    plt.figure(figsize=(6, 4))
    bars = plt.bar([r["label"] for r in rows], [r["test_acc"] for r in rows],
                   color=["#2563eb", "#16a34a", "#dc2626"], width=0.5)
    plt.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by Pruning Scope (sparsity=70%, 10 fine-tune epochs)")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    _save_plot("prune_scope_ablation.png")

    print("\nSummary:")
    _print_table(["Scope", "Nonzero Params", "Test Acc"],
                 [[r["label"], f"{r['nonzero']:,}", f"{r['test_acc']:.4f}"] for r in rows])


def run_global_vs_local(checkpoint: str, train_loader, test_loader) -> None:
    print("\n" + "=" * 60)
    print("Aspect 4: Global vs Local Pruning (sparsity=70%)")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    strategies = [
        ("Local (per-layer)", lambda m: apply_pruning(m, amount=FIXED_SPARSITY, scope="all")),
        ("Global (network-wide)", lambda m: apply_global_pruning(m, amount=FIXED_SPARSITY)),
    ]
    rows: list[dict] = []

    for label, prune_fn in strategies:
        print(f"\n  Strategy: {label} ...", flush=True)
        model = _load_model(checkpoint)
        prune_fn(model)
        remove_masks(model)
        _finetune(model, train_loader, test_loader, FINETUNE_EPOCHS, f"strategy_{label[:3].lower()}")

        _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        _, nonzero = count_parameters(model)
        rows.append({"label": label, "nonzero": nonzero, "test_acc": test_acc})
        print(f"    Nonzero: {nonzero:,}  Test acc: {test_acc:.4f}")

    plt.figure(figsize=(6, 4))
    bars = plt.bar([r["label"] for r in rows], [r["test_acc"] for r in rows],
                   color=["#2563eb", "#f59e0b"], width=0.4)
    plt.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    plt.ylabel("Test Accuracy")
    plt.title("Global vs Local Pruning (sparsity=70%, 10 fine-tune epochs)")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    _save_plot("prune_global_vs_local.png")

    print("\nSummary:")
    _print_table(["Strategy", "Nonzero Params", "Test Acc"],
                 [[r["label"], f"{r['nonzero']:,}", f"{r['test_acc']:.4f}"] for r in rows])


def run_iterative_vs_oneshot(checkpoint: str, train_loader, test_loader) -> None:
    print("\n" + "=" * 60)
    print("Aspect 5: Iterative vs One-shot Pruning (~70% sparsity, 10 total fine-tune epochs)")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    rows: list[dict] = []

    print("\n  One-shot ...", flush=True)
    model = _load_model(checkpoint)
    apply_pruning(model, amount=FIXED_SPARSITY, scope="all")
    remove_masks(model)
    _finetune(model, train_loader, test_loader, 10, "oneshot")
    _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    _, nonzero = count_parameters(model)
    rows.append({"label": "One-shot", "nonzero": nonzero, "test_acc": test_acc})
    print(f"    Nonzero: {nonzero:,}  Test acc: {test_acc:.4f}")

    print("\n  Iterative (3 rounds x 33% of remaining, 3-4 epochs each) ...", flush=True)
    model = _load_model(checkpoint)
    for i, epochs in enumerate([3, 3, 4]):
        apply_pruning(model, amount=0.33, scope="all")
        remove_masks(model)
        _finetune(model, train_loader, test_loader, epochs, f"iterative_r{i}")
    _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    _, nonzero = count_parameters(model)
    rows.append({"label": "Iterative (3 rounds)", "nonzero": nonzero, "test_acc": test_acc})
    print(f"    Nonzero: {nonzero:,}  Test acc: {test_acc:.4f}")

    plt.figure(figsize=(6, 4))
    bars = plt.bar([r["label"] for r in rows], [r["test_acc"] for r in rows],
                   color=["#2563eb", "#16a34a"], width=0.4)
    plt.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    plt.ylabel("Test Accuracy")
    plt.title("One-shot vs Iterative Pruning (~70% sparsity, 10 total fine-tune epochs)")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    _save_plot("prune_iterative_vs_oneshot.png")

    print("\nSummary:")
    _print_table(["Strategy", "Nonzero Params", "Test Acc"],
                 [[r["label"], f"{r['nonzero']:,}", f"{r['test_acc']:.4f}"] for r in rows])


def run_layer_sensitivity(checkpoint: str, train_loader, test_loader) -> None:
    print("\n" + "=" * 60)
    print("Aspect 6: Per-layer Sensitivity (sparsity=70%, no fine-tuning)")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()
    layers = ["conv1", "conv2", "conv3", "fc1", "fc2"]
    rows: list[dict] = []

    _, baseline_acc = evaluate(_load_model(checkpoint), test_loader, criterion, DEVICE)
    print(f"  Baseline test acc: {baseline_acc:.4f}")

    for layer_name in layers:
        print(f"\n  Pruning {layer_name} only ...", flush=True)
        model = _load_model(checkpoint)
        module = getattr(model, layer_name)
        import torch.nn.utils.prune as prune_utils
        prune_utils.l1_unstructured(module, name="weight", amount=FIXED_SPARSITY)
        prune_utils.remove(module, "weight")

        _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        drop = baseline_acc - test_acc
        rows.append({"layer": layer_name, "test_acc": test_acc, "drop": drop})
        print(f"    Test acc: {test_acc:.4f}  Drop: {drop:.4f}")

    plt.figure(figsize=(7, 4))
    colors = ["#dc2626" if r["drop"] > 0.02 else "#2563eb" for r in rows]
    bars = plt.bar([r["layer"] for r in rows], [r["drop"] for r in rows], color=colors, width=0.5)
    plt.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Layer")
    plt.ylabel("Accuracy Drop")
    plt.title("Per-layer Sensitivity to 70% Pruning (no fine-tuning)")
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    _save_plot("prune_layer_sensitivity.png")

    print("\nSummary:")
    _print_table(["Layer", "Test Acc", "Accuracy Drop"],
                 [[r["layer"], f"{r['test_acc']:.4f}", f"{r['drop']:.4f}"] for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--aspect", type=int, choices=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)

    if args.aspect in (None, 1):
        run_sparsity_sweep(args.checkpoint, train_loader, test_loader)
    if args.aspect in (None, 2):
        run_finetune_ablation(args.checkpoint, train_loader, test_loader)
    if args.aspect in (None, 3):
        run_scope_ablation(args.checkpoint, train_loader, test_loader)
    if args.aspect in (None, 4):
        run_global_vs_local(args.checkpoint, train_loader, test_loader)
    if args.aspect in (None, 5):
        run_iterative_vs_oneshot(args.checkpoint, train_loader, test_loader)
    if args.aspect in (None, 6):
        run_layer_sensitivity(args.checkpoint, train_loader, test_loader)

    print("\nAll experiments complete. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
