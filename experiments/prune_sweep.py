"""
Project 2 — Pruning experiments.

Runs three ablation aspects and saves all plots + a summary table.

Usage:
    python -m experiments.prune_sweep --checkpoint best_model.pt

Aspects
-------
1. Sparsity sweep        — accuracy vs remaining parameters (required plot)
2. Fine-tuning ablation  — with vs without fine-tuning after pruning
3. Pruning scope         — all layers vs conv-only vs fc-only
"""

import argparse
import copy
import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.data import get_dataloaders
from src.model import CIFAR10CNN
from src.prune import apply_pruning, remove_masks, count_parameters, compute_flops
from src.train import evaluate, train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PLOTS_DIR = "results/plots"
FINETUNE_LR = 1e-4
FINETUNE_EPOCHS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Aspect 1: Sparsity sweep  (required visualization)
# ---------------------------------------------------------------------------

def run_sparsity_sweep(checkpoint: str, train_loader, test_loader) -> None:
    print("\n" + "=" * 60)
    print("Aspect 1: Sparsity Sweep")
    print("=" * 60)

    sparsity_levels = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
    criterion = nn.CrossEntropyLoss()

    total_params_base, _ = count_parameters(_load_model(checkpoint))
    flops_base = compute_flops(_load_model(checkpoint))

    rows: list[dict] = []

    for sparsity in sparsity_levels:
        print(f"\n  Sparsity {sparsity:.0%} ...", flush=True)

        model = _load_model(checkpoint)

        if sparsity > 0.0:
            apply_pruning(model, amount=sparsity, scope="all")
            remove_masks(model)
            _finetune(model, train_loader, test_loader, FINETUNE_EPOCHS, f"sparsity_{sparsity:.2f}")

        train_loss, train_acc = evaluate(model, train_loader, criterion, DEVICE)
        _, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        total, nonzero = count_parameters(model)

        rows.append({
            "sparsity": sparsity,
            "nonzero": nonzero,
            "total": total,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "flops": flops_base,  # dense FLOPs don't change with unstructured pruning
            "eff_flops": int(flops_base * (1 - sparsity)),
        })

        print(f"    Nonzero params : {nonzero:>12,} / {total:,} ({nonzero/total:.1%})")
        print(f"    Train accuracy : {train_acc:.4f}")
        print(f"    Test accuracy  : {test_acc:.4f}")
        print(f"    Dense FLOPs    : {flops_base:>12,}")
        print(f"    Effective FLOPs: {int(flops_base * (1 - sparsity)):>12,}")

    # --- required plot: test accuracy vs remaining parameters ---
    nonzero_params = [r["nonzero"] for r in rows]
    test_accs = [r["test_acc"] for r in rows]

    plt.figure(figsize=(7, 5))
    plt.plot(nonzero_params, test_accs, marker="o", linewidth=2)
    for r in rows:
        plt.annotate(
            f"{r['sparsity']:.0%}",
            (r["nonzero"], r["test_acc"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    plt.xlabel("Remaining Parameters (non-zero)")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Remaining Parameters (L1 Pruning + Fine-tuning)")
    plt.grid(True)
    plt.tight_layout()
    _save_plot("prune_sparsity_sweep.png")

    # --- summary table ---
    print("\nSummary:")
    _print_table(
        ["Sparsity", "Nonzero Params", "Total Params", "Train Acc", "Test Acc", "Eff. FLOPs"],
        [
            [
                f"{r['sparsity']:.0%}",
                f"{r['nonzero']:,}",
                f"{r['total']:,}",
                f"{r['train_acc']:.4f}",
                f"{r['test_acc']:.4f}",
                f"{r['eff_flops']:,}",
            ]
            for r in rows
        ],
    )


# ---------------------------------------------------------------------------
# Aspect 2: Fine-tuning ablation
# Compare: no fine-tuning vs 5 / 10 / 20 fine-tune epochs at fixed sparsity
# ---------------------------------------------------------------------------

def run_finetune_ablation(checkpoint: str, train_loader, test_loader) -> None:
    print("\n" + "=" * 60)
    print("Aspect 2: Fine-tuning Epochs Ablation (sparsity=70%)")
    print("=" * 60)

    FIXED_SPARSITY = 0.7
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
        print(f"    Test accuracy: {test_acc:.4f}")

    epoch_vals = [r["epochs"] for r in rows]
    test_accs = [r["test_acc"] for r in rows]

    plt.figure(figsize=(6, 4))
    plt.plot(epoch_vals, test_accs, marker="o", linewidth=2)
    plt.xlabel("Fine-tuning Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Accuracy Recovery vs Fine-tuning Epochs (sparsity=70%)")
    plt.xticks(epoch_vals)
    plt.grid(True)
    plt.tight_layout()
    _save_plot("prune_finetune_ablation.png")

    print("\nSummary:")
    _print_table(
        ["Fine-tune Epochs", "Test Acc"],
        [[r["epochs"], f"{r['test_acc']:.4f}"] for r in rows],
    )


# ---------------------------------------------------------------------------
# Aspect 3: Pruning scope ablation
# Compare: all layers vs conv-only vs fc-only at fixed sparsity
# ---------------------------------------------------------------------------

def run_scope_ablation(checkpoint: str, train_loader, test_loader) -> None:
    print("\n" + "=" * 60)
    print("Aspect 3: Pruning Scope Ablation (sparsity=70%)")
    print("=" * 60)

    FIXED_SPARSITY = 0.7
    scopes = [
        ("all",  "All layers"),
        ("conv", "Conv only"),
        ("fc",   "FC only"),
    ]
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
        print(f"    Nonzero params: {nonzero:,}")
        print(f"    Test accuracy : {test_acc:.4f}")

    labels = [r["label"] for r in rows]
    test_accs = [r["test_acc"] for r in rows]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, test_accs, color=["#2563eb", "#16a34a", "#dc2626"], width=0.5)
    plt.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy by Pruning Scope (sparsity=70%, 10 fine-tune epochs)")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    _save_plot("prune_scope_ablation.png")

    print("\nSummary:")
    _print_table(
        ["Scope", "Nonzero Params", "Test Acc"],
        [[r["label"], f"{r['nonzero']:,}", f"{r['test_acc']:.4f}"] for r in rows],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Pruning ablation experiments for Project 2")
    parser.add_argument("--checkpoint", default="best_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--aspect", type=int, choices=[1, 2, 3], help="Run only one aspect (default: all)")
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

    print("\nAll experiments complete. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
