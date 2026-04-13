import argparse
import copy
import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from src.data import get_dataloaders
from src.model import CIFAR10CNN
from src.prune import compute_flops
from src.train import evaluate

PLOTS_DIR = "results/plots"
RESULTS_DIR = "results/quantize"
LATENCY_BATCHES = 50


def _state_dict_size_bytes(model: nn.Module) -> int:
    tmp = "/tmp/_qsize.pt"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp)
    os.remove(tmp)
    return size


def _measure_latency_cpu(model: nn.Module, batch_size: int = 1, n_batches: int = LATENCY_BATCHES) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        t0 = time.perf_counter()
        for _ in range(n_batches):
            model(x)
        elapsed = time.perf_counter() - t0
    return (elapsed / n_batches) * 1000.0


def _load_fp32(checkpoint: str) -> nn.Module:
    model = CIFAR10CNN()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _make_fp16_weights(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.data = p.data.half().float()
    return m


def _make_int8_dynamic(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    return torch.quantization.quantize_dynamic(m, {nn.Linear}, dtype=torch.qint8)


def _int8_dynamic_size_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        packed = getattr(module, "_packed_params", None)
        if packed is not None:
            w, _ = packed._packed_params
            total += w.numel()
        elif isinstance(module, (nn.Conv2d,)):
            total += module.weight.numel() * 4
            if module.bias is not None:
                total += module.bias.numel() * 4
        elif isinstance(module, nn.Linear):
            total += module.weight.numel() * 4
            if module.bias is not None:
                total += module.bias.numel() * 4
    return total


def _eval_cpu(model: nn.Module, test_loader) -> float:
    criterion = nn.CrossEntropyLoss()
    _, acc = evaluate(model, test_loader, criterion, "cpu")
    return acc


def run(checkpoint: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _, test_loader = get_dataloaders(batch_size=128, num_workers=0)

    print("=" * 60)
    print("Post-Training Quantization Sweep")
    print("=" * 60)

    fp32 = _load_fp32(checkpoint)
    fp32_flops = compute_flops(fp32)

    configs = []

    print("\n[1/3] FP32 baseline ...", flush=True)
    acc = _eval_cpu(fp32, test_loader)
    size = _state_dict_size_bytes(fp32)
    lat1 = _measure_latency_cpu(fp32, batch_size=1)
    lat128 = _measure_latency_cpu(fp32, batch_size=128, n_batches=10)
    configs.append({
        "name": "FP32",
        "bits": 32,
        "test_acc": acc,
        "size_bytes": size,
        "size_mb": size / 1e6,
        "latency_ms_b1": lat1,
        "latency_ms_b128": lat128,
        "flops": fp32_flops,
    })
    print(f"  acc={acc:.4f}  size={size/1e6:.2f} MB  lat(b=1)={lat1:.2f} ms")

    print("\n[2/3] FP16 weights ...", flush=True)
    fp16 = _make_fp16_weights(fp32)
    acc = _eval_cpu(fp16, test_loader)
    size_fp16 = _state_dict_size_bytes(fp32) // 2
    lat1 = _measure_latency_cpu(fp16, batch_size=1)
    lat128 = _measure_latency_cpu(fp16, batch_size=128, n_batches=10)
    configs.append({
        "name": "FP16",
        "bits": 16,
        "test_acc": acc,
        "size_bytes": size_fp16,
        "size_mb": size_fp16 / 1e6,
        "latency_ms_b1": lat1,
        "latency_ms_b128": lat128,
        "flops": fp32_flops,
    })
    print(f"  acc={acc:.4f}  size={size_fp16/1e6:.2f} MB  lat(b=1)={lat1:.2f} ms")

    print("\n[3/3] INT8 dynamic (Linear layers) ...", flush=True)
    int8 = _make_int8_dynamic(fp32)
    acc = _eval_cpu(int8, test_loader)
    size_int8 = _int8_dynamic_size_bytes(int8)
    lat1 = _measure_latency_cpu(int8, batch_size=1)
    lat128 = _measure_latency_cpu(int8, batch_size=128, n_batches=10)
    configs.append({
        "name": "INT8-dynamic",
        "bits": 8,
        "test_acc": acc,
        "size_bytes": size_int8,
        "size_mb": size_int8 / 1e6,
        "latency_ms_b1": lat1,
        "latency_ms_b128": lat128,
        "flops": fp32_flops,
    })
    print(f"  acc={acc:.4f}  size={size_int8/1e6:.2f} MB  lat(b=1)={lat1:.2f} ms")

    out_path = os.path.join(RESULTS_DIR, "quantize_sweep.json")
    with open(out_path, "w") as f:
        json.dump(configs, f, indent=2)
    print(f"\nSaved: {out_path}")

    plt.figure(figsize=(6, 4))
    xs = [c["bits"] for c in configs]
    ys = [c["test_acc"] for c in configs]
    labels = [c["name"] for c in configs]
    plt.plot(xs, ys, marker="o", linewidth=2)
    for x, y, lbl in zip(xs, ys, labels):
        plt.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.xlabel("Weight Bit-width")
    plt.ylabel("Test Accuracy")
    plt.title("Test Accuracy vs Bit-width (Post-Training Quantization)")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "quantize_accuracy_vs_bits.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, [c["size_mb"] for c in configs], color=["#2563eb", "#16a34a", "#dc2626"])
    plt.bar_label(bars, fmt="%.2f MB", padding=3, fontsize=9)
    plt.ylabel("Model Size (MB)")
    plt.title("Model Size by Quantization Method")
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "quantize_size.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    plt.figure(figsize=(7, 4))
    x = range(len(labels))
    w = 0.35
    plt.bar([i - w / 2 for i in x], [c["latency_ms_b1"] for c in configs], w, label="batch=1", color="#2563eb")
    plt.bar([i + w / 2 for i in x], [c["latency_ms_b128"] for c in configs], w, label="batch=128", color="#f59e0b")
    plt.xticks(list(x), labels)
    plt.ylabel("Latency per batch (ms)")
    plt.title("CPU Inference Latency by Quantization Method")
    plt.legend()
    plt.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "quantize_latency.png")
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")

    print("\nSummary:")
    print(f"{'Method':<14} {'Bits':>5} {'Test Acc':>9} {'Size MB':>9} {'Lat b1 ms':>11} {'Lat b128 ms':>13}")
    for c in configs:
        print(f"{c['name']:<14} {c['bits']:>5} {c['test_acc']:>9.4f} "
              f"{c['size_mb']:>9.2f} {c['latency_ms_b1']:>11.2f} {c['latency_ms_b128']:>13.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best_model.pt")
    args = parser.parse_args()
    run(args.checkpoint)


if __name__ == "__main__":
    main()
