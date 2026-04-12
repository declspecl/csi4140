import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch import Tensor


def apply_pruning(model: nn.Module, amount: float, scope: str = "all") -> nn.Module:
    for module in model.modules():
        if scope in ("all", "conv") and isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
        if scope in ("all", "fc") and isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
    return model


def apply_global_pruning(model: nn.Module, amount: float) -> nn.Module:
    parameters = [
        (module, "weight")
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    prune.global_unstructured(parameters, pruning_method=prune.L1Unstructured, amount=amount)
    return model


def remove_masks(model: nn.Module) -> nn.Module:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total: int = sum(p.numel() for p in model.parameters())
    nonzero: int = sum(int((p != 0).sum().item()) for p in model.parameters())
    return total, nonzero


def compute_flops(model: nn.Module, input_size: tuple[int, ...] = (1, 3, 32, 32)) -> int:
    flops: list[int] = [0]
    hooks = []

    def conv_hook(module: nn.Conv2d, _input: tuple, output: Tensor) -> None:
        n, c_out, h_out, w_out = output.shape
        kh, kw = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        flops[0] += 2 * n * module.in_channels * kh * kw * c_out * h_out * w_out

    def linear_hook(module: nn.Linear, input: tuple, _output: Tensor) -> None:
        n = input[0].shape[0]
        flops[0] += 2 * n * module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    x = torch.zeros(*input_size)
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return flops[0]
