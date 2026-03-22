from typing import Dict

import torch
import torch.nn as nn

from src.network import ParameterType
from src.network.layer import Layer


class Convolutional(Layer):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        std = torch.sqrt(torch.tensor(2.0 / (in_channels * kernel_size * kernel_size)))
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * std)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self._input_cache = None

    def _pad(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding == 0:
            return input
        N, C, H, W = input.shape
        padded = torch.zeros(N, C, H + 2 * self.padding, W + 2 * self.padding, dtype=input.dtype, device=input.device)
        padded[:, :, self.padding : self.padding + H, self.padding : self.padding + W] = input
        return padded

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got {input.dim()}D")

        N, C_in, H, W = input.shape
        K = self.kernel_size
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1

        padded = self._pad(input)
        self._input_cache = padded

        output = torch.zeros(N, self.out_channels, H_out, W_out, dtype=input.dtype, device=input.device)

        for kh in range(K):
            for kw in range(K):
                input_slice = padded[:, :, kh :: self.stride, kw :: self.stride][:, :, :H_out, :W_out]
                output += torch.einsum("nchw,oc->nohw", input_slice, self.weight[:, :, kh, kw])

        output += self.bias.view(1, -1, 1, 1)
        return output

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self._input_cache is None:
            raise ValueError("Must call forward() before backward()")

        padded = self._input_cache
        N, C_out, H_out, W_out = grad_output.shape
        K = self.kernel_size

        grad_weight = torch.zeros_like(self.weight)
        grad_padded = torch.zeros_like(padded)

        for kh in range(K):
            for kw in range(K):
                input_slice = padded[:, :, kh :: self.stride, kw :: self.stride][:, :, :H_out, :W_out]
                grad_weight[:, :, kh, kw] = torch.einsum("nohw,nchw->oc", grad_output, input_slice)
                grad_padded[:, :, kh :: self.stride, kw :: self.stride][:, :, :H_out, :W_out] += torch.einsum(
                    "nohw,oc->nchw", grad_output, self.weight[:, :, kh, kw]
                )

        self.weight.grad = grad_weight
        self.bias.grad = grad_output.sum(dim=(0, 2, 3))

        if self.padding > 0:
            grad_input = grad_padded[:, :, self.padding : -self.padding, self.padding : -self.padding]
        else:
            grad_input = grad_padded

        return grad_input

    def parameters(self) -> Dict[ParameterType, nn.Parameter]:
        return {
            ParameterType.WEIGHT: self.weight,
            ParameterType.BIAS: self.bias,
        }
