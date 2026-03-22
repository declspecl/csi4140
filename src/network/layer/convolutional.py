from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self._input_unfolded = None
        self._input_shape = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got {input.dim()}D")

        N, C_in, H, W = input.shape
        K = self.kernel_size
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1

        # im2col via F.unfold: extracts sliding patches as columns
        # (N, C_in*K*K, H_out*W_out)
        input_unfolded = F.unfold(input, K, padding=self.padding, stride=self.stride)
        self._input_unfolded = input_unfolded
        self._input_shape = input.shape

        # Reshape weight to (out_channels, C_in*K*K) for batched matmul
        weight_matrix = self.weight.view(self.out_channels, -1)

        # (N, out_channels, H_out*W_out) via batched matmul
        output = weight_matrix @ input_unfolded + self.bias.view(-1, 1)

        return output.view(N, self.out_channels, H_out, W_out)

    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self._input_unfolded is None:
            raise ValueError("Must call forward() before backward()")

        input_unfolded = self._input_unfolded
        N, C_in, H, W = self._input_shape
        K = self.kernel_size
        N, C_out, H_out, W_out = grad_output.shape

        # Reshape grad to (N, out_channels, H_out*W_out)
        grad_matrix = grad_output.view(N, C_out, H_out * W_out)

        # Weight gradient: sum over batch of grad_matrix @ input_unfolded^T
        # (N, out_channels, H_out*W_out) @ (N, H_out*W_out, C_in*K*K) -> (N, out_channels, C_in*K*K)
        weight_matrix = self.weight.view(self.out_channels, -1)
        self.weight.grad = (grad_matrix @ input_unfolded.transpose(1, 2)).sum(dim=0).view(self.weight.shape)
        self.bias.grad = grad_output.sum(dim=(0, 2, 3))

        # Input gradient: weight^T @ grad_matrix, then fold back
        # (N, C_in*K*K, H_out*W_out)
        grad_unfolded = weight_matrix.t() @ grad_matrix
        grad_input = F.fold(grad_unfolded, (H, W), K, padding=self.padding, stride=self.stride)

        self._input_unfolded = None  # FIX: free cached input to reduce peak GPU memory
        self._input_shape = None

        return grad_input

    def parameters(self) -> Dict[ParameterType, nn.Parameter]:
        return {
            ParameterType.WEIGHT: self.weight,
            ParameterType.BIAS: self.bias,
        }
