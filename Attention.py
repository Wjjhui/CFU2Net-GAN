import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_einsum(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [B, N, d].")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes.")
    scale = 1.0 / math.sqrt(q.shape[-1])
    weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
    return torch.matmul(weights, v)


class AttentionLayer(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        ratio_kq: int = 8,
        ratio_v: int = 8,
    ):
        super().__init__()
        if input_channels != output_channels:
            raise ValueError("Residual attention requires input_channels == output_channels.")
        if ratio_kq != ratio_v:
            raise ValueError("The Method requires identical reduced dimensions for Q, K, and V.")
        if output_channels % ratio_kq != 0:
            raise ValueError("The channel count must be divisible by the attention reduction ratio.")
        reduced_channels = output_channels // ratio_kq
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.reduced_channels = reduced_channels
        self.query = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, bias=False)
        self.key = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, bias=False)
        self.value = nn.Conv2d(input_channels, reduced_channels, kernel_size=1, bias=False)
        self.last_conv = nn.Conv2d(reduced_channels, output_channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Attention input must have shape [B, C, H, W].")
        batch, channels, height, width = x.shape
        if channels != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, received {channels}."
            )
        q = self.query(x).flatten(2).transpose(1, 2)
        k = self.key(x).flatten(2).transpose(1, 2)
        v = self.value(x).flatten(2).transpose(1, 2)
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q.unsqueeze(1),
                k.unsqueeze(1),
                v.unsqueeze(1),
                dropout_p=0.0,
                is_causal=False,
            ).squeeze(1)
        else:
            out = attention_einsum(q, k, v)
        out = out.transpose(1, 2).reshape(
            batch, self.reduced_channels, height, width
        )
        return x + self.gamma * self.last_conv(out)
