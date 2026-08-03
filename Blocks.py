from typing import Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Conv3d
from torch.nn.utils.parametrizations import spectral_norm

from Attention import AttentionLayer


def get_conv_layer(conv_type: str = "standard") -> Type[Union[Conv2d, Conv3d]]:
    if conv_type == "standard":
        return nn.Conv2d
    if conv_type == "3d":
        return nn.Conv3d
    raise ValueError(f"{conv_type} is not a recognized convolution type.")


class ConvGRU(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        sn_eps: float = 0.0001,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.read_gate_conv = spectral_norm(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding),
            eps=sn_eps,
        )
        self.update_gate_conv = spectral_norm(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding),
            eps=sn_eps,
        )
        self.output_conv = spectral_norm(
            nn.Conv2d(input_channels, output_channels, kernel_size, padding=padding),
            eps=sn_eps,
        )
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self, x: torch.Tensor, previous_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[0] != previous_state.shape[0] or x.shape[2:] != previous_state.shape[2:]:
            raise ValueError("ConvGRU feature and state batch/spatial dimensions must match.")
        combined = torch.cat([x, previous_state], dim=1)
        if combined.shape[1] != self.input_channels:
            raise ValueError(
                f"ConvGRU expected {self.input_channels} concatenated channels, received {combined.shape[1]}."
            )
        read_gate = self.sigmoid(self.read_gate_conv(combined))
        update_gate = self.sigmoid(self.update_gate_conv(combined))
        candidate_input = torch.cat([x, read_gate * previous_state], dim=1)
        candidate = self.relu(self.output_conv(candidate_input))
        new_state = update_gate * previous_state + (1.0 - update_gate) * candidate
        return new_state, new_state


class My_GRU(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        sn_eps: float = 0.0001,
        state_channels: int | None = None,
    ):
        super().__init__()
        if input_channels != 2 * output_channels:
            raise ValueError("My_GRU requires input_channels == 2 * output_channels.")
        self.output_channels = output_channels
        self.state_channels = output_channels if state_channels is None else state_channels
        if self.state_channels == output_channels:
            self.state_projection = nn.Identity()
        else:
            self.state_projection = spectral_norm(
                nn.Conv2d(self.state_channels, output_channels, kernel_size=1, bias=False),
                eps=sn_eps,
            )
        self.GRU = ConvGRU(input_channels, output_channels, kernel_size, sn_eps)

    def forward(
        self, x: list[torch.Tensor] | torch.Tensor, pre_state: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            if x.ndim != 5:
                raise ValueError("Temporal decoder features must have shape [T, B, C, H, W].")
            features = list(x.unbind(0))
        else:
            features = list(x)
        if not features:
            raise ValueError("At least one temporal feature is required.")
        if pre_state.shape[1] != self.state_channels:
            raise ValueError(
                f"Expected encoder state with {self.state_channels} channels, received {pre_state.shape[1]}."
            )
        state = self.state_projection(pre_state)
        outputs = []
        for feature in features:
            if feature.shape[1] != self.output_channels:
                raise ValueError(
                    f"Expected decoder feature with {self.output_channels} channels, received {feature.shape[1]}."
                )
            state_before_update = state
            recurrent_output, state = self.GRU(feature, state)
            merged = torch.cat([recurrent_output, state_before_update], dim=1)
            merged = F.interpolate(
                merged, scale_factor=2, mode="bilinear", align_corners=False
            )
            outputs.append(merged)
        return torch.stack(outputs, dim=0)


class DBlock(nn.Module):
    def __init__(
        self,
        input_channels: int = 12,
        output_channels: int = 12,
        conv_type: str = "standard",
        first_relu: bool = True,
        keep_same_output: bool = False,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        conv_layer = get_conv_layer(conv_type)
        if conv_type == "3d":
            self.pooling = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        else:
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv_1x1 = spectral_norm(
            conv_layer(input_channels, output_channels, kernel_size=1)
        )
        self.first_conv_3x3 = spectral_norm(
            conv_layer(input_channels, output_channels, kernel_size=3, padding=1)
        )
        self.last_conv_3x3 = spectral_norm(
            conv_layer(output_channels, output_channels, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.conv_1x1(x) if self.input_channels != self.output_channels else x
        if not self.keep_same_output:
            shortcut = self.pooling(shortcut)
        residual = self.relu(x) if self.first_relu else x
        residual = self.first_conv_3x3(residual)
        residual = self.relu(residual)
        residual = self.last_conv_3x3(residual)
        if not self.keep_same_output:
            residual = self.pooling(residual)
        return shortcut + residual


class attblock(nn.Module):
    def __init__(self, channels: int = 512):
        super().__init__()
        self.att = AttentionLayer(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.att(x)
