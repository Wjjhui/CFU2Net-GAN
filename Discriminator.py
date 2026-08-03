import torch
import torch.nn as nn
from torch.nn import PixelUnshuffle
from torch.nn.utils.parametrizations import spectral_norm

from Blocks import DBlock


class Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        self.D_1 = DBlock(4, 48, conv_type="3d", first_relu=False)
        self.D_2 = DBlock(48, 96, conv_type="3d", first_relu=False)
        self.end_d = nn.Sequential(
            DBlock(96, 192),
            DBlock(192, 384),
            DBlock(384, 768),
            DBlock(768, 768, keep_same_output=True),
        )
        self.relu = nn.ReLU()
        self.end = spectral_norm(nn.Linear(768, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError("Temporal discriminator input must have shape [B, T, 1, H, W].")
        batch, time_steps = x.shape[:2]
        x = self.downsample(x)
        x = self.space2depth(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = self.D_1(x)
        x = self.D_2(x)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch * time_steps, x.shape[2], x.shape[3], x.shape[4])
        x = self.end_d(x)
        x = torch.sum(self.relu(x), dim=(2, 3))
        x = x.reshape(batch, time_steps, 768).sum(dim=1)
        return self.end(x)


class Spatial(nn.Module):
    def __init__(self):
        super().__init__()
        self.downSample = nn.AvgPool2d(2)
        self.s2d = PixelUnshuffle(downscale_factor=2)
        self.d = nn.Sequential(
            DBlock(4, 48),
            DBlock(48, 96),
            DBlock(96, 192),
            DBlock(192, 384),
            DBlock(384, 384, keep_same_output=True),
            DBlock(384, 768),
        )
        self.end = spectral_norm(nn.Linear(768, 1))
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError("Spatial discriminator input must have shape [B, T, 1, H, W].")
        outputs = []
        for frame_index in range(x.shape[1]):
            frame = x[:, frame_index]
            frame = self.downSample(frame)
            frame = self.s2d(frame)
            frame = self.d(frame)
            frame = torch.sum(self.relu(frame), dim=(2, 3))
            outputs.append(self.end(frame))
        return torch.stack(outputs, dim=1)
