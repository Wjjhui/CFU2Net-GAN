import torch
import torch.nn as nn
import torch.nn.functional as F


class LCEM(nn.Module):
    def __init__(
        self,
        planes: list[int] | tuple[int, ...] = (512, 128, 128, 128, 512),
        stride: tuple[int, int, int] = (1, 1, 1),
        ksize: int = 3,
        do_padding: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        if tuple(planes) != (512, 128, 128, 128, 512):
            raise ValueError("The Method specifies LCEM channels 512-128-128-128-512.")
        if stride != (1, 1, 1):
            raise ValueError("The Method specifies unit stride in LCEM.")
        if ksize != 3 or do_padding:
            raise ValueError("The Method specifies two unpadded (1, 3, 3) convolutions.")
        self.planes = tuple(planes)
        self.conv1x1_in = nn.Conv2d(512, 128, kernel_size=1, bias=bias)
        self.conv1 = nn.Conv3d(
            128, 128, kernel_size=(1, 3, 3), stride=1, padding=0, bias=bias
        )
        self.conv2 = nn.Conv3d(
            128, 128, kernel_size=(1, 3, 3), stride=1, padding=0, bias=bias
        )
        self.conv1x1_out = nn.Conv2d(128, 512, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 6:
            raise ValueError("LCEM input must have shape [B, C, H, W, 5, 5].")
        batch, channels, height, width, window_h, window_w = x.shape
        if channels != 512 or window_h != 5 or window_w != 5:
            raise ValueError("LCEM expects 512 channels and 5 x 5 local windows.")
        x = x.reshape(batch, channels, height * width, window_h * window_w)
        x = self.conv1x1_in(x)
        x = x.reshape(batch, 128, height * width, window_h, window_w)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(batch, 128, height, width)
        x = self.conv1x1_out(x)
        return F.normalize(x, p=2, dim=1, eps=1e-12)


class SelfCorrelationComputation(nn.Module):
    def __init__(self, kernel_size: tuple[int, int] = (5, 5), padding: int = 2):
        super().__init__()
        if kernel_size != (5, 5) or padding != 2:
            raise ValueError("The Method specifies a 5 x 5 window with two-pixel padding.")
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Bottleneck input must have shape [B, C, H, W].")
        batch, channels, height, width = x.shape
        x = self.relu(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        x = self.unfold(x)
        x = x.reshape(
            batch,
            channels,
            self.kernel_size[0],
            self.kernel_size[1],
            height,
            width,
        )
        return x.permute(0, 1, 4, 5, 2, 3).contiguous()
