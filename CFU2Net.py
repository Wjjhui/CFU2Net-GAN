from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from fontTools.ttLib.tables.TupleVariation import PRIVATE_POINT_NUMBERS
from torch.nn import PixelUnshuffle
from torch.nn.utils.parametrizations import spectral_norm
from Blocks import attblock
from Attention import AttentionLayer
from Blocks import My_GRU, GBlock, UpsampleGBlock
from LCEM import LCEM, SelfCorrelationComputation
from spade import SPADE

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()

        padding = kernel_size // 2 if dilation == 1 else dilation
        self.conv = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DownConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.down_flag = flag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_flag:
            x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

        return self.LeakyReLU(self.conv(x))


class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1, flag: bool = True):
        super().__init__(in_ch, out_ch, kernel_size, dilation)
        self.up_flag = flag
        # 添加一个新的卷积层
        self.conv_after_skip = ConvBNReLU(out_ch, out_ch, kernel_size=3, dilation=1)
        # 添加注意力机制
        self.attention = AttentionLayer(input_channels=out_ch, output_channels=out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.up_flag:
            x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = self.LeakyReLU(self.conv(torch.cat([x1, x2], dim=1)))
        # 在跳跃连接后应用新的卷积层
        x = self.conv_after_skip(x)
        # 应用注意力机制
        x = self.attention(x)
        return x



class RSU(nn.Module):
    def __init__(self, height: int, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()

        assert height >= 2
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        encode_list = [DownConvBNReLU(out_ch, mid_ch, flag=False)]
        decode_list = [UpConvBNReLU(mid_ch * 2, mid_ch, flag=False)]
        for i in range(height - 2):
            encode_list.append(DownConvBNReLU(mid_ch, mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch * 2, mid_ch if i < height - 3 else out_ch))

        encode_list.append(ConvBNReLU(mid_ch, mid_ch, dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x, x2)

        return x + x_in


class RSU4F(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.encode_modules = nn.ModuleList([ConvBNReLU(out_ch, mid_ch),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch, mid_ch, dilation=8)])

        self.decode_modules = nn.ModuleList([ConvBNReLU(mid_ch * 2, mid_ch, dilation=4),
                                             ConvBNReLU(mid_ch * 2, mid_ch, dilation=2),
                                             ConvBNReLU(mid_ch * 2, out_ch)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.conv_in(x)

        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x = m(x)
            encode_outputs.append(x)

        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(torch.cat([x, x2], dim=1))

        return x + x_in

class U2Net(nn.Module):
    def __init__(self, cfg: dict, out_ch: int = 1, device='cuda:0', frames=6, dmw_channels_after_space2depth=16):
        super().__init__()
        assert "encode" in cfg
        assert "decode" in cfg
        self.encode_num = len(cfg["encode"])
        encode_list = []
        side_list = []
        for c in cfg["encode"]:
            # c: [height, in_ch, mid_ch, out_ch, RSU4F, side]
            assert len(c) == 6
            encode_list.append(RSU(*c[:4]) if c[4] is False else RSU4F(*c[1:4]))
            if c[5] is True:
                side_list.append(nn.Conv2d(c[3], out_ch, kernel_size=3, padding=1))
        self.encode_modules = nn.ModuleList(encode_list)

        # -------- SPADE modules for each encode stage ----------
        # label_nc = dmw_channels after space2depth (we keep dmw channels constant while pooling)
        label_nc = dmw_channels_after_space2depth
        spade_list = []
        for c in cfg["encode"]:
            in_ch_expected = c[1]  # this is the RSU's expected input channels
            spade_list.append(SPADE(norm_nc=in_ch_expected, label_nc=label_nc, hidden_nc=64))
        self.spade_modules = nn.ModuleList(spade_list)
        # ------------------------------------------------------

        self.convGRU6 = My_GRU(input_channels=1024, output_channels=512, kernel_size=3)
        self.convGRU6_ = My_GRU(input_channels=1024, output_channels=512, kernel_size=3)
        self.convGRU5 = My_GRU(input_channels=512, output_channels=256, kernel_size=3)
        self.convGRU4 = My_GRU(input_channels=256, output_channels=128, kernel_size=3)
        self.convGRU3 = My_GRU(input_channels=128, output_channels=64, kernel_size=3)
        self.gru_conv_6_1x1 = spectral_norm(torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)))
        self.g_6 = GBlock(input_channels=32, output_channels=32)
        self.g_6up = UpsampleGBlock(input_channels=32, output_channels=64)

        self.end = nn.Sequential(torch.nn.ReLU(),
                                 spectral_norm(torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))))
        self.maxPool = nn.MaxPool2d(kernel_size=2)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        self.decode5 = RSU4F(in_ch=1024, mid_ch=256, out_ch=512)
        self.decode4 = RSU(height=4, in_ch=1024, mid_ch=256, out_ch=512)
        self.decode3 = RSU(height=4, in_ch=1024, mid_ch=128, out_ch=256)
        self.decode2 = RSU(height=5, in_ch=512, mid_ch=64, out_ch=128)
        self.decode1 = RSU(height=6, in_ch=256, mid_ch=32, out_ch=64)
        self.decode0 = RSU(height=7, in_ch=128, mid_ch=16, out_ch=64)
        self.device = device
        self.att = attblock()
        self.frames = frames
        self.lcem_module = self._make_lcem_layer(planes=[512, 128, 128, 128, 512])

    def _make_lcem_layer(self, planes):
        stride, kernel_size, padding = (1, 1, 1), (5, 5), 2
        layers = list()
        corr_block = SelfCorrelationComputation(kernel_size=kernel_size, padding=padding)
        self_block = LCEM(planes=planes, stride=stride)
        layers.append(corr_block)
        layers.append(self_block)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, dmw: torch.Tensor):
        """
        x: (B, 4, H, W)  -- original input frames
        dmw: (B, 4, H, W) -- dmw auxiliary (same spatial H,W)
        """

        # --- bring both x and dmw through space2depth so channels match encoder expects ---
        x = self.space2depth(x)      # B, 16, H/2, W/2
        dmw = self.space2depth(dmw)  # B, 16, H/2, W/2

        # ensure both are float tensors on same device
        x = x.to(self.device) if not x.device.type == self.device.split(':')[0] else x
        dmw = dmw.to(self.device) if not dmw.device.type == self.device.split(':')[0] else dmw

        encode_outputs = []
        dmw_feats = []  # store dmw features aligned with encode_outputs (post-module)
        for i, m in enumerate(self.encode_modules):
            # apply SPADE modulation before encoder module m
            # spade expects x channels == c[1] (as used when constructing)
   

            # then pass through RSU/RSU4F
            x = m(x)
            encode_outputs.append(x)

            # store the dmw feature corresponding to this level (we store pre-pool dmw)
            dmw_feats.append(dmw)

            # downsample for next stage (if not last)
            if i != self.encode_num - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)
                dmw = F.max_pool2d(dmw, kernel_size=2, stride=2, ceil_mode=True)

        # LCEM and attention
        lcem_x = self.lcem_module(x)
        Z_sp = self.att(lcem_x + x)


        prestate_18list = [Z_sp] * self.frames
        hidden_states = self.convGRU6(prestate_18list, encode_outputs.pop())

        hidden_states = [self.decode5(h4，dmw) for h4 in hidden_states]
        hidden_states = self.convGRU6(hidden_states, encode_outputs.pop())

        hidden_states = [self.decode4(h4) for h4 in hidden_states]
        hidden_states = self.convGRU6(hidden_states, encode_outputs.pop())

        hidden_states = [self.decode3(h4) for h4 in hidden_states]
        hidden_states = self.convGRU5(hidden_states, encode_outputs.pop())

        hidden_states = [self.decode2(h4) for h4 in hidden_states]
        hidden_states = self.convGRU4(hidden_states, encode_outputs.pop())

        hidden_states = [self.decode1(h4) for h4 in hidden_states]
        hidden_states = self.convGRU3(hidden_states, encode_outputs.pop())

        hidden_states = [self.decode0(h4) for h4 in hidden_states]
        hidden_states = [self.end(h4) for h4 in hidden_states]

        # final shape: (B, frames, 1, H, W)
        return torch.stack(hidden_states, dim=1)

def Generator_full(out_ch: int = 1, frames=6, device="cuda:0"):
    cfg = {
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "encode": [[7, 16, 32, 64, False, False],  # En1
                   [6, 64, 32, 128, False, False],  # En2
                   [5, 128, 64, 256, False, False],  # En3
                   [4, 256, 128, 512, False, False],  # En4
                   [4, 512, 256, 512, True, False],  # En5
                   [4, 512, 256, 512, True, True]],  # En6
        # height, in_ch, mid_ch, out_ch, RSU4F, side
        "decode": [[4, 1024, 256, 512, True, True],  # De5
                   [4, 1024, 128, 256, False, True],  # De4
                   [5, 512, 64, 128, False, True],  # De3
                   [6, 256, 32, 64, False, True],  # De2
                   [7, 128, 16, 64, False, True]]  # De1
    }

    return U2Net(cfg, out_ch, device, frames)



