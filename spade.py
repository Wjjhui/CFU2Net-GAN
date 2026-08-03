import torch
import torch.nn as nn
import torch.nn.functional as F


class SPADE(nn.Module):
    """
    SPADE: Spatially-Adaptive (DE)Normalization
    适用于 U2Net / UNet / ResNet 中的任意特征层。
    输入:
        x   : 要被归一化的特征 (B, C, H, W)
        cond: 条件特征（例如 dmw）(B, Cc, Hc, Wc)
    输出:
        out : 调制后的特征 (B, C, H, W)
    """

    def __init__(self, norm_nc, label_nc, hidden_nc=64):
        """
        Parameters:
            norm_nc  : 输入特征 x 的通道数
            label_nc : 条件特征 cond 的通道数
        """
        super().__init__()

        # 1. 参数自由归一化：InstanceNorm / BatchNorm 都可
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # 2. SPADE 网络：对 cond 进行卷积，输出 γ 和 β
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden_nc, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.mlp_gamma = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta  = nn.Conv2d(hidden_nc, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, cond):
        """
        x:    (B, C, H, W)
        cond: (B, Cc, Hc, Wc)  -- 尺寸可不同，会自动 resize
        """

        # 归一化

        # normalized = self.param_free_norm(x)


        # 将 cond resize 到 x 的空间尺寸
        if cond.size(2) != x.size(2) or cond.size(3) != x.size(3):
            cond = F.interpolate(cond, size=x.shape[2:], mode='nearest')

        # 提取调制参数 γ 和 β
        actv = self.mlp_shared(cond)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Spatially Adaptive Normalization
        # out = normalized * (1 + gamma) + beta
        out = x * (1 + gamma) + beta

        return out
