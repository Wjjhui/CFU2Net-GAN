import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorAdversarialLoss(nn.Module):
    def forward(self, spatial_fake_scores: torch.Tensor, temporal_fake_scores: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(1.0 - spatial_fake_scores).mean() + \
               F.leaky_relu(1.0 - temporal_fake_scores).mean()


class Generator_loss_skillful(nn.Module):
    def __init__(
        self,
        Normalized: bool = False,
        weight_power: float = 0.5,
        lambda_pix: float = 1.0,
        lambda_mse: float = 1.0,
    ):
        super().__init__()
        if Normalized:
            raise ValueError("Inputs must already be normalized to [0, 1] as specified in the Method.")
        self.Normalized = False
        self.weight_power = weight_power
        self.lambda_pix = lambda_pix
        self.lambda_mse = lambda_mse

    def components(
        self, org_img: torch.Tensor, pre_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if org_img.shape != pre_img.shape:
            raise ValueError("Observed and predicted sequences must have identical shapes.")
        weights = torch.sigmoid(org_img).pow(self.weight_power)
        pixel_loss = torch.mean(torch.abs(org_img - pre_img) * weights)
        mse_loss = F.mse_loss(pre_img, org_img)
        return pixel_loss, mse_loss

    def forward(
        self, org_img: torch.Tensor, pre_img: torch.Tensor, loss_dis: torch.Tensor
    ) -> torch.Tensor:
        pixel_loss, mse_loss = self.components(org_img, pre_img)
        return loss_dis + self.lambda_pix * pixel_loss + self.lambda_mse * mse_loss


class DiscriminatorLoss_hinge(nn.Module):
    def forward(self, scores: torch.Tensor, org: bool) -> torch.Tensor:
        if org:
            return F.relu(1.0 - scores).mean()
        return F.relu(1.0 + scores).mean()
