"""Small pieces vendored so the Switti backbone has no utils/diffusers dependency."""
import numpy as np
import torch
import torch.nn as nn

RESOLUTION_PATCH_NUMS_MAPPING = {
    256: "1_2_3_4_5_6_8_10_13_16",
    512: "1_2_3_4_6_9_13_18_24_32",
    1024: "1_2_3_4_5_7_9_12_16_21_27_36_48_64",
}


class GaussianFourierProjection(nn.Module):
    """diffusers.models.embeddings.GaussianFourierProjection (same parameter names)."""

    def __init__(self, embedding_size=256, scale=1.0, set_W_to_weight=True, log=True, flip_sin_to_cos=False):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
        self.log, self.flip_sin_to_cos = log, flip_sin_to_cos
        if set_W_to_weight:
            del self.weight
            self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)
            self.weight = self.W
            del self.W

    def forward(self, x):
        if self.log:
            x = torch.log(x)
        x_proj = x[:, None] * self.weight[None, :] * 2 * np.pi
        if self.flip_sin_to_cos:
            return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
