import torch
import torch.nn.functional as F 
import torch.nn as nn
from typing import Tuple, List


class AdaIN:
    """
    Adaptive Instance Normalization as proposed in
    'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.
    """

    def _compute_mean_std(
        self, feats: torch.Tensor, eps=1e-8, infer=False
    ) -> torch.Tensor:
        return compute_mean_std(feats, eps, infer)

    def __call__(
        self,
        content_feats: torch.Tensor,
        style_feats: torch.Tensor,
        infer: bool = False,
    ) -> torch.Tensor:
        """
        __call__ Adaptive Instance Normalization as proposaed in
        'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.

        Args:
            content_feats (torch.Tensor): Content features
            style_feats (torch.Tensor): Style Features

        Returns:
            torch.Tensor: [description]
        """
        c_mean, c_std = self._compute_mean_std(content_feats, infer=infer)
        s_mean, s_std = self._compute_mean_std(style_feats, infer=infer)

        normalized = (s_std * (content_feats - c_mean) / c_std) + s_mean

        return normalized


def compute_mean_std(
    feats: torch.Tensor, eps=1e-8, infer=False
) -> torch.Tensor:
    assert (
        len(feats.shape) == 4
    ), "feature map should be 4-dimensional of the form N,C,H,W!"
    #  * Doing this to support ONNX.js inference.
    if infer:
        n = 1
        c = 512  # * fixed for vgg19
    else:
        n, c, _, _ = feats.shape

    feats = feats.view(n, c, -1)
    mean = torch.mean(feats, dim=-1).view(n, c, 1, 1)
    std = torch.std(feats, dim=-1).view(n, c, 1, 1) + eps

    return mean, std


class VggEncoder(nn.Module):
    def __init__(self, pretrained=True, requires_grad=False) -> None:
        super().__init__()
        vgg = models.vgg19(pretrained=pretrained).features
        # * block1: conv1_1, relu1_1,
        self.block1 = vgg[:2]
        # * block2: conv1_2, relu1_2, conv2_1, relu2_1
        self.block2 = vgg[2:7]
        # * block3: conv2_2, relu2_2, conv3_1, relu3_1
        self.block3 = vgg[7:12]
        # * block4
        self.block4 = vgg[12:21]

        self.__set_grad(requires_grad)

    def __set_grad(self, requires_grad: bool):
        for p in self.parameters():
            p.requires_grad = requires_grad

    def forward(self, x, return_last=True):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)

        return f4 if return_last else (f1, f2, f3, f4)
