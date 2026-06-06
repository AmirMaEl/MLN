"""Utility functions replacing the private Framework dependency."""

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T


def tensor_to_pil(tensor):
    """Convert a (C, H, W) or (B, C, H, W) tensor in [-1, 1] or [0, 1] to PIL Image(s)."""
    if tensor.dim() == 4:
        return [tensor_to_pil(t) for t in tensor]
    t = tensor.cpu().float()
    if t.min() < 0:
        t = (t + 1) / 2
    t = t.clamp(0, 1)
    return T.ToPILImage()(t)


def save_data_batch(tensor, path, grid=None):
    """Save a batch of images as a grid PNG.

    Args:
        tensor: (B, C, H, W) float tensor in [-1, 1] or [0, 1]
        path: output file path (without extension)
        grid: [nrow, ncol] — only nrow is used by torchvision
    """
    from torchvision.utils import save_image
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    nrow = grid[1] if grid else 8
    out = path if path.endswith('.png') else path + '.png'
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    save_image(tensor.float().cpu().clamp(-1, 1), out, nrow=nrow, normalize=True, value_range=(-1, 1))


def save_tensor_as_image(tensor, path):
    """Save a single (C, H, W) tensor as a PNG."""
    img = tensor_to_pil(tensor)
    if not isinstance(img, Image.Image):
        img = img[0]
    img.save(path if path.endswith('.png') else path + '.png')


def plot_data_batch(tensor, path, grid=None):
    save_data_batch(tensor, path, grid=grid)
