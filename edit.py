#!/usr/bin/env python3
"""Edit one image.

  python edit.py --method mln --input cat.jpg --source "a photo of a cat" --target "a photo of a dog"
  python edit.py --method bitresedit --reso 1024 --input cat.jpg --mask mask.png \\
      --source "a photo of a cat" --target "a photo of a dog" --out dog.png

Methods: mln (ours), aredit, varin, varin_tau9 (mask-free; varin* at 1024 px only) and
aredit_mask, bitresedit, bitresedit_lock (need --mask, white = edit).
"""
import argparse

import cv2
import numpy as np
import torch

from methods import METHODS, load_method


def read_square(path, reso, interp=None, flags=cv2.IMREAD_COLOR):
    """Center-crop to a square and resize to reso."""
    img = cv2.imread(path, flags)
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    if interp is None:
        interp = cv2.INTER_CUBIC if reso > s else cv2.INTER_AREA
    return cv2.resize(img, (reso, reso), interpolation=interp)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--method", default="mln", choices=list(METHODS))
    ap.add_argument("--reso", type=int, choices=[512, 1024], default=512)
    ap.add_argument("--input", required=True)
    ap.add_argument("--source", required=True, help="prompt describing the input image")
    ap.add_argument("--target", required=True, help="prompt describing the edited image")
    ap.add_argument("--mask", help="edit mask image, white = edit (only for mask methods)")
    ap.add_argument("--out", default="edited.png")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    kwargs = {}
    if METHODS[a.method][2]:
        if not a.mask:
            ap.error(f"--method {a.method} needs --mask")
        kwargs["mask"] = (read_square(a.mask, a.reso, cv2.INTER_NEAREST, cv2.IMREAD_GRAYSCALE) > 127).astype(np.float32)
    rgb = cv2.cvtColor(read_square(a.input, a.reso), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(rgb).float().permute(2, 0, 1)[None] / 127.5 - 1
    out = load_method(a.method, a.reso).edit(image, a.source, a.target, seed=a.seed, **kwargs)
    out = (out[0].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(a.out, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print("saved", a.out)


if __name__ == "__main__":
    main()
