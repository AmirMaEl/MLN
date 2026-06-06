"""
MLN image editing CLI.

Edits an input image guided by a source and target text prompt using
Masked Logit Nudging (MLN) in the Switti Visual Autoregressive model.

Example — dog to cat at 512px:
    python edit.py --input dog.jpg --source_prompt "a photo of a dog" \
        --target_prompt "a photo of a cat" --output cat_512.png --resolution 512

Example — same edit at 1024px:
    python edit.py --input dog.jpg --source_prompt "a photo of a dog" \
        --target_prompt "a photo of a cat" --output cat_1024.png --resolution 1024
"""

import argparse
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image


def load_image(path: str, resolution: int) -> torch.Tensor:
    """Load an image file and return a (1, 3, H, W) tensor in [-1, 1]."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (resolution, resolution))
    t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    t = t * 2 - 1
    return t.unsqueeze(0)


def save_output(tensor: torch.Tensor, path: str):
    """Save a (1, 3, H, W) tensor in [-1, 1] or [0, 1] as a PNG."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    t = tensor.squeeze(0).cpu().float()
    if t.min() < 0:
        t = (t + 1) / 2
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)
    print(f"Saved edited image to: {path}")


def build_pipeline(resolution: int, device: str):
    """Load the appropriate SwittiPipeline for the given resolution."""
    if resolution == 512:
        from pipeline_512 import SwittiPipeline
        model_id = "yresearch/Switti"
    elif resolution == 1024:
        from pipeline_1024 import SwittiPipeline
        model_id = "yresearch/Switti-1024"
    else:
        raise ValueError(f"Unsupported resolution: {resolution}. Use 512 or 1024.")

    print(f"Loading {model_id} for {resolution}px editing …")
    pipe = SwittiPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device=device,
        reso=resolution,
    )
    return pipe


def edit(
    pipe,
    image: torch.Tensor,
    source_prompt: str,
    target_prompt: str,
    seed: int,
    cfg: float,
    mask_quantile: float,
    nudge_alphas=None,
    gt_fix_scales=None,
):
    """Run MLN editing and return the edited image tensor in [-1, 1]."""
    image = image.to(pipe.device)
    kwargs = dict(
        image_B3HW=image,
        prompt=source_prompt,
        eprompt=target_prompt,
        seed=seed,
        cfg=cfg,
        quant=mask_quantile,
        visualize=False,
    )
    if nudge_alphas is not None:
        kwargs["nudge_alphas"] = nudge_alphas
    if gt_fix_scales is not None:
        kwargs["gt_fix_scales"] = gt_fix_scales
    result = pipe.invert_stepwise(**kwargs)
    # result['fhats'] is (N_scales, B, 3, H, W) — take the last scale
    return result['fhats'][-1]   # (B, 3, H, W)


def main():
    parser = argparse.ArgumentParser(
        description="MLN image editing: edit an image guided by source→target prompts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default="output/output.png", help="Path to save edited image")
    parser.add_argument(
        "--source_prompt",
        default="a photo of a dog",
        help="Text prompt describing the input image",
    )
    parser.add_argument(
        "--target_prompt",
        default="a photo of a cat",
        help="Text prompt describing the desired edit",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[512, 1024],
        default=512,
        help="Edit resolution: 512 or 1024 pixels",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--cfg", type=float, default=8.0, help="Classifier-free guidance scale")
    parser.add_argument(
        "--nudge_alphas",
        type=str,
        default=None,
        help="Comma-separated nudge strengths per VAR scale, e.g. '6,6,6,6,6,6,6,6,6,6'. "
             "Defaults to 6 for all scales when omitted.",
    )
    parser.add_argument(
        "--gt_fix_scales",
        type=int,
        default=None,
        help="Number of leading scales to seed from GT tokens (0 = run all scales through the transformer).",
    )
    parser.add_argument(
        "--mask_quantile",
        type=float,
        default=0.7,
        help="Quantile threshold for the attention-based edit mask (0–1)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    args = parser.parse_args()

    nudge_alphas = (
        [float(x) for x in args.nudge_alphas.split(",")]
        if args.nudge_alphas is not None
        else None
    )

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    image = load_image(args.input, args.resolution)
    pipe = build_pipeline(args.resolution, args.device)

    print(f"Editing: '{args.source_prompt}' → '{args.target_prompt}'")
    edited = edit(
        pipe=pipe,
        image=image,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        seed=args.seed,
        cfg=args.cfg,
        mask_quantile=args.mask_quantile,
        nudge_alphas=nudge_alphas,
        gt_fix_scales=args.gt_fix_scales,
    )

    save_output(edited, args.output)


if __name__ == "__main__":
    main()
