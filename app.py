"""
Gradio demo for MLN image editing.

Edits an input image guided by source → target prompts using
Masked Logit Nudging (MLN) in the Switti Visual Autoregressive model.
"""

import os
import time
import gradio as gr
import torch
import numpy as np
from PIL import Image


# ─── pipeline cache ───────────────────────────────────────────────────────────
_pipe = None
_pipe_resolution = None


def _get_pipeline(resolution: int, device: str):
    global _pipe, _pipe_resolution
    if _pipe is None or _pipe_resolution != resolution:
        from edit import build_pipeline
        _pipe = build_pipeline(resolution, device)
        _pipe_resolution = resolution
    return _pipe


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pil_to_tensor(img: Image.Image, resolution: int) -> torch.Tensor:
    img = img.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)
    return (t * 2 - 1).unsqueeze(0)              # (1, 3, H, W) in [-1, 1]


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0).cpu().float()
    if t.min() < 0:
        t = (t + 1) / 2
    t = t.clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _side_by_side(left: Image.Image, right: Image.Image, gap: int = 8) -> Image.Image:
    w = left.width + gap + right.width
    h = max(left.height, right.height)
    canvas = Image.new("RGB", (w, h), (30, 30, 30))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


# ─── inference ────────────────────────────────────────────────────────────────

def run_edit(
    source_image,
    source_prompt: str,
    target_prompt: str,
    resolution: int,
    gt_fix_scales: int,
    cfg: float,
    seed: int,
    mask_image,
):
    if source_image is None:
        return None, "Please upload a source image."
    if not source_prompt.strip():
        return None, "Please enter a source prompt."
    if not target_prompt.strip():
        return None, "Please enter a target prompt."

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not isinstance(source_image, Image.Image):
        source_image = Image.fromarray(source_image)
    image_tensor = _pil_to_tensor(source_image, resolution).to(device)

    mask = None
    if mask_image is not None:
        if not isinstance(mask_image, Image.Image):
            mask_image = Image.fromarray(mask_image)
        mask_np = np.array(mask_image.resize((resolution, resolution)).convert("L"))
        mask_bin = (mask_np > 127).astype(np.float32)
        # shape expected by invert_stepwise: (1, 1, H, W)
        mask = torch.from_numpy(mask_bin).unsqueeze(0).unsqueeze(0).to(device)

    try:
        pipe = _get_pipeline(resolution, device)
    except Exception as exc:
        return None, f"Failed to load pipeline: {exc}"

    try:
        result = pipe.invert_stepwise(
            image_B3HW=image_tensor,
            prompt=source_prompt,
            eprompt=target_prompt,
            seed=int(seed),
            cfg=cfg,
            gt_fix_scales=int(gt_fix_scales),
            mask=mask,
            visualize=False,
        )
    except Exception as exc:
        return None, f"Error during editing: {exc}"

    # fhats: (N_scales, B, 3, H, W) — last entry is the final output
    edited_pil = _tensor_to_pil(result["fhats"][-1])

    os.makedirs("output", exist_ok=True)
    out_path = f"output/edit_{int(time.time())}.png"
    edited_pil.save(out_path)

    src_resized = source_image.convert("RGB").resize((resolution, resolution), Image.LANCZOS)
    comparison = _side_by_side(src_resized, edited_pil)

    return comparison, f"Saved to {out_path}"


# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MLN Image Editing") as demo:
    gr.Markdown(
        "## MLN Image Editing\n"
        "Edit images using **Masked Logit Nudging** (MLN) in the Switti VAR model."
    )

    with gr.Row():
        # ── inputs ────────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            src_img = gr.Image(label="Source Image", type="pil")
            src_prompt = gr.Textbox(
                label="Source Prompt",
                placeholder="a photo of a dog",
            )
            tgt_prompt = gr.Textbox(
                label="Target Prompt",
                placeholder="a photo of a cat",
            )

            with gr.Row():
                resolution = gr.Radio(
                    choices=[512, 1024],
                    value=512,
                    label="Resolution",
                )
                seed = gr.Number(value=42, precision=0, label="Seed")

            gt_fix = gr.Slider(
                minimum=0,
                maximum=9,
                step=1,
                value=2,
                label="GT Fix Scales",
                info=(
                    "Number of leading VAR scales seeded directly from the "
                    "ground-truth image tokens before the transformer runs. "
                    "0 = transformer handles all scales freely; "
                    "higher values preserve more coarse structure."
                ),
            )
            cfg_scale = gr.Slider(
                minimum=1.0,
                maximum=20.0,
                step=0.5,
                value=8.0,
                label="CFG Scale",
                info="Classifier-free guidance strength.",
            )
            mask_img = gr.Image(
                label="Binary Mask (optional)",
                type="pil",
            )
            gr.Markdown(
                "_Upload a black-and-white image as a mask. "
                "White regions are treated as the edit area._"
            )
            run_btn = gr.Button("Edit", variant="primary")

        # ── outputs ───────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            output_img = gr.Image(label="Source  |  Edited")
            status_box = gr.Textbox(label="Status", interactive=False)

    run_btn.click(
        fn=run_edit,
        inputs=[src_img, src_prompt, tgt_prompt, resolution, gt_fix, cfg_scale, seed, mask_img],
        outputs=[output_img, status_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
