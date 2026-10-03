"""Gradio demo: pick backbone, method and resolution, upload an image, edit it.

  python app.py            # http://localhost:7860   (--share for a public link)

Methods marked (mask) edit only the region you paint with the brush.
One model is kept on the GPU at a time; switching method or resolution reloads.
"""
import argparse
import gc
import time

import cv2
import gradio as gr
import numpy as np
import torch

from methods import METHODS, load_method

LABEL = {v[0]: k for k, v in METHODS.items()}
BACKBONES = {}
for name, (label, backbone, _, _) in METHODS.items():
    BACKBONES.setdefault(backbone, []).append(label)
_loaded = {"key": None, "model": None}


def get_model(name, reso):
    if _loaded["key"] != (name, reso):
        _loaded["model"] = None
        gc.collect()
        torch.cuda.empty_cache()
        _loaded.update(key=(name, reso), model=load_method(name, reso))
    return _loaded["model"]


def square(img, reso, interp=None):
    """Center-crop to a square and resize to reso."""
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (reso, reso), interpolation=interp or (cv2.INTER_CUBIC if reso > s else cv2.INTER_AREA))


def run(editor, label, reso, source_prompt, target_prompt, seed):
    if editor is None or editor.get("background") is None:
        raise gr.Error("Upload an image first.")
    name, reso = LABEL[label], int(reso)
    _, _, needs_mask, resos = METHODS[name]
    if reso not in resos:
        raise gr.Error(f"{label} runs at {resos[0]} px only.")
    image = torch.from_numpy(square(editor["background"][..., :3], reso)).float().permute(2, 0, 1)[None] / 127.5 - 1
    kwargs = {}
    if needs_mask:
        painted = [l[..., 3] > 0 for l in editor.get("layers") or [] if l is not None and l.shape[-1] == 4]
        if not np.any(painted):
            raise gr.Error("This method needs an edit mask: paint the region to edit with the brush.")
        kwargs["mask"] = square(np.any(painted, 0).astype(np.float32), reso, cv2.INTER_NEAREST)
    model = get_model(name, reso)
    t = time.time()
    out = model.edit(image, source_prompt, target_prompt, seed=int(seed), **kwargs)
    img = (out[0].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return img, f"{label} @ {reso} px: {time.time() - t:.1f} s"


def on_backbone(backbone):
    labels = BACKBONES[backbone]
    return gr.update(choices=labels, value=labels[0]), on_method(labels[0])


def on_method(label):
    resos = METHODS[LABEL[label]][3]
    return gr.update(choices=[str(r) for r in resos], value=str(resos[0]))


with gr.Blocks(title="VAR image editing") as demo:
    gr.Markdown("## Training-free image editing with next-scale AR models\n"
                "Upload an image, describe it (source) and the edit (target). "
                "Methods marked *(mask)* edit only the region you paint with the brush.")
    with gr.Row():
        with gr.Column():
            editor = gr.ImageEditor(label="Input image", type="numpy", height=512,
                                    brush=gr.Brush(colors=["#ff0000"], default_size=40))
            backbone = gr.Dropdown(list(BACKBONES), value="Switti", label="Backbone")
            method = gr.Dropdown(BACKBONES["Switti"], value="MLN (ours)", label="Method")
            reso = gr.Radio(["512", "1024"], value="512", label="Resolution (px)")
            src = gr.Textbox(label="Source prompt (describes the input)", value="a cat sitting on a wooden table")
            tgt = gr.Textbox(label="Target prompt (describes the edit)", value="a dog sitting on a wooden table")
            seed = gr.Number(42, label="Seed", precision=0)
            go = gr.Button("Edit", variant="primary")
        with gr.Column():
            result = gr.Image(label="Edited image")
            info = gr.Markdown()
    backbone.change(on_backbone, backbone, [method, reso])
    method.change(on_method, method, reso)
    go.click(run, [editor, method, reso, src, tgt, seed], [result, info], api_name="edit")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    demo.queue(max_size=8).launch(server_name="0.0.0.0", server_port=a.port, share=a.share, show_error=True)
