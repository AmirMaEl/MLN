# MLN — Masked Logit Nudging

Official implementation of  
**"Prompt-Guided Image Editing with Masked Logit Nudging in Visual Autoregressive Models"**  
[[arXiv 2604.14591]](https://arxiv.org/abs/2604.14591)

---

## How it works

MLN edits a real image by nudging the token-level logits of a frozen
[Switti](https://github.com/yandex-research/switti) Visual Autoregressive (VAR)
model. Given a **source prompt** describing the input image and a **target
prompt** describing the desired edit, MLN:

1. Decodes the input image into VQ-VAE tokens.
2. At each VAR scale, computes an attention-based edit mask that localises
   regions relevant to the changed concept.
3. Nudges the predicted logits toward the target distribution inside the mask
   while leaving unmasked regions unchanged.

The result is a faithful edit that respects the unedited parts of the scene —
no fine-tuning or gradient-based inversion required.

---

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate gradio tqdm scikit-learn opencv-python
```

Models are downloaded automatically from Hugging Face on first use:

| Resolution | Model ID |
|---|---|
| 512 px | `yresearch/Switti` |
| 1024 px | `yresearch/Switti-1024` |

---

## Quick start

### CLI

```bash
# dog → cat at 512 px
python edit.py \
    --input dog.jpg \
    --source_prompt "a photo of a dog" \
    --target_prompt "a photo of a cat" \
    --output cat_512.png \
    --resolution 512

# same edit at 1024 px
python edit.py \
    --input dog.jpg \
    --source_prompt "a photo of a dog" \
    --target_prompt "a photo of a cat" \
    --output cat_1024.png \
    --resolution 1024
```

Full argument reference:

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Path to the input image |
| `--source_prompt` | required | Text describing the input image |
| `--target_prompt` | required | Text describing the desired edit |
| `--output` | required | Path to save the edited image |
| `--resolution` | `512` | Output resolution — `512` or `1024` |
| `--seed` | `42` | Random seed |
| `--cfg` | `8.0` | Classifier-free guidance scale |
| `--mask_quantile` | `0.7` | Attention mask threshold (0–1) |
| `--nudge_alphas` | auto | Comma-separated nudge strengths per VAR scale, e.g. `6,6,6,6,6,6,6,6,6,6` |
| `--gt_fix_scales` | `None` | Leading scales seeded from GT tokens (0 = all scales free) |
| `--device` | auto | `cuda` or `cpu` |

### Gradio web UI

```bash
python app.py
```

Opens a browser demo where you can upload an image, enter prompts, and adjust
editing parameters interactively.

---

## Repository structure

```
MLN/
├── edit.py              # CLI entry point
├── app.py               # Gradio web demo
├── pipeline_512.py      # MLN inference pipeline (512 px)
├── pipeline_1024.py     # MLN inference pipeline (1024 px)
├── mln_utils.py         # Image / tensor helpers
├── dist.py              # Distributed training utilities
├── models/
│   ├── switti.py        # Switti transformer
│   ├── basic_switti.py  # Attention / FFN blocks
│   ├── vqvae.py         # VQ-VAE architecture
│   ├── basic_vae.py     # VAE encoder / decoder
│   ├── quant.py         # Vector quantization codebook
│   ├── clip.py          # Frozen CLIP text encoder
│   ├── helpers.py       # Top-k / top-p / Gumbel sampling
│   ├── ste.py           # Straight-Through Estimator
│   ├── rope.py          # Rotary Position Embeddings
│   └── __init__.py      # Model factory
└── utils/
    ├── arg_util.py       # Training argument parser
    ├── misc.py           # Logging and distributed helpers
    ├── fsdp.py           # FSDP training wrapper
    ├── lr_control.py     # Learning rate scheduling
    ├── amp_sc.py         # Mixed precision + gradient scaling
    ├── data.py           # COCO dataset loader
    ├── data_sampler.py   # Infinite distributed sampler
    ├── fid_score_in_memory.py  # FID metric
    └── inception.py      # InceptionV3 feature extractor
```

---

## Citation

```bibtex
@inproceedings{el2026prompt,
  title={Prompt-Guided Image Editing with Masked Logit Nudging in Visual Autoregressive Models},
  author={El-Ghoussani, Amir and H{\"o}lle, Marc and Carneiro, Gustavo and Belagiannis, Vasileios},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4810--4820},
  year={2026}
}
```
