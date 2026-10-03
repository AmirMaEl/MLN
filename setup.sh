#!/usr/bin/env bash
# Set up everything: Python environment + dependencies, and all model weights.
#
#   bash setup.sh                  conda env "mln" (Python 3.12), dependencies, all weights
#   bash setup.sh --no-env         install into the currently active Python instead
#   bash setup.sh --deps-only      only the Python environment / dependencies
#   bash setup.sh --weights-only   only the weights
#   bash setup.sh --no-hart        skip VARIN's HART weights (~8 GB)
#
# Environment variables: ENV_NAME (default mln), WEIGHTS (default ./weights),
# HF_HOME (Hugging Face cache, where the Switti and CLIP weights go).
#
# Weights (~62 GB in total):
#   weights/infinity_2b_reg.pth, weights/infinity_vae_d32_reg.pth, weights/flan-t5-xl/   Infinity-2B (AREdit, BitResEdit)
#   weights/hart/hart-0.7b-1024px/, weights/hart/Qwen2-VL-1.5B-Instruct/                 HART-0.7B (VARIN)
#   Hugging Face cache: Switti, Switti-1024, VQVAE-Switti, two CLIP text encoders        Switti (MLN)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS="${WEIGHTS:-$ROOT/weights}"
ENV_NAME="${ENV_NAME:-mln}"
MAKE_ENV=1; DEPS=1; GET_WEIGHTS=1; HART=1
for arg in "$@"; do
  case "$arg" in
    --no-env) MAKE_ENV=0 ;;
    --deps-only) GET_WEIGHTS=0 ;;
    --weights-only) DEPS=0; MAKE_ENV=0 ;;
    --no-hart) HART=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done
step() { printf '\n==> %s\n' "$*"; }
# conda create/install; if conda's libmamba solver is broken (a common plugin version
# mismatch), retry with the built-in classic solver.
conda_do() { conda "$@" || { echo "conda failed; retrying with --solver=classic" >&2; conda "$@" --solver=classic; }; }

# ---- 1. Python environment -------------------------------------------------------
if [ "$MAKE_ENV" = 1 ]; then
  command -v conda >/dev/null || { echo "conda not found: install Miniconda, or rerun with --no-env" >&2; exit 1; }
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    step "creating conda env '$ENV_NAME' (Python 3.12)"
    conda_do create -y -n "$ENV_NAME" python=3.12
  fi
  conda activate "$ENV_NAME"
fi
PY="$(command -v python)"
echo "Python: $PY"

# ---- 2. dependencies ---------------------------------------------------------------
if [ "$DEPS" = 1 ]; then
  step "PyTorch 2.5.1 (CUDA 12.1)"
  "$PY" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
  step "requirements.txt"
  "$PY" -m pip install -r "$ROOT/requirements.txt"
  has_nvcc() { "$PY" -c "import os, sys; from torch.utils.cpp_extension import CUDA_HOME; sys.exit(not (CUDA_HOME and os.path.exists(os.path.join(CUDA_HOME, 'bin', 'nvcc'))))"; }
  if ! has_nvcc; then
    if [ "$MAKE_ENV" = 1 ]; then
      step "CUDA compiler (nvcc 12.1) for HART's RMSNorm kernel"
      conda_do install -y -c nvidia cuda-nvcc=12.1 cuda-cudart-dev=12.1
    else
      echo "WARNING: nvcc not found; VARIN (HART) needs it to build its RMSNorm kernel on first use." >&2
    fi
  fi
  step "flash-attn (required by Infinity-2B's attention)"
  # --no-cache-dir: flash-attn fetches a prebuilt wheel and moves it into pip's cache,
  # which fails when the cache and the temp dir are on different disks.
  "$PY" -m pip install flash-attn==2.8.3.post1 --no-build-isolation --no-cache-dir
fi

# ---- 3. weights ------------------------------------------------------------------
if [ "$GET_WEIGHTS" = 1 ]; then
  step "weights -> $WEIGHTS (existing files are kept)"
  WEIGHTS="$WEIGHTS" HART="$HART" "$PY" - <<'EOF'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

W = Path(os.environ["WEIGHTS"])
W.mkdir(parents=True, exist_ok=True)

def file(repo, name, dst):
    dst = W / dst
    if dst.exists():
        print("  ok", dst); return
    print("  downloading", repo, name)
    src = Path(hf_hub_download(repo, name, local_dir=W))
    if src != dst:
        src.rename(dst)

def folder(repo, dst, patterns=None):
    """dst=None: Hugging Face cache (models loaded by name, e.g. Switti)."""
    if dst is not None and (W / dst).exists() and any((W / dst).iterdir()):
        print("  ok", W / dst); return
    print("  fetching", repo, "(cached files are reused)")
    snapshot_download(repo, local_dir=None if dst is None else W / dst, allow_patterns=patterns)

# Infinity-2B + its VAE + T5 text encoder (AREdit, BitResEdit)
file("FoundationVision/Infinity", "infinity_2b_reg.pth", "infinity_2b_reg.pth")
file("FoundationVision/Infinity", "infinity_vae_d32reg.pth", "infinity_vae_d32_reg.pth")
folder("google/flan-t5-xl", "flan-t5-xl", ["*.json", "*.safetensors", "spiece.model"])
# Switti 512/1024, its VQ-VAE and the two CLIP text encoders (MLN) -> Hugging Face cache
for repo in ("yresearch/Switti", "yresearch/Switti-1024", "yresearch/VQVAE-Switti"):
    folder(repo, None)
folder("openai/clip-vit-large-patch14", None, ["*.json", "*.txt", "model.safetensors"])
folder("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", None, ["*.json", "*.txt", "pytorch_model-*.bin"])
# HART-0.7B + its Qwen2-VL text encoder (VARIN)
if os.environ["HART"] == "1":
    folder("mit-han-lab/hart-0.7b-1024px", "hart/hart-0.7b-1024px", ["llm/config.json", "llm/ema_model.bin", "tokenizer/*"])
    folder("mit-han-lab/Qwen2-VL-1.5B-Instruct", "hart/Qwen2-VL-1.5B-Instruct")
EOF
fi

# ---- 4. check --------------------------------------------------------------------
if [ "$DEPS" = 1 ]; then
  step "checking imports (and building HART's RMSNorm kernel once)"
  (cd "$ROOT" && "$PY" -c "import methods.mln, methods.aredit, methods.bitresedit, methods.varin; print('methods ok')")
  if [ "$HART" = 1 ]; then
    (cd "$ROOT" && "$PY" -c "import hart.modules.networks.basic_hart; print('HART kernel ok')")
  fi
fi
step "done. Try:  python edit.py --method mln --input <image> --source '...' --target '...'   or   python app.py"
