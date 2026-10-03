"""Settings of the modified HART used by VARIN (in the original fork: hyper_params.yaml + parse_params.py)."""
import os
from pathlib import Path
from types import SimpleNamespace

WEIGHTS = Path(os.environ.get("HART_WEIGHTS", Path(__file__).resolve().parent.parent / "weights" / "hart"))
PARAMS = SimpleNamespace(
    data=SimpleNamespace(ratio=1.0),                                  # height / width of the token maps
    pretrained_models=SimpleNamespace(
        hart=str(WEIGHTS / "hart-0.7b-1024px" / "llm"),
        patch_nums=[1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64],  # 1024 px
    ),
)
