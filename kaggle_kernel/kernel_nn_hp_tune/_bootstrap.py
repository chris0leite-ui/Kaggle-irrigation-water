"""GPU shim — must run BEFORE any `import torch`.

Kaggle occasionally allocates P100 (sm_60) which the stock
torch 2.10.0+cu128 doesn't support. Detect it and reinstall
torch 2.5.1+cu121 (last version that still ships sm_60 kernels).
"""
from __future__ import annotations
import subprocess
import sys


def _gpu_arch() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader"],
            text=True, timeout=10).strip().splitlines()
        return [x.strip() for x in out if x.strip()]
    except Exception as e:
        return [f"err:{e}"]


def _maybe_downgrade_torch() -> None:
    arches = _gpu_arch()
    print(f"[boot] gpu compute_cap = {arches}", flush=True)
    if not any(a in ("6.0", "6.1") for a in arches):
        return
    print("[boot] sm_60/61 detected — reinstalling torch 2.5.1 cu121",
          flush=True)
    # Drop --no-deps: a bare torch reinstall leaves _dynamo internals
    # inconsistent with the pre-existing deps, breaking AdamW import.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--upgrade", "--force-reinstall",
        "torch==2.5.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])


_maybe_downgrade_torch()
