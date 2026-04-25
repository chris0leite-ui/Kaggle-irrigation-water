"""Kaggle GPU boot: torch pin for P100 + pip install pytorch_frame.

Kaggle pre-installed torch doesn't have sm_60 kernel images. Install
cu121 build BEFORE pytorch_frame imports torch.
"""
from __future__ import annotations
import subprocess
import sys


def gpu_compute_caps() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
        return [x.strip() for x in out if x.strip()]
    except Exception as e:
        print(f"[boot] nvidia-smi error: {e}", flush=True)
        return []


def install_torch_if_pascal() -> None:
    arches = gpu_compute_caps()
    print(f"[boot] gpu compute_cap = {arches}", flush=True)
    if not any(a in ("6.0", "6.1") for a in arches):
        return
    print("[boot] sm_60/61 detected - pinning torch+torchvision cu121",
          flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--upgrade", "--force-reinstall", "--no-deps",
        "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])


def install_pytorch_frame() -> None:
    # pytorch_frame pulls torch_geometric deps; pyg-lib is sm_60-compatible
    # for torch 2.5.x. Use --no-deps to avoid re-installing torch, then
    # install the runtime deps separately.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "pytorch_frame",
    ])


def boot() -> None:
    install_torch_if_pascal()
    install_pytorch_frame()
    import torch
    import torch_frame
    print(f"[boot] torch={torch.__version__} "
          f"cuda_avail={torch.cuda.is_available()}", flush=True)
    print(f"[boot] torch_frame={torch_frame.__version__}", flush=True)
    try:
        info = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip()
        print(f"[boot] GPU info: {info}", flush=True)
    except Exception:
        pass
