"""Kaggle GPU boot: torch pin for P100 + pip install mambular.

Mirrors kernel_trompt boot pattern. Mambular pulls mamba_ssm which has
pure-PyTorch fallback for the selective scan when CUDA toolkit is
unavailable on Kaggle's image.
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


def install_mambular() -> None:
    # mambular: BASF tabular Mamba (sklearn-style API). Has a
    # pure-PyTorch fallback for selective scan (no nvcc required) but
    # the fallback is O(L^2) memory which OOMs the P100 16GB at
    # batch=1024. Install mamba_ssm + causal_conv1d FIRST so mambular
    # picks up the O(L) CUDA kernel path. Pre-built wheels for cu121
    # are available as of late 2024.
    for pkg in ("causal-conv1d", "mamba-ssm", "mambular", "lightning"):
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "--quiet", pkg,
            ])
            print(f"[boot] installed {pkg}", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"[boot] {pkg} install failed ({e}); continuing",
                  flush=True)


def boot() -> None:
    """Re-import + log torch / mambular versions. Pip work runs at
    module body below — by the time anything calls boot(), installs are
    done. Required because downstream model.py has module-level
    `from mambular.models import ...`.
    """
    import torch
    import mambular
    print(f"[boot] torch={torch.__version__} "
          f"cuda_avail={torch.cuda.is_available()}", flush=True)
    print(f"[boot] mambular={mambular.__version__}", flush=True)
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


# === module-level install: must run before any other sibling module's
# imports execute (they happen top-to-bottom in the assembled dist) ===
install_torch_if_pascal()
install_mambular()
