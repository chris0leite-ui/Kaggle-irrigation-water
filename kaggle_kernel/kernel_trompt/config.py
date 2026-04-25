"""Runtime config + paths + SMOKE flag for the Trompt kernel."""
from __future__ import annotations
import os
from pathlib import Path

# IS_SMOKE: top-level override (env vars don't travel through Kaggle
# push). Flip to False AFTER the SMOKE kernel completes successfully.
IS_SMOKE = True
SMOKE = IS_SMOKE or os.environ.get("SMOKE") == "1"

# 5-fold stratified aligned with every saved OOF on main (seed=42).
N_FOLDS = 2 if SMOKE else 5
N_EPOCHS = 2 if SMOKE else 15

# Trompt hyperparams (published kernel defaults).
CHANNELS = 128
NUM_PROMPTS = 128
NUM_LAYERS = 3
BATCH_SIZE = 512

# Safety nets — stricter than the RealMLP kernel because pytorch_frame
# setup time is less well-characterised.
FOLD1_KILL_SEC = 25 * 60 if not SMOKE else 10 * 60
TOTAL_KILL_SEC = 55 * 60 if not SMOKE else 15 * 60

KAGGLE_INPUT = Path("/kaggle/input")
_DEFAULT_OUT = Path("/kaggle/working")
if _DEFAULT_OUT.exists() or _DEFAULT_OUT.parent.exists():
    OUT_DIR = _DEFAULT_OUT
else:
    OUT_DIR = Path(os.environ.get("SMOKE_OUTPUT_DIR", "./trompt_local_out"))
OUT_DIR.mkdir(exist_ok=True, parents=True)
