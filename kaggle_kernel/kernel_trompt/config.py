"""Runtime config + paths + SMOKE flag for the Trompt kernel."""
from __future__ import annotations
import os
from pathlib import Path

# IS_SMOKE: 2-fold/20k/2-epoch structural check.
# IS_PROBE: 1-fold full-data run for compute-budget validation
# before committing to a 5-fold run. Mutually exclusive with SMOKE.
IS_SMOKE = False
IS_PROBE = True  # fold-1-only at full capacity + full data
SMOKE = IS_SMOKE or os.environ.get("SMOKE") == "1"
PROBE = IS_PROBE or os.environ.get("PROBE") == "1"

# 5-fold StratifiedKFold(seed=42) for OOF alignment with every saved
# OOF on main. PROBE mode runs only fold 1 (via MAX_FOLDS=1 break),
# preserving the val-index alignment for fold-1 Jaccard diagnostics.
N_FOLDS = 2 if SMOKE else 5
MAX_FOLDS = 2 if SMOKE else (1 if PROBE else 5)
N_EPOCHS = 2 if SMOKE else (8 if PROBE else 15)

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
