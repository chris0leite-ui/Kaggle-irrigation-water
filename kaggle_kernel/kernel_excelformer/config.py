"""Runtime config + paths + SMOKE flag for the ExcelFormer kernel."""
from __future__ import annotations
import os
from pathlib import Path

# IS_SMOKE: 2-fold/20k/2-epoch structural check
# IS_PROBE: 1-fold full-data run for compute-budget validation
IS_SMOKE = True   # SMOKE FIRST per CLAUDE.md rule
IS_PROBE = False
SMOKE = IS_SMOKE or os.environ.get("SMOKE") == "1"
PROBE = IS_PROBE or os.environ.get("PROBE") == "1"

N_FOLDS = 2 if SMOKE else 5
MAX_FOLDS = 2 if SMOKE else (1 if PROBE else 5)
N_EPOCHS = 3 if SMOKE else (15 if PROBE else 25)

# ExcelFormer hyperparams (paper defaults).
IN_CHANNELS = 32
NUM_LAYERS = 5
NUM_HEADS = 32
DIAM_DROPOUT = 0.2
AIUM_DROPOUT = 0.2
RESIDUAL_DROPOUT = 0.2
MIXUP_MODE = "hidden"  # paper's key contribution
BETA = 0.5
BATCH_SIZE = 512
LR = 3e-4

# Safety nets — stricter than TabNet's because ExcelFormer's attention
# is more compute-heavy per batch (DIAM+AIUM dual-attention).
FOLD1_KILL_SEC = 25 * 60 if not SMOKE else 10 * 60
TOTAL_KILL_SEC = 55 * 60 if not SMOKE else 15 * 60

KAGGLE_INPUT = Path("/kaggle/input")
_DEFAULT_OUT = Path("/kaggle/working")
if _DEFAULT_OUT.exists() or _DEFAULT_OUT.parent.exists():
    OUT_DIR = _DEFAULT_OUT
else:
    OUT_DIR = Path(os.environ.get("SMOKE_OUTPUT_DIR",
                                  "./excelformer_local_out"))
OUT_DIR.mkdir(exist_ok=True, parents=True)
