"""Runtime config + paths + SMOKE/PROBE flags for the Mamba kernel.

CLAUDE.md GPU rule: 1h hard cap. Wall budget:
  - SMOKE  (~5 min) : 2-fold x 20k x 2 epochs, structural check
  - PROBE  (~50 min): 1-fold x full data x 8 epochs, fold-1 OOF for
                       Jaccard-vs-LB-best gate
Total kill set 55 min so both runs fit individual kernel pushes.
"""
from __future__ import annotations
import os
from pathlib import Path

IS_SMOKE = True   # SMOKE first per CLAUDE.md "SMOKE-test before long runs"
IS_PROBE = False  # flip to True for the production probe push
SMOKE = IS_SMOKE or os.environ.get("SMOKE") == "1"
PROBE = IS_PROBE or os.environ.get("PROBE") == "1"

# 5-fold StratifiedKFold(seed=42) preserves alignment with every saved
# OOF on main. PROBE mode runs only fold 1 to keep wall under the cap.
N_FOLDS = 2 if SMOKE else 5
MAX_FOLDS = 2 if SMOKE else (1 if PROBE else 5)
N_EPOCHS = 2 if SMOKE else (8 if PROBE else 15)

# Mamba hyperparams. P100 has 16 GB; mambular's pure-PyTorch
# selective_scan fallback is O(L^2 * d_model * d_state) memory.
# Even with mamba_ssm CUDA kernel installed, conservative batch helps
# reliability across the SMOKE/PROBE/full-fold path.
D_MODEL = 64
N_LAYERS = 4
D_STATE = 16
D_CONV = 4
EXPAND = 2
DROPOUT = 0.1
BATCH_SIZE = 256 if SMOKE else 512
LR = 1e-3
WEIGHT_DECAY = 1e-5

# Safety nets — match the Trompt kernel's bounds.
FOLD1_KILL_SEC = 25 * 60 if not SMOKE else 10 * 60
TOTAL_KILL_SEC = 55 * 60 if not SMOKE else 15 * 60

KAGGLE_INPUT = Path("/kaggle/input")
_DEFAULT_OUT = Path("/kaggle/working")
if _DEFAULT_OUT.exists() or _DEFAULT_OUT.parent.exists():
    OUT_DIR = _DEFAULT_OUT
else:
    OUT_DIR = Path(os.environ.get("SMOKE_OUTPUT_DIR", "./mamba_local_out"))
OUT_DIR.mkdir(exist_ok=True, parents=True)
