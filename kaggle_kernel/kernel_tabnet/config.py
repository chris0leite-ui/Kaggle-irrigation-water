"""Runtime config + paths + SMOKE flag for the TabNet kernel."""
from __future__ import annotations
import os
from pathlib import Path

# IS_SMOKE: 2-fold/20k/2-epoch structural check (push first to validate
# pip install + GPU init + model fit pipeline end-to-end).
# IS_PROBE: 1-fold full-data run for compute-budget validation
# before committing to 5-fold. Mutually exclusive with SMOKE.
IS_SMOKE = True   # SMOKE FIRST per CLAUDE.md rule
IS_PROBE = False
SMOKE = IS_SMOKE or os.environ.get("SMOKE") == "1"
PROBE = IS_PROBE or os.environ.get("PROBE") == "1"

# 5-fold StratifiedKFold(seed=42) for OOF alignment.
N_FOLDS = 2 if SMOKE else 5
MAX_FOLDS = 2 if SMOKE else (1 if PROBE else 5)
N_EPOCHS = 2 if SMOKE else (10 if PROBE else 20)

# TabNet hyperparams (paper defaults, tuned for stability on 504k rows).
NUM_LAYERS = 8                 # decision steps
SPLIT_FEAT_CHANNELS = 64
SPLIT_ATTN_CHANNELS = 64
GAMMA = 1.5                    # relaxation parameter for sparse selection
BATCH_SIZE = 1024
LR = 2e-3

# Safety nets — mirror trompt kernel's stricter caps because TabNet's
# sequential decision steps may be slower per batch than Trompt's
# parallelized prompts.
FOLD1_KILL_SEC = 25 * 60 if not SMOKE else 10 * 60
TOTAL_KILL_SEC = 55 * 60 if not SMOKE else 15 * 60

KAGGLE_INPUT = Path("/kaggle/input")
_DEFAULT_OUT = Path("/kaggle/working")
if _DEFAULT_OUT.exists() or _DEFAULT_OUT.parent.exists():
    OUT_DIR = _DEFAULT_OUT
else:
    OUT_DIR = Path(os.environ.get("SMOKE_OUTPUT_DIR", "./tabnet_local_out"))
OUT_DIR.mkdir(exist_ok=True, parents=True)
