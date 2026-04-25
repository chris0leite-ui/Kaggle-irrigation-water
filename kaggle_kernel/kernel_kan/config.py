"""Runtime config + paths + SMOKE/PROBE flags for the KAN kernel.

CLAUDE.md GPU rule: 1h hard cap. Wall budget plan:
  - SMOKE  (~3 min) : 2-fold x 20k x 2 epochs, structural check
  - PROBE  (~30 min): 1-fold x full data x 12 epochs, fold-1 OOF for
                       Jaccard-vs-LB-best-4-stack gate
Total kill set 50 min so the run fits within the 1h cap with overhead.

efficient-kan is pure-PyTorch, ~10x faster than original pykan.
On a P100 with batch=2048 and ~70-dim input, expect ~5-8 sec/epoch on
504k rows for a [in, 96, 64, 3] architecture.

KAN inductive bias rationale: edges carry learnable spline activations
instead of fixed nonlinearities at nodes. The 2026-04-21 DGP-residuals
EDA established the host label generator is a smooth NN function of
non-rule continuous features; KAN's per-edge B-spline parameterisation
is uniquely suited to fit smooth, non-axis-aligned boundaries. None of
the 14 prior NN nulls (MLP / FT-T / TabPFN / RealMLP / Trompt / Mamba)
have this inductive bias.
"""
from __future__ import annotations
import os
from pathlib import Path

IS_SMOKE = False
IS_PROBE = True

SMOKE = IS_SMOKE or os.environ.get("SMOKE") == "1"
PROBE = IS_PROBE or os.environ.get("PROBE") == "1"

N_FOLDS = 2 if SMOKE else 5
MAX_FOLDS = 2 if SMOKE else (1 if PROBE else 5)
N_EPOCHS = 2 if SMOKE else (12 if PROBE else 20)

# KAN architecture (input dim auto-inferred from one-hot + nums).
HIDDEN = [128, 64] if SMOKE else [192, 96, 48]
GRID_SIZE = 5      # spline grid intervals per edge
SPLINE_ORDER = 3   # cubic B-splines
GRID_RANGE = (-1.0, 1.0)
DROPOUT = 0.1

BATCH_SIZE = 1024 if SMOKE else 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.0

# Safety nets — match the Trompt/Mamba kernel bounds.
FOLD1_KILL_SEC = 25 * 60 if not SMOKE else 8 * 60
TOTAL_KILL_SEC = 50 * 60 if not SMOKE else 12 * 60

KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
_DEFAULT_OUT = Path("/kaggle/working")
if _DEFAULT_OUT.exists() or _DEFAULT_OUT.parent.exists():
    OUT_DIR = _DEFAULT_OUT
else:
    OUT_DIR = Path(os.environ.get("SMOKE_OUTPUT_DIR", "./kan_local_out"))
OUT_DIR.mkdir(exist_ok=True, parents=True)
