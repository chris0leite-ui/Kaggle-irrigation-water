"""L2 SupCon-NCM — config + constants."""
from __future__ import annotations

import os

SEED = 42
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

IS_SMOKE = os.environ.get("IS_SMOKE", "False").lower() in ("1", "true")
EPOCHS = int(os.environ.get("EPOCHS", "3" if IS_SMOKE else "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4096))
EMBED_DIM = int(os.environ.get("EMBED_DIM", 32))
N_FOLDS = int(os.environ.get("N_FOLDS", "2" if IS_SMOKE else "5"))

# Hard wall-time cap per CLAUDE.md GPU 1h rule.
TOTAL_KILL_SEC = 55 * 60
