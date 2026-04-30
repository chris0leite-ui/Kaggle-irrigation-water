"""#6 HyperBand on rawashishsin XGB — Kaggle GPU kernel scaffold.

Search space (HyperBand 5 brackets, ~150 effective trials):
  depth ∈ [3, 5]
  lr ∈ [0.02, 0.10]
  reg_alpha ∈ [0.0, 5.0]
  reg_lambda ∈ [0.0, 5.0]
  ORIG_ROW_WEIGHT ∈ [0.3, 1.0]
  smooth ∈ {auto, 1, 5, 10}

Objective: 5-fold StratifiedKFold(seed=42) tuned-bias OOF macro-recall.

Gate (per CLAUDE.md):
  Standalone OOF >= 0.98010 (rawashishsin v3 baseline) + 5e-4
  OR LB >= 0.98109 (rawashishsin v3 baseline)
  OR PCR delta in macro-recall-favorable direction (+High recall)

Cost: ~3h Kaggle T4/P100 GPU (15-20 min per HyperBand trial × 150 trials with
early-stopping pruning).

Smoke first per CLAUDE.md (IS_SMOKE=True): 2-fold × 50k × 100-iter × 5 trials
to validate end-to-end pipeline before production push.

NOTE: This is a SCAFFOLD only. The full HyperBand sweep needs the
rawashishsin FE+training loop inlined here from kernel_rawashishsin_v3.

To execute:
  1. Inline FE pipeline from kernel_rawashishsin_v3/rawashishsin.py
  2. Add HyperBand bracket scheduler around the HP sampler below
  3. SMOKE locally with IS_SMOKE=1
  4. Push as private kernel: kaggle kernels push
  5. Pull artifacts after ~3h: kaggle kernels output ... -p output_n6_hyperband/
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Auto-detect Kaggle vs local
IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    DATA_DIR = Path("/kaggle/input/playground-series-s6e4")
    OUT_DIR = Path("/kaggle/working")
else:
    DATA_DIR = Path("data")
    OUT_DIR = Path("scripts/artifacts")

OUT_DIR.mkdir(parents=True, exist_ok=True)

IS_SMOKE = bool(int(os.getenv("IS_SMOKE", "0")))
N_TRIALS = 8 if IS_SMOKE else 150
N_FOLDS = 2 if IS_SMOKE else 5
TIME_LIMIT_S = 60 * 60 * 3
START = time.time()


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


HP_SPACE = {
    "depth":      [3, 4, 5],
    "lr":         [0.02, 0.04, 0.05, 0.07, 0.10],
    "reg_alpha":  [0.0, 1.0, 2.0, 5.0],
    "reg_lambda": [0.0, 1.0, 2.0, 5.0],
    "orig_w":     [0.3, 0.5, 0.7, 1.0],
    "smooth":     ["auto", 1.0, 5.0, 10.0],
}


def main():
    log(f"n6 HyperBand SCAFFOLD. SMOKE={IS_SMOKE} N_TRIALS={N_TRIALS} N_FOLDS={N_FOLDS}")
    log(f"Time budget: {TIME_LIMIT_S}s")
    log(f"HP search space (random sample 150 trials with 5-bracket HyperBand pruning):")
    for k, v in HP_SPACE.items():
        log(f"  {k}: {v}")
    log("")
    log("This scaffold is incomplete on purpose:")
    log("  1. Inline the FE+training loop from kernel_rawashishsin_v3/rawashishsin.py")
    log("  2. Wrap each fold's XGB training with the HP sampler above")
    log("  3. Apply HyperBand bracket scheduling: s_max=4, eta=3,")
    log("     resource = num_boost_round (R=2000)")
    log("  4. Track per-trial OOF tuned-bias macro-recall + per-class recall")
    log("  5. Output the best HP config's full 5-fold OOF + test predictions")
    log("")
    log("Estimated cost on Kaggle T4: 12-20 min per full-resource trial,")
    log("with HyperBand pruning ~150 effective trials in 3h wall.")

    summary = {
        "is_smoke": IS_SMOKE,
        "n_trials": N_TRIALS,
        "n_folds": N_FOLDS,
        "hp_space": {k: list(v) for k, v in HP_SPACE.items()},
        "scaffold_only": True,
        "production_ready": False,
        "next_steps": [
            "Inline FE+training from kernel_rawashishsin_v3/rawashishsin.py",
            "Add HyperBand bracket scheduling",
            "SMOKE locally; push as private Kaggle kernel",
            "Run ~3h on T4 GPU; pull artifacts and add to natural-cal bank",
        ],
    }
    with open(OUT_DIR / "n6_hyperband_scaffold.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved metadata: {OUT_DIR / 'n6_hyperband_scaffold.json'}")


if __name__ == "__main__":
    main()
