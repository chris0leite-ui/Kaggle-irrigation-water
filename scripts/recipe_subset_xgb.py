"""Recipe-subset XGB variants (N1 from main's next-steps menu).

Each variant drops one feature block from the full recipe to force a
structurally different XGB signal path. Controlled via env var
`RECIPE_SUBSET`:

  no_digits : drop 66 digit cols + their OTE derivatives
  no_combos : drop 28 cat-pair combo cols + their OTE derivatives
  no_ote    : skip OrderedTE entirely; pure-numeric XGB
  no_orig   : drop ~48 ORIG mean/std numeric cols

`SMOKE=1` shrinks to 20k train / 10k test, 2 folds — ~2 min on CPU.

Outputs per variant to `scripts/artifacts/` and `submissions/`:
  oof_recipe_{variant}.npy
  test_recipe_{variant}.npy
  recipe_{variant}_results.json
  submission_recipe_{variant}.csv
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_subset_fe import load_and_engineer  # noqa: E402
from recipe_subset_cv import run_cv  # noqa: E402

SEED = 42
TARGET = "Irrigation_Need"
IDX2CLS = {0: "Low", 1: "Medium", 2: "High"}

VALID = ("no_digits", "no_combos", "no_ote", "no_orig")
VARIANT = os.environ.get("RECIPE_SUBSET", "no_ote")
assert VARIANT in VALID, f"RECIPE_SUBSET must be one of {VALID}, got {VARIANT}"
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(exist_ok=True, parents=True)
SUB.mkdir(exist_ok=True, parents=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    log(f"variant={VARIANT}  smoke={SMOKE}  n_folds={N_FOLDS}")
    train, test, info, test_ids = load_and_engineer(VARIANT, SMOKE)
    result = run_cv(train, test, info, n_folds=N_FOLDS, smoke=SMOKE)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    tag = VARIANT
    oof_path = ART / f"oof_recipe_{tag}.npy"
    test_path = ART / f"test_recipe_{tag}.npy"
    np.save(oof_path, result["oof"])
    np.save(test_path, result["test"])
    log(f"wrote {oof_path} + {test_path}")

    # Submission with tuned log-bias
    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in pred_idx]})
    sub_path = SUB / f"submission_recipe_{tag}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}  "
        f"pred_dist={dict(sub[TARGET].value_counts())}")

    summary = dict(
        variant=VARIANT, seed=SEED, n_folds=N_FOLDS, smoke=SMOKE,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=result["n_features"],
        feature_group_sizes={
            k: len(v) for k, v in info.items()
            if isinstance(v, list) and k != "te_cols"
        },
        te_col_count=len(info["te_cols"]),
    )
    with open(ART / f"recipe_{tag}_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote scripts/artifacts/recipe_{tag}_results.json")


if __name__ == "__main__":
    main()
