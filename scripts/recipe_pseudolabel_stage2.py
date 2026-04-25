"""Stage-2 pseudo-label retrain: labeler = recipe + recipe_pseudolabel 50/50 blend.

Compared to stage-1 (`recipe_pseudolabel.py`), the only thing that changes
is the labeler used to gate test rows: instead of recipe_full_te alone
(LB 0.97939), we use the 50/50 log-blend of recipe + stage-1 pseudo
(LB 0.97998). Stronger labeler → higher pseudo purity → expected lift on
rare-class boundary rows.

Outputs:
  scripts/artifacts/oof_recipe_pseudolabel_stage2.npy
  scripts/artifacts/test_recipe_pseudolabel_stage2.npy
  scripts/artifacts/recipe_pseudolabel_stage2_results.json
  submissions/submission_recipe_pseudolabel_stage2.csv
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_full_te import load_and_engineer, TARGET, IDX2CLS  # noqa: E402
from recipe_pseudolabel import build_pseudo_subset, run_cv  # noqa: E402

SEED = 42
N_FOLDS = 5
TAU = 0.98
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def softmax_log_blend(*test_arrays: np.ndarray) -> np.ndarray:
    """Equal-weight geometric mean of probability arrays, normalized per-row."""
    eps = 1e-9
    w = 1.0 / len(test_arrays)
    log_p = sum(w * np.log(np.clip(t, eps, 1.0)) for t in test_arrays)
    log_p -= log_p.max(1, keepdims=True)
    p = np.exp(log_p)
    return p / p.sum(1, keepdims=True)


def main() -> None:
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    recipe_bias = np.array(recipe_res["log_bias"])

    test_recipe = np.load(ART / "test_recipe_full_te.npy")
    test_pseudo = np.load(ART / "test_recipe_pseudolabel.npy")
    log(f"loaded labeler legs: recipe={test_recipe.shape}  "
        f"pseudo_s1={test_pseudo.shape}")

    # 2-way labeler: 50/50 log-blend (matches submission_recipe_greedy_recipe_pseudolabel
    # at LB 0.97998). Use recipe's tuned bias since the 2-way blend's tuned
    # bias matches recipe's to within rounding.
    labeler_test_probs = softmax_log_blend(test_recipe, test_pseudo)
    log(f"2-way blend bias = {recipe_bias.round(4).tolist()}")

    keep_mask, pseudo_labels = build_pseudo_subset(
        labeler_test_probs, recipe_bias, TAU
    )
    log(f"τ={TAU}  keep_rate={keep_mask.mean():.4f}  "
        f"({keep_mask.sum()}/{len(keep_mask)} rows)")
    log(f"  pseudo label dist = "
        f"{np.bincount(pseudo_labels[keep_mask], minlength=3).tolist()}")

    train, test, info, test_ids = load_and_engineer()

    if SMOKE:
        log("SMOKE: synthesising a pseudo subset for the 10k smoke test")
        rng = np.random.default_rng(SEED)
        pseudo_test_idx = rng.choice(len(test), size=min(6000, len(test)),
                                     replace=False)
        pseudo_test_labels = rng.choice(3, size=len(pseudo_test_idx),
                                        p=[0.587, 0.380, 0.033])
    else:
        pseudo_test_idx = np.where(keep_mask)[0]
        pseudo_test_labels = pseudo_labels[keep_mask].astype(np.int64)

    result = run_cv(train, test, info, pseudo_test_idx, pseudo_test_labels)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y.astype(np.int32), prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / "oof_recipe_pseudolabel_stage2.npy", result["oof"])
    np.save(ART / "test_recipe_pseudolabel_stage2.npy", result["test"])
    log(f"wrote {ART}/oof_recipe_pseudolabel_stage2.npy + test_recipe_pseudolabel_stage2.npy")

    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in pred_idx],
    })
    sub_path = SUB / "submission_recipe_pseudolabel_stage2.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, tau=TAU,
        labeler="recipe_full_te + recipe_pseudolabel (50/50 log-blend)",
        labeler_bias=recipe_bias.tolist(),
        pseudo_n=int(len(pseudo_test_idx)),
        pseudo_label_dist=[int(x) for x in np.bincount(
            pseudo_test_labels, minlength=3)],
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        baseline_stage1_tuned=json.loads(
            (ART / "recipe_pseudolabel_results.json").read_text()
        )["tuned_log_bias_bal_acc"],
    )
    summary["delta_vs_stage1"] = (
        tuned - summary["baseline_stage1_tuned"]
    )
    res_path = ART / "recipe_pseudolabel_stage2_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}  Δ vs stage-1 = {summary['delta_vs_stage1']:+.5f}")


if __name__ == "__main__":
    main()
