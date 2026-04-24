"""Blend-gate analysis for A1 RealMLP OOF/test outputs from Kaggle kernel.

Runs AFTER `kaggle kernels output chrisleitescha/irrigation-realmlp-pytabkit`
downloads oof_realmlp.npy + test_realmlp.npy into scripts/artifacts/.

Reports:
  - standalone tuned OOF + errors
  - Jaccard vs recipe_full_te + LB-best 2-way + 3-way
  - fixed-bias α-sweep vs all three anchors
  - emits submission CSV only if peak Δ ≥ +0.0002 at LB-best anchor

Blend heuristic (from CLAUDE.md):
  Jaccard < 0.80 AND errs ≤ anchor → component passes gate
  Jaccard 0.80–0.90 with few-errors → conditional pass (capped lift)
  Jaccard ≥ 0.90 → skip (redundant)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, log_blend, tune_log_bias

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _errmask(probs: np.ndarray, y: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1) != y


def _eval(probs: np.ndarray, y: np.ndarray, bias: np.ndarray):
    pred = (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1)
    cc = np.bincount(y, minlength=3)
    return fast_bal_acc(y, pred, class_counts=cc), int((pred != y).sum())


def main() -> None:
    if not (ART / "oof_realmlp.npy").exists():
        raise SystemExit(
            "oof_realmlp.npy not found — run `kaggle kernels output "
            "chrisleitescha/irrigation-realmlp-pytabkit -p scripts/artifacts/` "
            "first, or copy from the kernel output dir."
        )

    train = pd.read_csv("data/train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    log("loading realmlp + anchor OOFs")
    realmlp_oof = np.load(ART / "oof_realmlp.npy")
    realmlp_test = np.load(ART / "test_realmlp.npy")
    recipe_oof = np.load(ART / "oof_recipe_full_te.npy")
    recipe_test = np.load(ART / "test_recipe_full_te.npy")
    pseudo_s1_oof = np.load(ART / "oof_recipe_pseudolabel.npy")
    pseudo_s1_test = np.load(ART / "test_recipe_pseudolabel.npy")
    pseudo_s7_oof = np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")
    pseudo_s7_test = np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")

    res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias_recipe = np.array(res["log_bias"], dtype=np.float64)

    log(f"recipe anchor bias = {bias_recipe.round(4).tolist()}")

    # Standalone diagnostics at recipe's bias.
    log("=== standalone @ recipe bias ===")
    ba_recipe, n_recipe = _eval(recipe_oof, y, bias_recipe)
    ba_realmlp, n_realmlp = _eval(realmlp_oof, y, bias_recipe)
    log(f"  recipe      bal={ba_recipe:.5f}  errs={n_recipe}")
    log(f"  realmlp     bal={ba_realmlp:.5f}  errs={n_realmlp}")

    # RealMLP tuned standalone.
    prior = np.bincount(y, minlength=3) / len(y)
    bias_realmlp, ba_realmlp_tuned = tune_log_bias(
        realmlp_oof, y, prior
    )
    log(f"  realmlp tuned bal={ba_realmlp_tuned:.5f}  "
        f"bias={bias_realmlp.round(4).tolist()}")

    # Error Jaccards.
    errs_recipe = _errmask(recipe_oof, y, bias_recipe)
    errs_realmlp = _errmask(realmlp_oof, y, bias_recipe)
    inter = int((errs_recipe & errs_realmlp).sum())
    union = int((errs_recipe | errs_realmlp).sum())
    jacc_recipe = inter / max(union, 1)
    log(f"  jaccard(recipe, realmlp) = {jacc_recipe:.4f}")

    # LB-best 2-way anchor (recipe × pseudo_s1 @50/50).
    lb2_oof = log_blend([recipe_oof, pseudo_s1_oof], np.array([0.5, 0.5]))
    lb2_test = log_blend([recipe_test, pseudo_s1_test], np.array([0.5, 0.5]))
    ba_lb2, n_lb2 = _eval(lb2_oof, y, bias_recipe)
    errs_lb2 = _errmask(lb2_oof, y, bias_recipe)
    inter2 = int((errs_lb2 & errs_realmlp).sum())
    union2 = int((errs_lb2 | errs_realmlp).sum())
    jacc_lb2 = inter2 / max(union2, 1)
    log(f"  LB-best 2-way bal={ba_lb2:.5f}  errs={n_lb2}  "
        f"jaccard(LB2, realmlp)={jacc_lb2:.4f}")

    # LB-best 3-way.
    lb3_oof = log_blend([recipe_oof, pseudo_s1_oof, pseudo_s7_oof],
                        np.array([0.25, 0.35, 0.40]))
    lb3_test = log_blend([recipe_test, pseudo_s1_test, pseudo_s7_test],
                         np.array([0.25, 0.35, 0.40]))
    ba_lb3, n_lb3 = _eval(lb3_oof, y, bias_recipe)
    errs_lb3 = _errmask(lb3_oof, y, bias_recipe)
    inter3 = int((errs_lb3 & errs_realmlp).sum())
    union3 = int((errs_lb3 | errs_realmlp).sum())
    jacc_lb3 = inter3 / max(union3, 1)
    log(f"  LB-best 3-way bal={ba_lb3:.5f}  errs={n_lb3}  "
        f"jaccard(LB3, realmlp)={jacc_lb3:.4f}")

    # Blend gate decision.
    log("=== blend gate ===")
    if jacc_lb3 >= 0.90:
        verdict = "abort (Jaccard >= 0.90 vs LB-best 3-way — redundant)"
    elif jacc_lb3 >= 0.85:
        verdict = "warn (Jaccard 0.85-0.90 — blend lift capped ~+0.00015)"
    elif n_realmlp > int(1.15 * n_lb3):
        verdict = "magnitude-trap risk (realmlp has >15% more errors than LB3)"
    else:
        verdict = "pass — proceed with fixed-bias sweep"
    log(f"  verdict: {verdict}")

    # Fixed-bias sweeps.
    log("=== fixed-bias log-blend α sweep ===")
    alphas = np.arange(0.0, 0.55, 0.025)
    rows = []
    for target_name, target_oof, target_test in [
        ("recipe",   recipe_oof,   recipe_test),
        ("LB2",      lb2_oof,      lb2_test),
        ("LB3",      lb3_oof,      lb3_test),
    ]:
        best = (0.0, 0.0, None, None)
        for a in alphas:
            if a == 0.0:
                blend_oof = target_oof
                blend_test = target_test
            else:
                blend_oof = log_blend([target_oof, realmlp_oof],
                                      np.array([1.0 - a, a]))
                blend_test = log_blend([target_test, realmlp_test],
                                       np.array([1.0 - a, a]))
            ba, _ = _eval(blend_oof, y, bias_recipe)
            if ba > best[0]:
                best = (ba, float(a), blend_oof, blend_test)
        base = {"recipe": ba_recipe, "LB2": ba_lb2, "LB3": ba_lb3}[target_name]
        delta = best[0] - base
        log(f"  vs {target_name}: peak α={best[1]:.3f}  "
            f"bal={best[0]:.5f}  Δ={delta:+.5f}")
        rows.append({
            "target": target_name, "peak_alpha": best[1],
            "peak_bal": best[0], "base_bal": float(base),
            "delta": float(delta),
        })

        # Emit submission only for the LB-best 3-way if Δ ≥ +0.0002.
        if target_name == "LB3" and delta >= 2e-4:
            log(f"  -> Δ {delta:+.5f} clears +0.0002 gate; "
                f"emitting submission")
            pred = (np.log(np.clip(best[3], 1e-9, 1.0)) +
                    bias_recipe).argmax(1)
            sample = pd.read_csv("data/sample_submission.csv")
            sub = sample.copy()
            sub[TARGET] = [CLASSES[i] for i in pred]
            sub_path = SUB / f"submission_realmlp_blend_a{best[1]:.2f}.csv"
            sub.to_csv(sub_path, index=False)
            log(f"  wrote {sub_path}")

    out = {
        "standalone": {
            "bal_at_recipe_bias": ba_realmlp,
            "tuned_bal": ba_realmlp_tuned,
            "errs_at_recipe_bias": n_realmlp,
            "tuned_bias": bias_realmlp.tolist(),
        },
        "jaccards": {
            "vs_recipe": jacc_recipe,
            "vs_LB2": jacc_lb2,
            "vs_LB3": jacc_lb3,
        },
        "blend_gate_verdict": verdict,
        "sweeps": rows,
    }
    (ART / "blend_realmlp_results.json").write_text(json.dumps(out, indent=2))
    log(f"wrote {ART / 'blend_realmlp_results.json'}")


if __name__ == "__main__":
    main()
