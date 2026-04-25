"""Blend-gate analysis for Mamba PROBE OOF/test outputs.

Runs AFTER `kaggle kernels output chrisleitescha/irrigation-mambular-ssm`
downloads oof_mamba_probe.npy + test_mamba_probe.npy into scripts/artifacts/.

PROBE mode fills only fold 1 of the OOF (alignment with all other OOFs
preserved via StratifiedKFold(seed=42)). Diagnostic computed on
filled rows only.

Gate (from CLAUDE.md S1 Mamba decision rule):
  fold-1 Jaccard < 0.75 vs LB-best 4-stack AND errs ≤ 9572 → PROCEED full 5-fold
  Jaccard 0.75-0.85: cap blend lift expectation at ~+0.00015 → don't push
  Jaccard >= 0.85: redundant → close lever
  errs > 1.05 * anchor: magnitude trap → close lever

LB-best 4-stack composition (LB 0.98094, OOF 0.98084):
  step 0: lb3 = log_blend(recipe, pseudo_s1, pseudo_s7; 0.25/0.35/0.40)
  step 1: stack3 = log_blend(lb3, realmlp; 0.80/0.20)
  step 2: stack3a = log_blend(stack3, xgb_nonrule_iso; 0.925/0.075)
  step 3: stack4 = log_blend(stack3a, xgb_metastack_iso; 0.70/0.30)
  bias  = recipe's tuned bias = [1.4324, 1.4689, 3.4008]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, log_blend, tune_log_bias

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _eval(probs, y, bias, mask=None):
    pred = (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1)
    if mask is not None:
        pred = pred[mask]
        y_ = y[mask]
    else:
        y_ = y
    cc = np.bincount(y_, minlength=3)
    return fast_bal_acc(y_, pred, class_counts=cc), int((pred != y_).sum())


def _errmask(probs, y, bias):
    return (np.log(np.clip(probs, 1e-9, 1.0)) + bias).argmax(1) != y


def _per_class_iso(oof, y):
    """Per-class isotonic calibration on full OOF (matches tier1b)."""
    out = np.zeros_like(oof)
    for k in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(oof[:, k], (y == k).astype(np.float64))
        out[:, k] = ir.transform(oof[:, k])
    out = out / out.sum(axis=1, keepdims=True).clip(1e-9)
    return out


def main() -> None:
    suffix = sys.argv[1] if len(sys.argv) > 1 else "_probe"

    oof_path = ART / f"oof_mamba{suffix}.npy"
    test_path = ART / f"test_mamba{suffix}.npy"
    if not oof_path.exists():
        raise SystemExit(
            f"{oof_path} not found - run "
            "`kaggle kernels output chrisleitescha/irrigation-mambular-ssm "
            "-p scripts/artifacts/` first."
        )

    log(f"loading mamba{suffix} OOF / test")
    mamba_oof = np.load(oof_path)
    mamba_test = np.load(test_path)
    nz = mamba_oof.sum(axis=1) > 0
    log(f"  filled rows = {int(nz.sum()):,}/{len(mamba_oof):,} "
        f"({'PROBE 1-fold' if nz.sum() < len(mamba_oof) * 0.5 else 'full'})")

    train = pd.read_csv("data/train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    # LB-best 4-stack reconstruction.
    log("reconstructing LB-best 4-stack from components")
    recipe_oof = np.load(ART / "oof_recipe_full_te.npy")
    recipe_test = np.load(ART / "test_recipe_full_te.npy")
    pseudo_s1_oof = np.load(ART / "oof_recipe_pseudolabel.npy")
    pseudo_s1_test = np.load(ART / "test_recipe_pseudolabel.npy")
    pseudo_s7_oof = np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")
    pseudo_s7_test = np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")
    realmlp_oof = np.load(ART / "oof_realmlp.npy")
    realmlp_test = np.load(ART / "test_realmlp.npy")
    nonrule_oof = np.load(ART / "oof_xgb_nonrule.npy")
    nonrule_test = np.load(ART / "test_xgb_nonrule.npy")
    meta_oof = np.load(ART / "oof_xgb_metastack.npy")
    meta_test = np.load(ART / "test_xgb_metastack.npy")

    bias_recipe = np.array(json.loads(
        (ART / "recipe_full_te_results.json").read_text()
    )["log_bias"], dtype=np.float64)
    log(f"  bias_recipe = {bias_recipe.round(4).tolist()}")

    nonrule_iso_oof = _per_class_iso(nonrule_oof, y)
    nonrule_iso_test = _per_class_iso(nonrule_test, y)
    meta_iso_oof = _per_class_iso(meta_oof, y)
    meta_iso_test = _per_class_iso(meta_test, y)

    lb3_oof = log_blend([recipe_oof, pseudo_s1_oof, pseudo_s7_oof],
                        np.array([0.25, 0.35, 0.40]))
    lb3_test = log_blend([recipe_test, pseudo_s1_test, pseudo_s7_test],
                         np.array([0.25, 0.35, 0.40]))
    stack3_oof = log_blend([lb3_oof, realmlp_oof], np.array([0.80, 0.20]))
    stack3_test = log_blend([lb3_test, realmlp_test], np.array([0.80, 0.20]))
    stack3a_oof = log_blend([stack3_oof, nonrule_iso_oof],
                            np.array([0.925, 0.075]))
    stack3a_test = log_blend([stack3_test, nonrule_iso_test],
                             np.array([0.925, 0.075]))
    stack4_oof = log_blend([stack3a_oof, meta_iso_oof],
                           np.array([0.70, 0.30]))
    stack4_test = log_blend([stack3a_test, meta_iso_test],
                            np.array([0.70, 0.30]))

    ba_lb3, n_lb3 = _eval(lb3_oof, y, bias_recipe, mask=nz)
    ba_lb4, n_lb4 = _eval(stack4_oof, y, bias_recipe, mask=nz)
    ba_mam, n_mam = _eval(mamba_oof, y, bias_recipe, mask=nz)
    log(f"  LB-best 3-stack on filled rows: bal={ba_lb3:.5f}  errs={n_lb3}")
    log(f"  LB-best 4-stack on filled rows: bal={ba_lb4:.5f}  errs={n_lb4}")
    log(f"  Mamba         on filled rows: bal={ba_mam:.5f}  errs={n_mam}")

    # Tuned bias on filled rows only (PROBE diagnostic).
    prior = np.bincount(y[nz], minlength=3) / max(int(nz.sum()), 1)
    bias_mam, ba_mam_tuned = tune_log_bias(mamba_oof[nz], y[nz], prior)
    log(f"  Mamba tuned (filled): bal={ba_mam_tuned:.5f}  "
        f"bias={bias_mam.round(4).tolist()}")

    # Jaccards on filled rows.
    err_mam = _errmask(mamba_oof[nz], y[nz], bias_recipe)
    err_lb3 = _errmask(lb3_oof[nz], y[nz], bias_recipe)
    err_lb4 = _errmask(stack4_oof[nz], y[nz], bias_recipe)
    err_realmlp = _errmask(realmlp_oof[nz], y[nz], bias_recipe)

    def jacc(a, b):
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return inter / max(union, 1)

    j_lb3 = jacc(err_mam, err_lb3)
    j_lb4 = jacc(err_mam, err_lb4)
    j_realmlp = jacc(err_mam, err_realmlp)
    log(f"  Jaccard mamba vs LB-3stack = {j_lb3:.4f}")
    log(f"  Jaccard mamba vs LB-4stack = {j_lb4:.4f}")
    log(f"  Jaccard mamba vs RealMLP   = {j_realmlp:.4f}")

    # Gate decision.
    log("=== gate ===")
    if ba_mam_tuned < 0.50:
        verdict = ("FAIL: standalone tuned bal_acc < 0.50 — model "
                   "didn't converge")
    elif j_lb4 >= 0.85:
        verdict = (f"REDUNDANT: Jaccard {j_lb4:.3f} >= 0.85 vs LB-4stack — "
                   "blend null prior")
    elif n_mam > int(1.05 * n_lb4):
        verdict = (f"MAGNITUDE-TRAP RISK: errs {n_mam} > 1.05x anchor "
                   f"({n_lb4}); blend likely net-negative")
    elif j_lb4 >= 0.75:
        verdict = (f"WARN: Jaccard {j_lb4:.3f} in [0.75, 0.85) - "
                   "blend lift capped ~+0.00015")
    else:
        verdict = (f"PASS: Jaccard {j_lb4:.3f} < 0.75 AND errs "
                   f"{n_mam} <= 1.05x anchor - PROCEED to 5-fold")
    log(f"  verdict: {verdict}")

    # Quick blend sweep diagnostic on filled rows (no submission).
    log("=== fixed-bias α-sweep on filled rows (diagnostic) ===")
    rows = []
    for a in np.arange(0.0, 0.40, 0.05):
        if a == 0.0:
            blended = stack4_oof
        else:
            blended = log_blend([stack4_oof, mamba_oof],
                                np.array([1.0 - a, a]))
        ba, errs = _eval(blended, y, bias_recipe, mask=nz)
        log(f"    α={a:.3f}  bal={ba:.5f}  Δ={ba-ba_lb4:+.5f}  errs={errs}")
        rows.append({"alpha": float(a), "bal": ba, "delta": ba - ba_lb4,
                     "errs": errs})

    out = {
        "suffix": suffix,
        "filled_rows": int(nz.sum()),
        "standalone": {
            "bal_at_recipe_bias": float(ba_mam),
            "tuned_bal": float(ba_mam_tuned),
            "errs_at_recipe_bias": int(n_mam),
            "tuned_bias": bias_mam.tolist(),
        },
        "anchors_on_filled": {
            "lb3_bal": float(ba_lb3), "lb3_errs": n_lb3,
            "lb4_bal": float(ba_lb4), "lb4_errs": n_lb4,
        },
        "jaccards": {
            "vs_lb3": float(j_lb3),
            "vs_lb4": float(j_lb4),
            "vs_realmlp": float(j_realmlp),
        },
        "verdict": verdict,
        "sweep_filled": rows,
    }
    out_path = ART / f"blend_mamba{suffix}_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
