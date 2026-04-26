"""Blend-gate analysis for TabM PROBE/PROD OOF/test outputs.

Runs AFTER `kaggle kernels output chrisleitescha/irrigation-tabm`
downloads oof_tabm.npy + test_tabm.npy into scripts/artifacts/.

Mirrors blend_mamba.py exactly except for the file suffix.

Gate (CLAUDE.md NN family decision rule):
  Jaccard < 0.75 vs LB-best 4-stack AND errs <= 1.05x anchor → PROCEED
  Jaccard 0.75-0.85: WARN, blend lift capped ~+0.00015
  Jaccard >= 0.85: REDUNDANT, close lever
  errs > 1.05x anchor: MAGNITUDE-TRAP, close lever

LB-best 4-stack: LB 0.98094, OOF 0.98084
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


def _per_class_iso(oof, y, also_transform=None):
    out = np.zeros_like(oof)
    out_extra = (np.zeros_like(also_transform)
                 if also_transform is not None else None)
    for k in range(3):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(oof[:, k], (y == k).astype(np.float64))
        out[:, k] = ir.transform(oof[:, k])
        if out_extra is not None:
            out_extra[:, k] = ir.transform(also_transform[:, k])
    out = out / out.sum(axis=1, keepdims=True).clip(1e-9)
    if out_extra is not None:
        out_extra = out_extra / out_extra.sum(axis=1, keepdims=True).clip(1e-9)
        return out, out_extra
    return out


def main() -> None:
    suffix = sys.argv[1] if len(sys.argv) > 1 else ""
    oof_path = ART / f"oof_tabm{suffix}.npy"
    test_path = ART / f"test_tabm{suffix}.npy"
    if not oof_path.exists():
        raise SystemExit(
            f"{oof_path} not found - run "
            "`kaggle kernels output chrisleitescha/irrigation-tabm "
            "-p scripts/artifacts/` first."
        )

    log(f"loading tabm{suffix} OOF / test")
    tabm_oof = np.load(oof_path)
    tabm_test = np.load(test_path)
    nz = tabm_oof.sum(axis=1) > 0
    full_run = bool(nz.sum() >= len(tabm_oof) * 0.99)
    log(f"  filled rows = {int(nz.sum()):,}/{len(tabm_oof):,} "
        f"({'FULL 5-fold' if full_run else 'PARTIAL/PROBE'})")

    train = pd.read_csv("data/train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

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

    nonrule_iso_oof, nonrule_iso_test = _per_class_iso(
        nonrule_oof, y, also_transform=nonrule_test)
    meta_iso_oof, meta_iso_test = _per_class_iso(
        meta_oof, y, also_transform=meta_test)

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
    ba_tab, n_tab = _eval(tabm_oof, y, bias_recipe, mask=nz)
    log(f"  LB-best 3-stack on filled rows: bal={ba_lb3:.5f}  errs={n_lb3}")
    log(f"  LB-best 4-stack on filled rows: bal={ba_lb4:.5f}  errs={n_lb4}")
    log(f"  TabM          on filled rows: bal={ba_tab:.5f}  errs={n_tab}")

    prior = np.bincount(y[nz], minlength=3) / max(int(nz.sum()), 1)
    bias_tab, ba_tab_tuned = tune_log_bias(tabm_oof[nz], y[nz], prior)
    log(f"  TabM tuned (filled): bal={ba_tab_tuned:.5f}  "
        f"bias={bias_tab.round(4).tolist()}")

    err_tab = _errmask(tabm_oof[nz], y[nz], bias_recipe)
    err_lb3 = _errmask(lb3_oof[nz], y[nz], bias_recipe)
    err_lb4 = _errmask(stack4_oof[nz], y[nz], bias_recipe)
    err_realmlp = _errmask(realmlp_oof[nz], y[nz], bias_recipe)

    def jacc(a, b):
        inter = int((a & b).sum())
        union = int((a | b).sum())
        return inter / max(union, 1)

    j_lb3 = jacc(err_tab, err_lb3)
    j_lb4 = jacc(err_tab, err_lb4)
    j_realmlp = jacc(err_tab, err_realmlp)
    log(f"  Jaccard tabm vs LB-3stack = {j_lb3:.4f}")
    log(f"  Jaccard tabm vs LB-4stack = {j_lb4:.4f}")
    log(f"  Jaccard tabm vs RealMLP   = {j_realmlp:.4f}")

    log("=== gate ===")
    if ba_tab_tuned < 0.50:
        verdict = "FAIL: standalone tuned bal_acc < 0.50 - didn't converge"
    elif j_lb4 >= 0.85:
        verdict = (f"REDUNDANT: Jaccard {j_lb4:.3f} >= 0.85 vs LB-4stack")
    elif n_tab > int(1.05 * n_lb4):
        verdict = (f"MAGNITUDE-TRAP: errs {n_tab} > 1.05x anchor ({n_lb4})")
    elif j_lb4 >= 0.75:
        verdict = (f"WARN: Jaccard {j_lb4:.3f} in [0.75, 0.85)")
    else:
        verdict = (f"PASS: Jaccard {j_lb4:.3f} < 0.75 AND errs "
                   f"{n_tab} <= 1.05x anchor - PROCEED")
    log(f"  verdict: {verdict}")

    log("=== fixed-bias α-sweep (diagnostic) ===")
    rows = []
    for a in np.arange(0.0, 0.40, 0.05):
        if a == 0.0:
            blended = stack4_oof
        else:
            blended = log_blend([stack4_oof, tabm_oof],
                                np.array([1.0 - a, a]))
        ba, errs = _eval(blended, y, bias_recipe, mask=nz)
        log(f"    α={a:.3f}  bal={ba:.5f}  Δ={ba-ba_lb4:+.5f}  errs={errs}")
        rows.append({"alpha": float(a), "bal": ba, "delta": ba - ba_lb4,
                     "errs": errs})

    out = {
        "suffix": suffix,
        "filled_rows": int(nz.sum()),
        "full_run": full_run,
        "standalone": {
            "bal_at_recipe_bias": float(ba_tab),
            "tuned_bal": float(ba_tab_tuned),
            "errs_at_recipe_bias": int(n_tab),
            "tuned_bias": bias_tab.tolist(),
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
    out_path = ART / f"blend_tabm{suffix}_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
