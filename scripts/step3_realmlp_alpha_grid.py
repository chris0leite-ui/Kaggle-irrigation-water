"""Step 3: RealMLP α grid refinement on the LB-best stack path.

The emit_realmlp_3stack.py used α_realmlp=0.200 (greedy-picked on a coarse
grid). Refine to {0.15, 0.175, 0.200, 0.225, 0.25} — fine grid around the
peak. Then also sweep the step-2 α_nonrule_iso around 0.075 for completeness.

Diagnostics:
  - OOF bal_acc per (α_realmlp, α_nonrule_iso) pair
  - error count and per-class recall at the peak
  - Jaccard vs current LB-best 3-stack (should be high — we're perturbing it)

No retraining; this is pure OOF diagnostic on saved OOFs.
If peak α differs from (0.200, 0.075), flag as a potential LB-insurance probe
candidate, but only if OOF delta >= +1e-4.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"
BIAS = np.array([1.4324, 1.4689, 3.4008])


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip",
                                y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def bal(p, y):
    return balanced_accuracy_score(y, (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1))


def per_class_recall(y, pred):
    cc = np.bincount(y, minlength=3)
    hit = np.array([((pred == k) & (y == k)).sum() for k in range(3)],
                   dtype=np.int64)
    return hit / np.maximum(cc, 1)


def main():
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    recipe_oof = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    ps1_oof = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    ps7_oof = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    rm_oof = _normed(np.load(ART / "oof_realmlp.npy"))
    nr_oof = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nr_iso_oof, _ = iso_cal(nr_oof,
                             _normed(np.load(ART / "test_xgb_nonrule.npy")),
                             y)

    recipe_test = _normed(np.load(ART / "test_recipe_full_te.npy"))
    ps1_test = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    ps7_test = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm_test = _normed(np.load(ART / "test_realmlp.npy"))
    nr_test = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    _, nr_iso_test = iso_cal(nr_oof, nr_test, y)

    w3 = np.array([0.25, 0.35, 0.40])
    lb3_oof = log_blend([recipe_oof, ps1_oof, ps7_oof], w3)
    lb3_test = log_blend([recipe_test, ps1_test, ps7_test], w3)

    print(f"{'α_rmlp':>7} {'α_nr':>7} {'OOF':>9} {'errs':>6}")
    print("-" * 35)
    grid_rmlp = [0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300]
    grid_nr = [0.025, 0.050, 0.075, 0.100, 0.125, 0.150]
    results = []

    # Column 1: current (0.200, 0.075) baseline
    for a_rmlp in grid_rmlp:
        s1_o = log_blend([lb3_oof, rm_oof], np.array([1 - a_rmlp, a_rmlp]))
        s1_t = log_blend([lb3_test, rm_test], np.array([1 - a_rmlp, a_rmlp]))
        for a_nr in grid_nr:
            s2_o = log_blend([s1_o, nr_iso_oof], np.array([1 - a_nr, a_nr]))
            s2_t = log_blend([s1_t, nr_iso_test], np.array([1 - a_nr, a_nr]))
            b = bal(s2_o, y)
            pred = (np.log(np.clip(s2_o, 1e-12, 1)) + BIAS).argmax(1)
            errs = int((pred != y).sum())
            tag = " ← CUR" if (abs(a_rmlp - 0.200) < 1e-6 and
                               abs(a_nr - 0.075) < 1e-6) else ""
            print(f"{a_rmlp:>7.3f} {a_nr:>7.3f} {b:>9.5f} {errs:>6}{tag}")
            results.append(dict(a_rmlp=a_rmlp, a_nr=a_nr, oof=float(b), errs=errs))

    best = max(results, key=lambda r: r["oof"])
    cur = [r for r in results
           if abs(r["a_rmlp"] - 0.200) < 1e-6 and abs(r["a_nr"] - 0.075) < 1e-6][0]
    print()
    print(f"current baseline: α_rmlp=0.200 α_nr=0.075  OOF={cur['oof']:.5f}")
    print(f"best on grid:     α_rmlp={best['a_rmlp']}  α_nr={best['a_nr']}  "
          f"OOF={best['oof']:.5f}  Δ={best['oof'] - cur['oof']:+.5f}")

    # Per-class recall for both
    s1_o_cur = log_blend([lb3_oof, rm_oof], np.array([0.8, 0.2]))
    s2_o_cur = log_blend([s1_o_cur, nr_iso_oof], np.array([0.925, 0.075]))
    pred_cur = (np.log(np.clip(s2_o_cur, 1e-12, 1)) + BIAS).argmax(1)
    r_cur = per_class_recall(y, pred_cur)

    a_r, a_n = best["a_rmlp"], best["a_nr"]
    s1_o_best = log_blend([lb3_oof, rm_oof], np.array([1 - a_r, a_r]))
    s2_o_best = log_blend([s1_o_best, nr_iso_oof], np.array([1 - a_n, a_n]))
    pred_best = (np.log(np.clip(s2_o_best, 1e-12, 1)) + BIAS).argmax(1)
    r_best = per_class_recall(y, pred_best)

    print()
    print(f"per-class recall (current):  L={r_cur[0]:.4f} M={r_cur[1]:.4f} H={r_cur[2]:.4f}")
    print(f"per-class recall (best):     L={r_best[0]:.4f} M={r_best[1]:.4f} H={r_best[2]:.4f}")

    delta = best["oof"] - cur["oof"]
    out = dict(
        current=cur, best=best, delta=float(delta), grid=results,
        per_class_current=r_cur.tolist(), per_class_best=r_best.tolist(),
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "step3_realmlp_alpha_grid_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote scripts/artifacts/step3_realmlp_alpha_grid_results.json")

    if delta >= 1e-4:
        print(f"\nΔ={delta:+.5f} ≥ +1e-4 — new config is potentially LB-insurance")
        # Build submission at best α
        s1_t_best = log_blend([lb3_test, rm_test], np.array([1 - a_r, a_r]))
        s2_t_best = log_blend([s1_t_best, nr_iso_test], np.array([1 - a_n, a_n]))
        pred_t = (np.log(np.clip(s2_t_best, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_t]
        tag = f"rmlp{int(a_r*1000):03d}_nr{int(a_n*1000):03d}"
        path = SUB / f"submission_step3_{tag}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {path}  class dist: {sub[TARGET].value_counts().to_dict()}")
    else:
        print(f"\nΔ={delta:+.5f} below +1e-4 — keeping current (0.200, 0.075)")


if __name__ == "__main__":
    main()
