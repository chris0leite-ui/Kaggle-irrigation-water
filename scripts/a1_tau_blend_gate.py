"""A1 τ-sweep blend gate.

For each τ ∈ {0.95, 0.97, 0.99}, compute:
  - Standalone tuned OOF + bias (already on disk via results JSON)
  - Errors at recipe bias [1.4324, 1.4689, 3.4008]
  - Jaccard vs stage-1 (`oof_recipe_pseudolabel`)
  - Jaccard vs LB-best 3-way (`recipe + s1 + s7`)
  - Build the τ-substituted primary candidate (replace stage-1 in 3-way)
  - Modified-primary tuned OOF + bias + errs + per-class recall

Gates:
  G1: tuned OOF > stage-1 (0.97993)              ✓ if standalone better
  G2: errs at recipe bias ≤ stage-1's            ✓ if magnitude OK
  G3: Jaccard vs stage-1 < 0.85                  ✓ if structurally orthogonal
  G4: substituted-primary OOF > current primary 0.98084
  G5: substituted-primary errs ≤ current primary
  G6: per-class recall floor (no class drops > 0.0010 vs current primary)

If G4-G6 pass: the τ variant is an LB-probe candidate.

Reads same OOF files as tier1b_greedy_with_meta.py. Writes diagnostic
JSON only — no submission emitted unless invoked separately.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def bal_at_bias(p, y, bias=RECIPE_BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1))


def errs_at_bias(p, y, bias=RECIPE_BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    return int((pred != y).sum()), pred


def jaccard_err(p1, p2, y, bias=RECIPE_BIAS):
    e1 = (np.log(np.clip(p1, 1e-12, 1)) + bias).argmax(1) != y
    e2 = (np.log(np.clip(p2, 1e-12, 1)) + bias).argmax(1) != y
    inter = (e1 & e2).sum()
    union = (e1 | e2).sum()
    return float(inter / max(union, 1))


def main():
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()

    # Stage-1 reference + components
    r = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    rt = _normed(np.load(ART / "test_recipe_full_te.npy"))
    s1 = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1t = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7 = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7t = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm = _normed(np.load(ART / "oof_realmlp.npy"))
    rmt = _normed(np.load(ART / "test_realmlp.npy"))
    nr = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nrt = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    meta_o = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = _normed(np.load(ART / "test_xgb_metastack.npy"))

    s1_errs, _ = errs_at_bias(s1, y)
    print(f"stage-1 (tau=0.98) standalone:")
    print(f"  errs at recipe bias = {s1_errs}")
    print(f"  bal at recipe bias  = {bal_at_bias(s1, y):.5f}")
    print()

    # Build current primary (LB-best 4-stack with iso-meta) for reference
    nr_iso_o, nr_iso_t = iso_cal(nr, nrt, y)
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    st1_o = log_blend([lb3_o, rm], np.array([0.8, 0.2]))
    st1_t = log_blend([lb3_t, rmt], np.array([0.8, 0.2]))
    st2_o = log_blend([st1_o, nr_iso_o], np.array([0.925, 0.075]))
    st2_t = log_blend([st1_t, nr_iso_t], np.array([0.925, 0.075]))
    primary_o = log_blend([st2_o, meta_iso_o], np.array([0.7, 0.3]))
    primary_t = log_blend([st2_t, meta_iso_t], np.array([0.7, 0.3]))
    primary_bal = bal_at_bias(primary_o, y)
    primary_errs, primary_pred = errs_at_bias(primary_o, y)

    primary_recL = recall_score(y, primary_pred, labels=[0], average=None)[0]
    primary_recM = recall_score(y, primary_pred, labels=[1], average=None)[0]
    primary_recH = recall_score(y, primary_pred, labels=[2], average=None)[0]

    print(f"current primary (tau=0.98 in 3-way) reference:")
    print(f"  bal at recipe bias = {primary_bal:.5f}")
    print(f"  errs at recipe bias = {primary_errs}")
    print(f"  per-class recall = L={primary_recL:.4f} M={primary_recM:.4f} H={primary_recH:.4f}")
    print()

    out = {
        "stage_1": {"errs": s1_errs, "bal_at_recipe_bias": float(bal_at_bias(s1, y))},
        "current_primary": {
            "bal_at_recipe_bias": float(primary_bal),
            "errs": primary_errs,
            "recall": {"L": float(primary_recL), "M": float(primary_recM), "H": float(primary_recH)},
        },
        "tau_results": {},
    }

    for tau_str in ["095", "097", "099"]:
        tau_p = ART / f"oof_recipe_pseudolabel_tau{tau_str}.npy"
        tau_pt = ART / f"test_recipe_pseudolabel_tau{tau_str}.npy"
        if not tau_p.exists():
            print(f"tau={tau_str} oof not found, skipping")
            continue
        tau_o = _normed(np.load(tau_p))
        tau_test = _normed(np.load(tau_pt))

        # Standalone diagnostics
        tau_standalone_bal = bal_at_bias(tau_o, y)
        tau_standalone_errs, _ = errs_at_bias(tau_o, y)
        # Tuned bias (re-run for sanity, also returns final tuned bal)
        prior = np.bincount(y, minlength=3) / len(y)
        tau_bias, tau_tuned_bal = tune_log_bias(tau_o, y, prior)

        # Jaccards vs stage-1 and vs LB-best 3-way
        j_s1 = jaccard_err(tau_o, s1, y)
        j_lb3 = jaccard_err(tau_o, lb3_o, y)

        # Build τ-substituted primary
        sub_lb3_o = log_blend([r, tau_o, s7], w3)
        sub_lb3_t = log_blend([rt, tau_test, s7t], w3)
        sub_st1_o = log_blend([sub_lb3_o, rm], np.array([0.8, 0.2]))
        sub_st1_t = log_blend([sub_lb3_t, rmt], np.array([0.8, 0.2]))
        sub_st2_o = log_blend([sub_st1_o, nr_iso_o], np.array([0.925, 0.075]))
        sub_st2_t = log_blend([sub_st1_t, nr_iso_t], np.array([0.925, 0.075]))
        sub_primary_o = log_blend([sub_st2_o, meta_iso_o], np.array([0.7, 0.3]))
        sub_primary_t = log_blend([sub_st2_t, meta_iso_t], np.array([0.7, 0.3]))
        sub_bal = bal_at_bias(sub_primary_o, y)
        sub_errs, sub_pred = errs_at_bias(sub_primary_o, y)
        sub_recL = recall_score(y, sub_pred, labels=[0], average=None)[0]
        sub_recM = recall_score(y, sub_pred, labels=[1], average=None)[0]
        sub_recH = recall_score(y, sub_pred, labels=[2], average=None)[0]

        # Gates
        G1 = tau_tuned_bal > 0.97993                       # standalone > stage-1
        G2 = tau_standalone_errs <= s1_errs                # magnitude OK
        G3 = j_s1 < 0.85                                   # orthogonal vs stage-1
        G4 = sub_bal > primary_bal                         # substituted primary > current
        G5 = sub_errs <= primary_errs                      # primary errs OK
        G6 = (sub_recL >= primary_recL - 0.0010 and        # per-class floor
              sub_recM >= primary_recM - 0.0010 and
              sub_recH >= primary_recH - 0.0010)
        all_pass = all([G1, G2, G3, G4, G5, G6])

        print(f"=== tau=0.{tau_str} ===")
        print(f"  standalone:    bal_recipe_bias={tau_standalone_bal:.5f}  tuned={tau_tuned_bal:.5f}  errs={tau_standalone_errs}")
        print(f"  Jaccard vs stage-1 = {j_s1:.4f}   vs LB3 = {j_lb3:.4f}")
        print(f"  substituted primary: bal={sub_bal:.5f}  errs={sub_errs}")
        print(f"    recall L={sub_recL:.4f} M={sub_recM:.4f} H={sub_recH:.4f}")
        print(f"    delta vs current primary: bal={sub_bal-primary_bal:+.5f}  errs={sub_errs-primary_errs:+}")
        print(f"  GATES: G1(tuned>stage1)={G1}  G2(errs<=stage1)={G2}  G3(jacc<0.85)={G3}")
        print(f"         G4(sub>current)={G4}  G5(errs<=current)={G5}  G6(recall_floor)={G6}")
        print(f"  ALL PASS = {all_pass}")
        print()

        out["tau_results"][f"tau{tau_str}"] = {
            "standalone": {
                "bal_at_recipe_bias": float(tau_standalone_bal),
                "tuned_bal": float(tau_tuned_bal),
                "tuned_bias": tau_bias.tolist(),
                "errs": tau_standalone_errs,
            },
            "jaccard_vs_stage1": float(j_s1),
            "jaccard_vs_lb3": float(j_lb3),
            "substituted_primary": {
                "bal_at_recipe_bias": float(sub_bal),
                "errs": sub_errs,
                "recall": {"L": float(sub_recL), "M": float(sub_recM), "H": float(sub_recH)},
                "delta_bal_vs_current": float(sub_bal - primary_bal),
                "delta_errs_vs_current": int(sub_errs - primary_errs),
            },
            "gates": {
                "G1_tuned_gt_stage1": bool(G1),
                "G2_errs_le_stage1": bool(G2),
                "G3_jaccard_lt_085": bool(G3),
                "G4_subprimary_gt_current": bool(G4),
                "G5_subprimary_errs_le_current": bool(G5),
                "G6_recall_floor": bool(G6),
                "all_pass": bool(all_pass),
            },
        }

        # Save substituted primary OOF + test for emission later if approved
        np.save(ART / f"oof_primary_sub_tau{tau_str}.npy", sub_primary_o.astype(np.float32))
        np.save(ART / f"test_primary_sub_tau{tau_str}.npy", sub_primary_t.astype(np.float32))

    out_path = ART / "a1_tau_blend_gate_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
