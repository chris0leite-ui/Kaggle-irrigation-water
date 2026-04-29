"""Analyze T2 pseudo-label retrain (LB 0.98129 labeler at τ=0.99).

Two-phase analysis:
  Phase A: standalone candidate diagnostic
    - Tuned OOF + bias vs recipe baseline (0.97967) and prior pseudo_s1 (0.97993)
    - Jaccard vs LB-best v1 RF natural standalone (LB 0.98129)
    - Test-side disagreement count
  Phase B: bank-add prediction
    - Computed via correlation with v1 bank components
    - Predicts whether T2 will help or LB-regress as a v1 RF natural input

Output: scripts/artifacts/analyze_t2_pseudo_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUFFIX = "lb98129labeler_t099"

CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
TARGET = "Irrigation_Need"


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def main():
    print("loading train labels")
    y = pd.read_csv("data/train.csv")[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    n_tr = len(y)
    prior = np.bincount(y, minlength=3) / n_tr

    # T2 outputs
    t2_oof = np.load(ART / f"oof_recipe_pseudolabel_{SUFFIX}.npy").astype(np.float32)
    t2_test = np.load(ART / f"test_recipe_pseudolabel_{SUFFIX}.npy").astype(np.float32)
    with open(ART / f"recipe_pseudolabel_{SUFFIX}_results.json") as f:
        t2_res = json.load(f)

    print(f"T2 OOF shape {t2_oof.shape}  test shape {t2_test.shape}")

    # Anchor: v1 RF natural standalone (LB 0.98129)
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_bias_arr = np.array([0.4324, 0.8689, 3.2008])  # documented v1 LB-best bias
    v1_pred_oof = (safelog(v1_oof) + v1_bias_arr).argmax(1)
    v1_pred_test = (safelog(v1_test) + v1_bias_arr).argmax(1)
    v1_oof_bal = float(np.mean([
        per_class_recall(y, v1_pred_oof)[k] for k in range(3)
    ]))
    print(f"v1 LB 0.98129 reproduced OOF bal_acc = {v1_oof_bal:.5f} (expect 0.98063)")

    # Phase A — standalone diagnostic
    t2_bias = np.array(t2_res["log_bias"])
    t2_tuned = float(t2_res.get("tuned_log_bias_bal_acc",
                                t2_res.get("tuned_log_bias")))
    print()
    print("=== PHASE A — STANDALONE DIAGNOSTIC ===")
    print(f"  recipe baseline (no pseudo)        OOF tuned ~ 0.97967")
    print(f"  recipe + pseudo_s1 (recipe labeler) OOF tuned ~ 0.97993")
    print(f"  recipe + pseudo (LB-blend labeler)  OOF tuned ~ 0.98002 (NULL on LB)")
    print(f"  T2 (LB 0.98129 labeler, τ=0.99)    OOF tuned = {t2_tuned:.5f}")
    print(f"    bias = {t2_bias.round(4).tolist()}")

    # Per-class recall at tuned bias
    t2_pred_oof = (safelog(t2_oof) + t2_bias).argmax(1)
    t2_pcr = per_class_recall(y, t2_pred_oof)
    print(f"  T2 PCR = [L={t2_pcr[0]:.5f} M={t2_pcr[1]:.5f} H={t2_pcr[2]:.5f}]")

    # Bias drift
    minus_log_prior = -np.log(prior)
    drift = t2_bias - minus_log_prior
    print(f"  -log(prior) = {minus_log_prior.round(4).tolist()}")
    print(f"  drift = {drift.round(4).tolist()}  |max| = {abs(drift).max():.4f}")
    if abs(drift).max() < 0.5:
        print(f"  natural-cal? PASS (close to natural-cal band)")
    else:
        print(f"  natural-cal? recipe-family bias profile")

    # Phase B — orthogonality vs v1 RF natural
    print()
    print("=== PHASE B — ORTHOGONALITY VS v1 RF NATURAL (LB 0.98129) ===")

    # Test-side disagreement at OWN tuned bias each
    t2_pred_test = (safelog(t2_test) + t2_bias).argmax(1)
    diff_test = (t2_pred_test != v1_pred_test).sum()
    print(f"  Test-side argmax disagreement: {int(diff_test)} / {len(v1_pred_test)} ({100*diff_test/len(v1_pred_test):.3f}%)")

    # Error Jaccard at recipe bias (use both at same anchor for apples-to-apples)
    recipe_bias = np.array([1.4324, 1.4689, 3.4008])
    t2_pred_at_recipe = (safelog(t2_oof) + recipe_bias).argmax(1)
    v1_pred_at_recipe = (safelog(v1_oof) + recipe_bias).argmax(1)
    t2_errs = (t2_pred_at_recipe != y)
    v1_errs = (v1_pred_at_recipe != y)
    inter = (t2_errs & v1_errs).sum()
    union = (t2_errs | v1_errs).sum()
    jaccard = float(inter / max(union, 1))
    print(f"  At recipe-bias (both): T2 errs={int(t2_errs.sum())} v1 errs={int(v1_errs.sum())}")
    print(f"  Error Jaccard = {jaccard:.4f}  ({'novel' if jaccard < 0.80 else 'redundant'})")

    # Net rare-class flips on TEST vs v1
    add_h = int(((t2_pred_test == 2) & (v1_pred_test != 2)).sum())
    rem_h = int(((v1_pred_test == 2) & (t2_pred_test != 2)).sum())
    net_h = add_h - rem_h
    churn_h = add_h + rem_h
    g4_ratio = abs(net_h) / max(churn_h, 1)
    print(f"  H-flips: +{add_h} ADD-H, -{rem_h} REMOVE-H, net {net_h:+d}, ratio {g4_ratio:.3f}")
    direction = "ADD-High" if net_h > 0 else ("REMOVE-High" if net_h < 0 else "neutral")
    print(f"  Direction: {direction}")

    # Phase C — bank-add prediction
    print()
    print("=== PHASE C — V1 RF NATURAL BANK-ADD PREDICTION ===")

    # Standalone projection at v1's gap (-0.00066 OOF→LB)
    v1_gap = -0.00066
    proj_lb_standalone = t2_tuned + (-v1_gap)
    print(f"  T2 standalone projected LB (at v1 gap): {proj_lb_standalone:.5f}")

    # Bank-add predictor: based on the prior 3 RF natural bank-extension nulls,
    # adding any new component to v1's 7-component bank LB-regresses ~-0.00031.
    # Exception: components with HIGH orthogonality (Jaccard < 0.75) AND
    # FEWER errors than v1 might break the pattern.
    print(f"  Orthogonality test: Jaccard {jaccard:.4f} (threshold 0.75)")
    if jaccard < 0.75 and t2_errs.sum() < v1_errs.sum():
        print(f"  → BANK-ADD CANDIDATE: passes orthogonality+magnitude")
    elif jaccard < 0.80:
        print(f"  → BORDERLINE: may help if other 4-gate criteria pass")
    else:
        print(f"  → REDUNDANT: bank-add likely null/negative per prior 3 nulls")

    summary = dict(
        t2_oof_tuned=t2_tuned,
        t2_bias=t2_bias.tolist(),
        t2_drift=drift.tolist(),
        t2_drift_max=float(abs(drift).max()),
        t2_pcr=t2_pcr.tolist(),
        v1_oof_bal_reproduced=v1_oof_bal,
        test_diff_vs_v1=int(diff_test),
        errs_t2_at_recipe_bias=int(t2_errs.sum()),
        errs_v1_at_recipe_bias=int(v1_errs.sum()),
        error_jaccard_t2_v1=jaccard,
        net_high_flip=net_h,
        churn_high=churn_h,
        g4_ratio=g4_ratio,
        direction=direction,
        projected_lb_standalone=float(proj_lb_standalone),
        keep_rate=t2_res.get("keep_rate"),
        n_pseudo=t2_res.get("n_pseudo"),
        pseudo_label_dist=t2_res.get("pseudo_label_dist"),
    )
    out_p = ART / "analyze_t2_pseudo_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
