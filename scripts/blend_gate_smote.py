"""Blend-gate for SMOTE-NC component vs LB-best 0.98094 (Tier 1b meta-stacker).

Reconstruction of the LB-0.98094 anchor:
    1. lb3 = log_blend(recipe, pseudo_s1, pseudo_s7; 0.25/0.35/0.40)
    2. stack1 = log_blend(lb3, realmlp; 0.80/0.20)
    3. stack2 = log_blend(stack1, xgb_nonrule__iso; 0.925/0.075)
    4. xgb_metastack__iso = isotonic-calibrated meta-stacker output
    5. final = log_blend(stack2, xgb_metastack__iso; 0.70/0.30)

Smoke result before launch: SMOTE-NC standalone OOF (smoke 20k/2-fold) =
0.96555 tuned. Recipe at same SMOKE config = 0.96381. Smoke Δ +0.00174
suggests training-data augmentation produces non-trivial signal.

This gate runs the blend math vs the LB-0.98094 anchor at recipe's
fixed bias [1.43, 1.47, 3.40] (binhigh-rule compliance).

Outputs:
    blend_gate_smote_results.json — Jaccard, errs, sweep, per-class recall
    submission_lb_best_098094_smote_blend.csv — emitted only if Δ ≥ +2e-4
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

import sys; sys.path.insert(0, "scripts")
from common import fast_bal_acc, log_blend  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions"); SUB.mkdir(exist_ok=True)
DATA = Path("data")
TARGET = "Irrigation_Need"
CLS = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS.items()}
BIAS = np.array([1.4324, 1.4689, 3.4008])


def _normed(p):
    return p / np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    o = np.zeros_like(oof, dtype=np.float32)
    t = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        o[:, c] = ir.predict(oof[:, c])
        t[:, c] = ir.predict(test[:, c])
    return _normed(o), _normed(t)


def build_lb_best_098094(y):
    r_o = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    r_t = _normed(np.load(ART / "test_recipe_full_te.npy"))
    s1_o = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1_t = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7_o = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7_t = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm_o = _normed(np.load(ART / "oof_realmlp.npy"))
    rm_t = _normed(np.load(ART / "test_realmlp.npy"))
    nr_o = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nr_t = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    meta_o = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = _normed(np.load(ART / "test_xgb_metastack.npy"))

    nr_iso_o, nr_iso_t = iso_cal(nr_o, nr_t, y)
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)

    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r_o, s1_o, s7_o], w3)
    lb3_t = log_blend([r_t, s1_t, s7_t], w3)
    s1stack_o = log_blend([lb3_o, rm_o], np.array([0.8, 0.2]))
    s1stack_t = log_blend([lb3_t, rm_t], np.array([0.8, 0.2]))
    s2stack_o = log_blend([s1stack_o, nr_iso_o], np.array([0.925, 0.075]))
    s2stack_t = log_blend([s1stack_t, nr_iso_t], np.array([0.925, 0.075]))
    final_o = log_blend([s2stack_o, meta_iso_o], np.array([0.70, 0.30]))
    final_t = log_blend([s2stack_t, meta_iso_t], np.array([0.70, 0.30]))
    return final_o, final_t


def argmax_at_bias(p, b=BIAS):
    return (np.log(np.clip(p, 1e-9, 1.0)) + b).argmax(1)


def main():
    print("[gate] loading SMOTE OOF + test")
    smote_oof = _normed(np.load(ART / "oof_recipe_smote2x.npy"))
    smote_test = _normed(np.load(ART / "test_recipe_smote2x.npy"))
    print(f"[gate] SMOTE OOF shape {smote_oof.shape}, test shape {smote_test.shape}")

    print("[gate] loading y")
    y = pd.read_csv(DATA / "train.csv", usecols=[TARGET])[TARGET].map(CLS).to_numpy(dtype=np.int64)

    print("[gate] reconstructing LB-best 0.98094 anchor")
    anchor_oof, anchor_test = build_lb_best_098094(y)
    cc = np.bincount(y, minlength=3)
    anchor_pred = argmax_at_bias(anchor_oof)
    anchor_bal = fast_bal_acc(y, anchor_pred, class_counts=cc)
    print(f"[gate] anchor OOF bal_acc @ recipe bias = {anchor_bal:.5f}  errs={int((anchor_pred != y).sum()):,}")

    print("[gate] standalone SMOTE diagnostics")
    smote_pred = argmax_at_bias(smote_oof)
    smote_bal = fast_bal_acc(y, smote_pred, class_counts=cc)
    print(f"[gate] SMOTE  OOF bal_acc @ recipe bias = {smote_bal:.5f}  errs={int((smote_pred != y).sum()):,}")

    # Per-class recall comparison
    cm_a = confusion_matrix(y, anchor_pred)
    cm_s = confusion_matrix(y, smote_pred)
    rec_a = cm_a.diagonal() / cm_a.sum(axis=1)
    rec_s = cm_s.diagonal() / cm_s.sum(axis=1)
    print(f"[gate] anchor per-class recall: L={rec_a[0]:.5f} M={rec_a[1]:.5f} H={rec_a[2]:.5f}")
    print(f"[gate] SMOTE  per-class recall: L={rec_s[0]:.5f} M={rec_s[1]:.5f} H={rec_s[2]:.5f}")

    a_err = anchor_pred != y
    s_err = smote_pred != y
    inter = (a_err & s_err).sum()
    union = (a_err | s_err).sum()
    jacc = inter / max(union, 1)
    print(f"[gate] Jaccard(SMOTE, anchor) = {jacc:.4f}")

    # Blend-gate decision
    if jacc < 0.80 and s_err.sum() <= a_err.sum():
        print(f"[gate] BLEND-GATE PASS: Jaccard<0.80 AND errs<=anchor → PLAUSIBLE")
    elif jacc < 0.85:
        print(f"[gate] BORDERLINE: Jaccard {jacc:.4f}, errs Δ={int(s_err.sum() - a_err.sum())}")
    else:
        print(f"[gate] REDUNDANT: Jaccard ≥ 0.85")

    # Fixed-bias log-blend sweep
    sweep = []
    peak_alpha, peak_bal = 0.0, anchor_bal
    for alpha in np.linspace(0, 0.5, 21):
        if alpha == 0:
            mixed = anchor_oof
        else:
            mixed = log_blend([anchor_oof, smote_oof],
                              np.array([1 - alpha, alpha]))
        pred = argmax_at_bias(mixed)
        bal = fast_bal_acc(y, pred, class_counts=cc)
        sweep.append((float(alpha), float(bal)))
        if bal > peak_bal:
            peak_alpha, peak_bal = float(alpha), float(bal)

    print(f"[gate] fixed-bias α-sweep vs LB-0.98094 anchor:")
    for a, b in sweep:
        marker = "  ← peak" if abs(a - peak_alpha) < 1e-9 and a > 0 else ""
        print(f"       α={a:.3f}  bal_acc={b:.5f}  Δ={b-anchor_bal:+.5f}{marker}")
    print(f"[gate] peak α={peak_alpha:.3f}  Δ={peak_bal-anchor_bal:+.5f}")

    # Auto-emit submission if Δ ≥ +2e-4 (LB-transfer threshold)
    candidate_path = None
    if peak_bal - anchor_bal >= 2e-4:
        print(f"[gate] EMIT: Δ ≥ +2e-4, building submission")
        mixed_test = log_blend([anchor_test, smote_test],
                               np.array([1 - peak_alpha, peak_alpha]))
        pred_idx = argmax_at_bias(mixed_test)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = pd.DataFrame({
            "id": sample["id"],
            TARGET: [IDX2CLS[int(i)] for i in pred_idx],
        })
        candidate_path = SUB / f"submission_lb_098094_smote_a{peak_alpha:.3f}.csv".replace(".", "p", 1).replace("psv", ".csv")
        # cleaner filename:
        candidate_path = SUB / f"submission_lb_098094_smote_blend.csv"
        sub.to_csv(candidate_path, index=False)
        print(f"[gate] wrote {candidate_path}")
    else:
        print(f"[gate] NO EMIT: Δ {peak_bal-anchor_bal:+.5f} below +2e-4 threshold")

    out = dict(
        anchor_oof_bal=float(anchor_bal),
        anchor_errs=int(a_err.sum()),
        smote_oof_bal=float(smote_bal),
        smote_errs=int(s_err.sum()),
        jaccard=float(jacc),
        anchor_per_class_recall=rec_a.tolist(),
        smote_per_class_recall=rec_s.tolist(),
        peak_alpha=peak_alpha,
        peak_bal=peak_bal,
        delta_vs_anchor=peak_bal - anchor_bal,
        sweep=sweep,
        candidate_submission=str(candidate_path) if candidate_path else None,
    )
    with open(ART / "blend_gate_smote_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gate] wrote scripts/artifacts/blend_gate_smote_results.json")


if __name__ == "__main__":
    main()
