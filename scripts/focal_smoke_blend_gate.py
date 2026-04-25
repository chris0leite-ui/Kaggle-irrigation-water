"""Post-hoc blend-gate for the focal smoke OOF.

Standalone script: loads oof_recipe_focal_g2_aH1.npy (2-fold smoke)
and evaluates Jaccard + error-count + fixed-bias blend sweep vs
LB-best 3-way teacher. Same protocol as recipe_full_te blend gates.

Note: the focal OOF was produced with FOLD_SEED=42 but 2-FOLD
StratifiedKFold, so only half of the 630k rows have predictions
from "held-out" folds (the other half appear in training). This
makes the OOF NOT directly comparable to the 5-fold teacher OOF.
Still a useful smoke diagnostic — if even under this weaker OOF
the focal shows blend-gate potential (Jaccard < 0.85), production
is worth launching. If it's already redundant with teacher, skip.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys; sys.path.insert(0, "scripts")
from common import fast_bal_acc, log_blend, load_oof_pair  # noqa: E402

ART = Path("scripts/artifacts")
TARGET = "Irrigation_Need"
CLS = {"Low": 0, "Medium": 1, "High": 2}


def main():
    y = pd.read_csv("data/train.csv", usecols=[TARGET])[TARGET].map(CLS).to_numpy(dtype=np.int64)

    oof_focal = np.load(ART / "oof_recipe_focal_g2_aH1.npy")
    # Determine which rows were actually held-out (nonzero probs).
    held_mask = oof_focal.sum(axis=1) > 1e-4
    print(f"[gate] focal OOF: {held_mask.sum():,}/{len(y):,} rows held-out "
          f"({100*held_mask.mean():.1f}%)")

    o_rec, t_rec = load_oof_pair("recipe_full_te")
    o_s1, t_s1 = load_oof_pair("recipe_pseudolabel")
    o_s7, t_s7 = load_oof_pair("recipe_pseudolabel_seed7labeler")

    with open(ART / "recipe_full_te_results.json") as f:
        bias = np.array(json.load(f)["log_bias"], dtype=np.float64)
    print(f"[gate] recipe tuned bias = {bias.tolist()}")

    w = np.array([0.25, 0.35, 0.40])
    o_teacher = log_blend([o_rec, o_s1, o_s7], w)

    def argmax_at_bias(p, b):
        return (np.log(np.clip(p, 1e-9, 1.0)) + b).argmax(1)

    # Restrict everything to held-out rows only — honest comparison.
    y_h = y[held_mask]
    cc = np.bincount(y_h, minlength=3)

    teach_pred = argmax_at_bias(o_teacher[held_mask], bias)
    focal_pred = argmax_at_bias(oof_focal[held_mask], bias)
    teach_bal = fast_bal_acc(y_h, teach_pred, class_counts=cc)
    focal_bal = fast_bal_acc(y_h, focal_pred, class_counts=cc)

    teach_err = teach_pred != y_h
    focal_err = focal_pred != y_h
    inter = (teach_err & focal_err).sum()
    union = (teach_err | focal_err).sum()
    jacc = inter / max(union, 1)

    print(f"[gate] evaluated on {len(y_h):,} held-out rows")
    print(f"[gate] teacher bal_acc @ recipe bias = {teach_bal:.5f}  errs={int(teach_err.sum()):,}")
    print(f"[gate] focal   bal_acc @ recipe bias = {focal_bal:.5f}  errs={int(focal_err.sum()):,}")
    print(f"[gate] Jaccard(focal vs teacher) = {jacc:.4f}")
    if jacc < 0.80 and focal_err.sum() <= teach_err.sum():
        print(f"[gate] BLEND-GATE PASS: Jaccard<0.80 AND errs<=teacher → PLAUSIBLE")
    elif jacc < 0.85:
        print(f"[gate] BORDERLINE: Jaccard in 0.80-0.85 band — production may or may not lift")
    else:
        print(f"[gate] REDUNDANT: Jaccard >= 0.85 — skip production, errors overlap too much")

    # Fixed-bias log-blend sweep vs teacher on held-out rows
    results = []
    peak_alpha, peak_bal = 0.0, teach_bal
    for alpha in np.linspace(0, 0.5, 11):
        if alpha == 0:
            mixed = o_teacher[held_mask]
        else:
            mixed = log_blend([o_teacher[held_mask], oof_focal[held_mask]],
                              np.array([1 - alpha, alpha]))
        pred = argmax_at_bias(mixed, bias)
        bal = fast_bal_acc(y_h, pred, class_counts=cc)
        results.append((float(alpha), float(bal)))
        if bal > peak_bal:
            peak_alpha, peak_bal = float(alpha), float(bal)

    print(f"[gate] blend sweep vs teacher (fixed recipe bias, held-out only):")
    for a, b in results:
        marker = " <-- peak" if abs(a - peak_alpha) < 1e-9 else ""
        print(f"       α={a:.3f}  bal_acc={b:.5f}  Δ={b-teach_bal:+.5f}{marker}")

    # Write diagnostic JSON
    out = dict(
        holdout_rows=int(held_mask.sum()),
        teacher_bal=teach_bal,
        teacher_errs=int(teach_err.sum()),
        focal_bal=focal_bal,
        focal_errs=int(focal_err.sum()),
        jaccard_vs_teacher=jacc,
        peak_alpha=peak_alpha,
        peak_bal=peak_bal,
        delta_vs_teacher=peak_bal - teach_bal,
        blend_sweep=results,
    )
    with open(ART / "recipe_focal_g2_aH1_blend_gate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[gate] wrote recipe_focal_g2_aH1_blend_gate.json")


if __name__ == "__main__":
    main()
