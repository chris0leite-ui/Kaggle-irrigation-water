"""J4 KAN blend-gate analysis vs LB-best 4-stack.

PROBE = 1-fold (126k val rows out of 630k).

Apples-to-apples gate vs LB-best 4-stack restricted to the same fold-1
val rows: compute Jaccard, errors, per-class recall, and a fixed-bias
log-blend sweep. Decision rule:
  - Jaccard < 0.75 AND errs ≤ 1.05 × anchor → PROCEED to full 5-fold
  - 0.75 ≤ Jaccard < 0.85 → cap blend lift expectation +0.00015
  - Jaccard ≥ 0.85 OR errs > 1.05 × anchor → close as 15th NN null
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    fast_bal_acc, log_blend, tune_log_bias, CLS2IDX,
)
from tier1b_helpers import build_lbbest_stack, load_y, BIAS  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True, parents=True)
SEED = 42
N_FOLDS = 5
W_RECIPE, W_S1, W_S7 = 0.25, 0.35, 0.40


def per_class_recall(y, pred, K=3):
    cc = np.bincount(y, minlength=K)
    matches = (pred == y)
    return np.array(
        [matches[y == k].sum() / max(cc[k], 1) for k in range(K)]
    )


def main() -> None:
    print("=== J4 KAN blend-gate (fold-1 only) ===")
    y = load_y()
    prior = np.bincount(y, minlength=3) / len(y)

    # Reproduce fold split
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_id = np.full(len(y), -1, dtype=np.int32)
    for fid, (_, va_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold_id[va_idx] = fid
    fold1_mask = fold_id == 0

    # Load KAN OOF (only fold-1 rows are populated by PROBE)
    oof_kan = np.load(ART / "oof_kan_probe.npy")
    nz_kan = oof_kan.sum(1) > 0
    assert nz_kan.sum() > 0
    print(f"KAN PROBE populated rows: {int(nz_kan.sum()):,}  "
          f"fold-1 rows: {int(fold1_mask.sum()):,}")

    # Build the LB-best 3-stack (anchor used in tier1b_xgb_metastack);
    # this is the EXACT 0.98061 OOF anchor underneath the LB 0.98094
    # primary. Using BIAS = LB-best 4-stack tuned bias for fixed-bias eval.
    print("building LB-best 3-stack via tier1b_helpers")
    oof_anchor, _ = build_lbbest_stack(y)
    bias = np.array(BIAS)
    anchor_name = "LB-best 3-stack (lb3+RM+nonrule_iso)"
    bal_anchor_full = fast_bal_acc(
        y, (np.log(np.clip(oof_anchor, 1e-9, 1.0)) + bias).argmax(1))
    print(f"\nanchor: {anchor_name}")
    print(f"anchor full-OOF bal_acc @ fixed BIAS = {bal_anchor_full:.5f}  "
          f"bias = {bias.round(4).tolist()}")

    # Restrict to fold-1 only for apples-to-apples
    y_f1 = y[fold1_mask].astype(np.int32)
    oof_anchor_f1 = oof_anchor[fold1_mask]
    oof_kan_f1 = oof_kan[fold1_mask]
    eps = 1e-9
    pred_anchor = (np.log(np.clip(oof_anchor_f1, eps, 1.0))
                   + bias).argmax(1).astype(np.int32)
    pred_kan = (np.log(np.clip(oof_kan_f1, eps, 1.0))
                + bias).argmax(1).astype(np.int32)
    bal_anchor = fast_bal_acc(y_f1, pred_anchor)
    bal_kan_at_anchor_bias = fast_bal_acc(y_f1, pred_kan)

    # Tune bias on the KAN fold-1 directly (its own optimum)
    bias_kan, bal_kan_tuned = tune_log_bias(oof_kan_f1, y_f1, prior)
    pred_kan_tuned = (np.log(np.clip(oof_kan_f1, eps, 1.0))
                      + bias_kan).argmax(1).astype(np.int32)
    err_anchor = int((pred_anchor != y_f1).sum())
    err_kan_anc = int((pred_kan != y_f1).sum())
    err_kan_own = int((pred_kan_tuned != y_f1).sum())
    rec_anchor = per_class_recall(y_f1, pred_anchor)
    rec_kan_anc = per_class_recall(y_f1, pred_kan)
    rec_kan_own = per_class_recall(y_f1, pred_kan_tuned)

    # Jaccard of error sets at anchor bias (apples-to-apples)
    e_a = pred_anchor != y_f1
    e_k = pred_kan != y_f1
    jac = (e_a & e_k).sum() / max((e_a | e_k).sum(), 1)

    print(f"\n=== fold-1 standalone (anchor bias) ===")
    print(f"  anchor      bal={bal_anchor:.5f}  errs={err_anchor:,}  "
          f"PCR={rec_anchor.round(4).tolist()}")
    print(f"  KAN @anchor bal={bal_kan_at_anchor_bias:.5f}  "
          f"errs={err_kan_anc:,}  PCR={rec_kan_anc.round(4).tolist()}")
    print(f"  KAN @own    bal={bal_kan_tuned:.5f}  "
          f"bias={bias_kan.round(4).tolist()}  errs={err_kan_own:,}  "
          f"PCR={rec_kan_own.round(4).tolist()}")
    print(f"  Jaccard(KAN, anchor) errs @ anchor bias = {jac:.4f}")
    err_ratio = err_kan_anc / max(err_anchor, 1)
    print(f"  errs ratio  KAN / anchor = {err_ratio:.3f}")

    # Blend sweep at fixed anchor bias
    print(f"\n=== blend sweep (fixed anchor bias, fold-1 only) ===")
    print(f"{'alpha':>6} {'bal':>9} {'delta':>10} {'errs':>6}")
    rows = []
    for a in [0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30,
              0.40, 0.50]:
        if a == 0.0:
            blend = oof_anchor_f1
        else:
            blend = log_blend([oof_anchor_f1, oof_kan_f1],
                              np.array([1.0 - a, a]))
        pred = (np.log(np.clip(blend, eps, 1.0))
                + bias).argmax(1).astype(np.int32)
        bal = fast_bal_acc(y_f1, pred)
        errs = int((pred != y_f1).sum())
        d = bal - bal_anchor
        rows.append(dict(alpha=a, bal=float(bal), delta=float(d),
                         errs=errs))
        print(f"{a:>6.3f} {bal:>9.5f} {d:>+10.5f} {errs:>6}")

    # Verdict
    gate_jaccard = jac < 0.75
    gate_errs = err_kan_anc <= int(1.05 * err_anchor)
    print("\n=== gate verdict (PROBE → full 5-fold decision) ===")
    print(f"  Jaccard < 0.75 ? {gate_jaccard}  ({jac:.4f})")
    print(f"  errs ≤ 1.05*anchor ? {gate_errs}  "
          f"({err_kan_anc} vs {int(1.05*err_anchor)})")
    if gate_jaccard and gate_errs:
        verdict = "PROCEED — promising for full 5-fold"
    elif gate_jaccard:
        verdict = "BORDERLINE — orthogonality good but magnitude too high"
    elif jac < 0.85:
        verdict = "CAP at +0.00015 lift expectation"
    else:
        verdict = "CLOSE — 15th NN null pattern (Jaccard >= 0.85)"
    print(f"  → {verdict}")

    out = dict(
        anchor=anchor_name,
        anchor_full_oof_bal=float(bal_anchor_full),
        bias=bias.tolist(),
        fold1_n_rows=int(fold1_mask.sum()),
        fold1_anchor=dict(bal=float(bal_anchor), errs=err_anchor,
                          pcr=rec_anchor.tolist()),
        fold1_kan_at_anchor_bias=dict(
            bal=float(bal_kan_at_anchor_bias), errs=err_kan_anc,
            pcr=rec_kan_anc.tolist()),
        fold1_kan_at_own_bias=dict(
            bal=float(bal_kan_tuned), errs=err_kan_own,
            bias=bias_kan.tolist(), pcr=rec_kan_own.tolist()),
        jaccard_at_anchor_bias=float(jac),
        errs_ratio_kan_over_anchor=float(err_ratio),
        blend_sweep=rows,
        gate_jaccard_lt_075=bool(gate_jaccard),
        gate_errs_le_105x_anchor=bool(gate_errs),
        verdict=verdict,
    )
    with open(ART / "blend_kan_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote scripts/artifacts/blend_kan_results.json")


if __name__ == "__main__":
    main()
