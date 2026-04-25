"""Aggregate per-fold N3 (5-shuffle OTE concat) outputs into final OOF + test.

Per-fold execution writes oof_recipe_2shuffle_fold{f}.npy after each fold
completes (since run_cv_5shuffle accumulates and saves the running totals).
This script reads the LAST fold's checkpoint (which contains all folds'
contributions because each per-fold save writes the running OOF + test
arrays), then computes log-bias tune + standalone bal_acc + blend gate
vs LB-best 4-stack.

Usage: after all 5 RUN_FOLD={1..5} N3 invocations complete.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias, CLS2IDX  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, DATA, SUB, TARGET,
    bal_at_bias as bal, build_lbbest_stack, iso_cal, log,
)


def main():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values

    # Per-fold execution (RUN_FOLD=N mode): each per-fold .npy contains ONLY
    # that fold's val rows (OOF) and that fold's test contribution scaled by
    # 1/N_FOLDS. Sum all 5 to get the final OOF (val folds are disjoint) +
    # test (5 × 1/5 contributions).
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    for f in range(1, 6):
        oof_p = ART / f"oof_recipe_2shuffle_fold{f}.npy"
        test_p = ART / f"test_recipe_2shuffle_fold{f}.npy"
        if not oof_p.exists():
            log(f"FATAL: {oof_p} missing — run all 5 folds first")
            return
        oof += np.load(oof_p).astype(np.float32)
        test_pred += np.load(test_p).astype(np.float32)
    log(f"summed 5 per-fold checkpoints: oof shape {oof.shape}, test shape {test_pred.shape}")

    # Sanity: every row of OOF should have probs > 0 (each row is in exactly
    # one val fold, summed across folds the row appears once).
    n_zero = (oof.sum(1) < 1e-6).sum()
    log(f"  n_zero_rows = {n_zero} (expect 0 if all 5 folds completed)")
    n_double = (oof.sum(1) > 1.5).sum()
    log(f"  n_double_assigned_rows = {n_double} (expect 0)")

    # Save final aggregated arrays
    final_oof = ART / "oof_recipe_2shuffle.npy"
    final_test = ART / "test_recipe_2shuffle.npy"
    np.save(final_oof, oof)
    np.save(final_test, test_pred)
    log(f"saved final {final_oof} + {final_test}")

    # Standalone diagnostic
    from sklearn.metrics import balanced_accuracy_score
    argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"\n=== N3 (K=2 shuffle concat) standalone ===")
    log(f"  argmax bal = {argmax_bal:.5f}")
    log(f"  tuned bal  = {tuned:.5f}  bias = {bias.round(4).tolist()}")

    # Compare vs recipe baseline
    recipe_oof = np.load(ART / "oof_recipe_full_te.npy").astype(np.float32)
    recipe_argmax = balanced_accuracy_score(y, recipe_oof.argmax(1))
    log(f"  recipe baseline argmax = {recipe_argmax:.5f}  Δ = {argmax_bal - recipe_argmax:+.5f}")

    # Errors / Jaccard vs LB-best 4-stack
    log("\n=== blend gate vs LB-best 4-stack (fixed bias) ===")
    lb3_o, lb3_t = build_lbbest_stack(y)
    xgb_oof = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    xgb_test = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    xgb_iso_o, xgb_iso_t = iso_cal(xgb_oof, xgb_test, y)
    lb4_o = log_blend([lb3_o, xgb_iso_o], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, xgb_iso_t], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_o, y)
    log(f"  LB-best 4-stack OOF = {lb4_bal:.5f}")

    # iso-cal N3 first (matches recipe family calibration)
    iso_o, iso_t = iso_cal(oof, test_pred, y)
    iso_tuned = bal(iso_o, y)
    log(f"  N3 iso-cal tuned bal = {iso_tuned:.5f}")

    # Errors + Jaccard
    pred_lb = (np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS).argmax(1)
    pred_n3_iso = (np.log(np.clip(iso_o, 1e-12, 1)) + BIAS).argmax(1)
    errs_lb = int((pred_lb != y).sum())
    errs_n3 = int((pred_n3_iso != y).sum())
    inter = int(((pred_lb != y) & (pred_n3_iso != y)).sum())
    union = int(((pred_lb != y) | (pred_n3_iso != y)).sum())
    jacc = inter / max(union, 1)
    log(f"\n  errs LB-best = {errs_lb}  N3_iso = {errs_n3}  Jaccard = {jacc:.4f}")

    # Per-class recall
    def pcr(p):
        pred = (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)
        return [(pred[y == c] == c).mean() for c in range(3)]

    pcr_lb = pcr(lb4_o)
    pcr_n3 = pcr(iso_o)
    log(f"  PCR LB-best = [L {pcr_lb[0]:.4f} M {pcr_lb[1]:.4f} H {pcr_lb[2]:.4f}]")
    log(f"  PCR N3_iso  = [L {pcr_n3[0]:.4f} M {pcr_n3[1]:.4f} H {pcr_n3[2]:.4f}]")

    # Blend sweep
    log(f"\n=== fixed-bias blend sweep: N3_iso × LB-best 4-stack ===")
    log(f"{'α':>6} {'OOF':>9} {'Δ':>9} {'errs':>6}")
    rows = []
    for a in (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5):
        blend = log_blend([lb4_o, iso_o], np.array([1 - a, a]))
        b = bal(blend, y)
        d = b - lb4_bal
        e_blend = int(((np.log(np.clip(blend, 1e-12, 1)) + BIAS).argmax(1) != y).sum())
        rows.append({"alpha": float(a), "oof": float(b), "delta": float(d), "errs": e_blend})
        tag = " *" if d > 1e-4 else ""
        log(f"{a:>6.3f} {b:>9.5f} {d:>+9.5f} {e_blend:>6}{tag}")
    best = max(rows, key=lambda r: r["delta"])

    # Emit if Δ ≥ +2e-4 — but per CLAUDE.md submission rule, ASK USER before LB probe
    if best["delta"] >= 2e-4:
        a = best["alpha"]
        test_blend = log_blend([lb4_t, iso_t], np.array([1 - a, a]))
        pred_t = (np.log(np.clip(test_blend, 1e-12, 1)) + BIAS).argmax(1)
        sample = pd.read_csv(DATA / "sample_submission.csv")
        sub = sample.copy()
        sub[TARGET] = [CLASSES[i] for i in pred_t]
        path = SUB / f"submission_n3_2shuffle_a{int(a*100):03d}.csv"
        sub.to_csv(path, index=False)
        log(f"\nΔ +{best['delta']:.5f} ≥ +2e-4 → wrote {path} (NOT submitted)")
    else:
        log(f"\nbest Δ +{best['delta']:.5f} below +2e-4 gate — no submission")

    out = dict(
        n_shuffle=2, n_folds=5,
        argmax_bal=float(argmax_bal), tuned_bal=float(tuned),
        iso_tuned_bal=float(iso_tuned), bias=bias.tolist(),
        errs_lb=errs_lb, errs_n3=errs_n3, jaccard=float(jacc),
        pcr_lb=pcr_lb, pcr_n3=pcr_n3,
        recipe_baseline_argmax=float(recipe_argmax),
        lb4_baseline_oof=float(lb4_bal),
        blend_sweep=rows, best=best,
    )
    (ART / "n3_2shuffle_results.json").write_text(json.dumps(out, indent=2))
    log("wrote scripts/artifacts/n3_2shuffle_results.json")


if __name__ == "__main__":
    main()
