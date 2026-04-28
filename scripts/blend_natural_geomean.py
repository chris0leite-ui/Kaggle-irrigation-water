"""Phase 2: geometric-mean blend of rawashishsin v3 + Phase 1 CatBoost natural.

The bias-mismatch trap (LB 0.98049 on 2026-04-28 from log-blending
rawashishsin into our recipe-family stack at fixed recipe bias) means
calibration-PROFILE compatibility is the key requirement. Here both
inputs target the macro-recall optimum natively (rawashishsin tuned
bias [-1.36, -1.19, 0.00]; CB-natural target similar after Phase 1).

Blend mechanics:
  log p_blend = 0.5 * log p_rawashishsin + 0.5 * log p_cb_natural
  argmax(softmax(log p_blend)) — NO post-hoc bias retune (that's the
  leak channel we're avoiding)

Diagnostic: report tuned and untuned OOF bal_acc + per-class recall +
Jaccard vs LB-best primary + test-side disagreement count.

Emit submission CSV only when blend OOF (NO RETUNE) ≥ 0.98000 (a
meaningful lift over rawashishsin's CV 0.98016).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, log_blend, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def softmax_rows(z):
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)


def per_class_recall(y, pred, n_class=3):
    cm = np.zeros((n_class, n_class), dtype=np.int64)
    for k in range(n_class):
        mask = y == k
        for j in range(n_class):
            cm[k, j] = int(((pred[mask] == j)).sum())
    rec = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    return rec, cm


def main():
    log("loading rawashishsin v3 + recipe_full_te_catboost_natural")
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_test = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)
    cb_oof_p = ART / "oof_recipe_full_te_catboost_natural.npy"
    cb_test_p = ART / "test_recipe_full_te_catboost_natural.npy"
    if not (cb_oof_p.exists() and cb_test_p.exists()):
        log(f"ERROR: Phase 1 outputs missing. Run scripts/recipe_catboost_natural.py first.")
        return
    cb_oof = np.load(cb_oof_p).astype(np.float32)
    cb_test = np.load(cb_test_p).astype(np.float32)

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    log(f"  raw oof={raw_oof.shape}  cb oof={cb_oof.shape}  y={y.shape}")

    # Per-component standalone diagnostic at NO RETUNE (their natural bias).
    raw_argmax = balanced_accuracy_score(y, raw_oof.argmax(1))
    cb_argmax = balanced_accuracy_score(y, cb_oof.argmax(1))
    log(f"standalone OOF (no retune):")
    log(f"  rawashishsin argmax:  {raw_argmax:.5f}")
    log(f"  cb_natural argmax:    {cb_argmax:.5f}")

    # Tuned (post-hoc bias) bal_acc — purely diagnostic.
    prior = np.bincount(y, minlength=3) / len(y)
    raw_bias, raw_tuned = tune_log_bias(raw_oof, y, prior)
    cb_bias, cb_tuned = tune_log_bias(cb_oof, y, prior)
    log(f"  rawashishsin tuned: {raw_tuned:.5f}  bias={raw_bias.round(4).tolist()}")
    log(f"  cb_natural tuned:   {cb_tuned:.5f}  bias={cb_bias.round(4).tolist()}")

    # Calibration-compatibility check
    bias_drift_H = abs(raw_bias[2] - cb_bias[2])
    bias_drift_max = float(np.abs(raw_bias - cb_bias).max())
    log(f"  bias drift (max class): {bias_drift_max:.3f}  (High-only: {bias_drift_H:.3f})")
    if bias_drift_max > 1.5:
        log("  WARN: bias profiles differ by >1.5 on at least one class.")
        log("        Geometric mean may suffer from operating-point mismatch.")

    # Geometric mean (no retune). Each component contributes its raw
    # log-probabilities equally; argmax taken on the softmaxed sum.
    log("computing 50/50 geometric mean")
    blend_oof_unnorm = log_blend([raw_oof, cb_oof], np.array([0.5, 0.5]))
    blend_test_unnorm = log_blend([raw_test, cb_test], np.array([0.5, 0.5]))

    # No-retune evaluation
    bal_no_retune = balanced_accuracy_score(y, blend_oof_unnorm.argmax(1))
    rec_no_retune, _ = per_class_recall(y, blend_oof_unnorm.argmax(1))
    log(f"BLEND OOF (NO RETUNE): {bal_no_retune:.5f}  "
        f"PCR=[L={rec_no_retune[0]:.4f} M={rec_no_retune[1]:.4f} H={rec_no_retune[2]:.4f}]")

    # Retune diagnostic (fallback only — leak channel risk)
    blend_bias, blend_tuned = tune_log_bias(blend_oof_unnorm, y, prior)
    log(f"BLEND OOF (TUNED, diagnostic): {blend_tuned:.5f}  bias={blend_bias.round(4).tolist()}")

    # Compatibility: how close is no-retune to tuned?
    delta_retune = blend_tuned - bal_no_retune
    log(f"  delta (tuned − no-retune) = {delta_retune:.5f}")
    if abs(delta_retune) < 0.0002:
        log("  → calibration profiles are compatible (no-retune ≈ tuned).")
    else:
        log(f"  → some operating-point mismatch (retune helps by {delta_retune:.4f}).")

    # Jaccard + per-class recall vs anchors
    raw_argmax_arr = raw_oof.argmax(1)
    raw_err = raw_argmax_arr != y
    blend_argmax_arr = blend_oof_unnorm.argmax(1)
    blend_err = blend_argmax_arr != y
    jacc_raw = float((raw_err & blend_err).sum() / max(1, (raw_err | blend_err).sum()))
    log(f"  Jaccard(blend_no_retune, rawashishsin) = {jacc_raw:.4f}")

    # Test disagreement
    raw_test_pred = raw_test.argmax(1)
    blend_test_pred = blend_test_unnorm.argmax(1)
    disagree_raw = int((raw_test_pred != blend_test_pred).sum())
    log(f"  test-row diff vs rawashishsin: {disagree_raw}/{len(test_ids)}")

    # Optionally compare against current PRIMARY (LB-best 4-stack reconstructed)
    try:
        from tier1b_helpers import build_lbbest_stack, iso_cal
        lb3_oof, lb3_test = build_lbbest_stack(y)
        # primary = 0.7 × lb3 + 0.3 × xgb_metastack_iso
        meta_o = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
        meta_t = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
        meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
        primary_oof = log_blend([lb3_oof, meta_o_iso], np.array([0.7, 0.3]))
        primary_test = log_blend([lb3_test, meta_t_iso], np.array([0.7, 0.3]))
        # Apply LB-validated recipe bias
        recipe_bias = np.array([1.4324, 1.4689, 3.4008])
        primary_pred = (safelog(primary_oof) + recipe_bias).argmax(1)
        primary_test_pred = (safelog(primary_test) + recipe_bias).argmax(1)
        primary_bal = fast_bal_acc(y, primary_pred)
        log(f"  PRIMARY (LB 0.98094) OOF tuned = {primary_bal:.5f}")
        primary_err = primary_pred != y
        jacc_primary = float((primary_err & blend_err).sum() / max(1, (primary_err | blend_err).sum()))
        log(f"  Jaccard(blend_no_retune, PRIMARY) = {jacc_primary:.4f}")
        disagree_primary = int((primary_test_pred != blend_test_pred).sum())
        log(f"  test-row diff vs PRIMARY: {disagree_primary}/{len(test_ids)}")
    except Exception as e:
        log(f"  could not reconstruct PRIMARY: {e}")
        primary_bal = None
        jacc_primary = None
        disagree_primary = None

    # Save blend OOF/test for downstream Phase 3
    np.save(ART / "oof_blend_natural_geomean.npy", blend_oof_unnorm.astype(np.float32))
    np.save(ART / "test_blend_natural_geomean.npy", blend_test_unnorm.astype(np.float32))

    # Submission emit (gated on no-retune ≥ 0.98000)
    sub_emit_path = SUB / "submission_blend_natural_geomean.csv"
    if bal_no_retune >= 0.98000:
        sub = pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in blend_test_pred],
        })
        sub.to_csv(sub_emit_path, index=False)
        log(f"  wrote {sub_emit_path}")
        # Also emit retuned variant as separate file (for diagnostic comparison)
        retuned_test_pred = (safelog(blend_test_unnorm) + blend_bias).argmax(1)
        sub_ret = pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in retuned_test_pred],
        })
        sub_ret_path = SUB / "submission_blend_natural_geomean_retuned.csv"
        sub_ret.to_csv(sub_ret_path, index=False)
        log(f"  wrote {sub_ret_path}  (retuned variant — leak risk, not deploy default)")
    else:
        log(f"  blend OOF {bal_no_retune:.5f} < 0.98000 gate — no submission emitted.")

    summary = dict(
        rawashishsin_oof_argmax=float(raw_argmax),
        rawashishsin_oof_tuned=float(raw_tuned),
        rawashishsin_bias=raw_bias.tolist(),
        cb_natural_oof_argmax=float(cb_argmax),
        cb_natural_oof_tuned=float(cb_tuned),
        cb_natural_bias=cb_bias.tolist(),
        bias_drift_max=bias_drift_max,
        bias_drift_H=float(bias_drift_H),
        blend_oof_no_retune=float(bal_no_retune),
        blend_per_class_recall=rec_no_retune.tolist(),
        blend_oof_tuned=float(blend_tuned),
        blend_bias_tuned=blend_bias.tolist(),
        delta_tuned_minus_no_retune=float(delta_retune),
        jaccard_vs_rawashishsin=jacc_raw,
        test_disagree_vs_rawashishsin=disagree_raw,
        primary_oof_tuned=primary_bal,
        jaccard_vs_primary=jacc_primary,
        test_disagree_vs_primary=disagree_primary,
        emitted_submission=bool(bal_no_retune >= 0.98000),
    )
    out_p = ART / "blend_natural_geomean_results.json"
    out_p.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
