"""R6: log-blend RF natural × LightGBM-meta with α sweep.

RF natural (LB 0.98129) has macro-recall-optimal calibration
(drift_H = -0.20, the "sweet spot"). LightGBM-meta (R5) has more
accurate argmax (+0.00066) but perfect-natural-cal drift (sub-optimal).

Hypothesis: log-blend at the right α combines RF's calibration with
LGBM's argmax-discovered novel decisions. Test α ∈ {0.3, 0.4, 0.5,
0.6, 0.7} — α is RF weight, (1-α) is LGBM weight.

Each blend: tune log-bias, report tuned bal_acc + drift + per-class
recall delta vs LB 0.98129.

If best blend Δ tuned ≥ +0.0002 AND G2 PASS AND G4 PASS, emit candidate
for user-approved LB probe.

Outputs:
  scripts/artifacts/oof_rf_lgbm_blend_aXXX.npy
  scripts/artifacts/test_rf_lgbm_blend_aXXX.npy
  scripts/artifacts/rf_lgbm_blend_results.json
  submissions/submission_rf_lgbm_blend_aXXX.csv (if gates pass)
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
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def main():
    log("loading y + test ids")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_te = len(test_ids)
    prior = np.bincount(y, minlength=3) / len(y)
    neg_log_prior = -np.log(prior)
    log(f"  -log(prior) = {[round(float(p), 4) for p in neg_log_prior]}")

    log("loading RF natural (LB 0.98129) + LightGBM-meta (R5)")
    rf_oof = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    rf_test = normed(np.load(ART / "test_sklearn_rf_meta_natural.npy").astype(np.float32))
    lg_oof = normed(np.load(ART / "oof_lgbm_meta_natural.npy").astype(np.float32))
    lg_test = normed(np.load(ART / "test_lgbm_meta_natural.npy").astype(np.float32))

    bias_rf, tuned_rf = tune_log_bias(rf_oof, y, prior)
    bias_lg, tuned_lg = tune_log_bias(lg_oof, y, prior)
    pred_rf = (safelog(rf_oof) + bias_rf).argmax(1)
    pred_lg = (safelog(lg_oof) + bias_lg).argmax(1)
    pcr_rf = per_class_recall(y, pred_rf)
    pcr_lg = per_class_recall(y, pred_lg)
    errs_rf = int((pred_rf != y).sum())
    errs_lg = int((pred_lg != y).sum())
    log(f"  RF (LB 0.98129): tuned={tuned_rf:.5f}  bias={bias_rf.round(4).tolist()}  errs={errs_rf}")
    log(f"  LGBM (R5):       tuned={tuned_lg:.5f}  bias={bias_lg.round(4).tolist()}  errs={errs_lg}")

    log("=== log-blend α-sweep (α = RF weight) ===")
    sweep = []
    for alpha in [0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70]:
        log_blend_oof = alpha * safelog(rf_oof) + (1 - alpha) * safelog(lg_oof)
        # Re-normalize the log-prob blend back to a probability
        blend_oof = np.exp(log_blend_oof - log_blend_oof.max(1, keepdims=True))
        blend_oof = blend_oof / blend_oof.sum(1, keepdims=True)

        log_blend_te = alpha * safelog(rf_test) + (1 - alpha) * safelog(lg_test)
        blend_te = np.exp(log_blend_te - log_blend_te.max(1, keepdims=True))
        blend_te = blend_te / blend_te.sum(1, keepdims=True)

        bias, tuned = tune_log_bias(blend_oof, y, prior)
        drift = bias - neg_log_prior
        max_drift = float(np.abs(drift).max())
        pred = (safelog(blend_oof) + bias).argmax(1)
        pcr = per_class_recall(y, pred)
        errs = int((pred != y).sum())
        delta_tuned = tuned - tuned_rf
        delta_pcr = (pcr - pcr_rf).tolist()
        delta_errs = errs - errs_rf

        # Test side
        test_pred = (safelog(blend_te) + bias).argmax(1)
        test_pred_rf = (safelog(rf_test) + bias_rf).argmax(1)
        rows_diff = int((test_pred != test_pred_rf).sum())
        net_h = int(((test_pred == 2) & (test_pred_rf != 2)).sum() -
                    ((test_pred_rf == 2) & (test_pred != 2)).sum())
        churn_h = int(((test_pred == 2) ^ (test_pred_rf == 2)).sum())
        asym = abs(net_h) / max(churn_h, 1)

        # 4-gate verdict
        g1 = delta_tuned >= 2e-4
        g2 = all(d >= -5e-4 for d in delta_pcr)
        g4_dir = net_h > 0
        g4_asym = asym >= 0.5

        log(f"  α_RF={alpha:.2f}: tuned={tuned:.5f} Δ={delta_tuned:+.5f}  "
            f"errs={errs} (Δ{delta_errs:+d})  drift_max={max_drift:.3f}  "
            f"PCR Δ=[{delta_pcr[0]:+.5f},{delta_pcr[1]:+.5f},{delta_pcr[2]:+.5f}]  "
            f"test_diff={rows_diff} net_H={net_h:+d} asym={asym:.2f}  "
            f"G1{'+' if g1 else '-'}G2{'+' if g2 else '-'}G4{'+' if g4_dir and g4_asym else '-'}")

        sweep.append(dict(
            alpha=alpha,
            tuned=float(tuned),
            delta_tuned=float(delta_tuned),
            bias=bias.tolist(),
            drift=drift.tolist(),
            max_abs_drift=max_drift,
            pcr=pcr.tolist(),
            delta_pcr=delta_pcr,
            errs=errs,
            delta_errs=delta_errs,
            test_rows_diff=rows_diff,
            net_h=net_h,
            churn_h=churn_h,
            asymmetry=asym,
            g1=g1, g2=g2, g4_direction=g4_dir, g4_asymmetry=g4_asym,
            test_pred=test_pred.tolist() if g1 and g2 else None,
        ))

    # Find the best gate-passing variant
    best = None
    for s in sweep:
        if s["g1"] and s["g2"] and s["g4_direction"] and s["g4_asymmetry"]:
            if best is None or s["delta_tuned"] > best["delta_tuned"]:
                best = s
    if best is None:
        # Fallback: best by tuned regardless of gates
        best = max(sweep, key=lambda s: s["delta_tuned"])
        log(f"\n  NO gate-passing variant; reporting best by Δ tuned (α={best['alpha']:.2f})")
    else:
        log(f"\n  best gate-passing α={best['alpha']:.2f}  Δ={best['delta_tuned']:+.5f}")

    # Save artifacts (without giant test_pred lists to keep JSON small)
    summary = dict(
        rf_tuned=float(tuned_rf), rf_bias=bias_rf.tolist(), rf_errs=errs_rf,
        lgbm_tuned=float(tuned_lg), lgbm_bias=bias_lg.tolist(), lgbm_errs=errs_lg,
        sweep=[{k: v for k, v in s.items() if k != "test_pred"} for s in sweep],
        best_alpha=best["alpha"],
        best_delta_tuned=best["delta_tuned"],
        best_passes_all_gates=(best["g1"] and best["g2"] and best["g4_direction"] and best["g4_asymmetry"]),
    )
    out_p = ART / "rf_lgbm_blend_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"\nwrote {out_p}")

    # Save best blend OOF + test for cross-branch reuse
    a = best["alpha"]
    log_blend_oof = a * safelog(rf_oof) + (1 - a) * safelog(lg_oof)
    blend_oof = np.exp(log_blend_oof - log_blend_oof.max(1, keepdims=True))
    blend_oof = blend_oof / blend_oof.sum(1, keepdims=True)
    log_blend_te = a * safelog(rf_test) + (1 - a) * safelog(lg_test)
    blend_te = np.exp(log_blend_te - log_blend_te.max(1, keepdims=True))
    blend_te = blend_te / blend_te.sum(1, keepdims=True)
    suffix = f"_a{int(a * 100):03d}"
    np.save(ART / f"oof_rf_lgbm_blend{suffix}.npy", blend_oof.astype(np.float32))
    np.save(ART / f"test_rf_lgbm_blend{suffix}.npy", blend_te.astype(np.float32))
    log(f"saved best blend OOF + test (α={a:.2f}, suffix={suffix})")

    # Build candidate CSV if all gates pass
    if best["g1"] and best["g2"] and best["g4_direction"] and best["g4_asymmetry"]:
        bias = np.array(best["bias"])
        test_pred = (safelog(blend_te) + bias).argmax(1)
        out_csv = SUB / f"submission_rf_lgbm_blend{suffix}.csv"
        sub = pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in test_pred],
        })
        sub.to_csv(out_csv, index=False)
        log(f"GATE-PASS candidate emitted: {out_csv}")
        log(f"  class counts: {sub[TARGET].value_counts().to_dict()}")
    else:
        log("NO gate-pass candidate — see sweep table above")


if __name__ == "__main__":
    main()
