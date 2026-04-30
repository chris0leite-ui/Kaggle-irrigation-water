"""R9: log-blend RF natural (LB 0.98129) × Tier 1b 4-stack (LB 0.98094).

Two structurally-orthogonal LB-validated submissions:
  RF natural (LB 0.98129):    bagging-meta on 7-component natural-cal bank
  Tier 1b 4-stack (LB 0.98094): LB-best 3-stack + xgb_metastack__iso α=0.30

Reconstruct 4-stack via tier1b_helpers + iso_cal(xgb_metastack), then
α-sweep log-blend with RF natural at fixed bias. Tune bias on each blend.

Untested combination per CLAUDE.md cross-branch saturation log. Both
LB-validated submissions; structurally different L2 architectures
(bagging vs gradient-boosted-meta).

Outputs:
  scripts/artifacts/oof_rf_x_tier1b_a{XXX}.npy
  scripts/artifacts/test_rf_x_tier1b_a{XXX}.npy
  scripts/artifacts/r9_rf_x_tier1b_results.json
  submissions/submission_rf_x_tier1b_a{XXX}.csv (if all gates pass)
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
from tier1b_helpers import build_lbbest_stack, iso_cal, normed  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
    log("loading y + test ids")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_te = len(test_ids)
    prior = np.bincount(y, minlength=3) / len(y)
    neg_log_prior = -np.log(prior)

    log("reconstructing Tier 1b 4-stack (LB 0.98094)")
    lb3_o, lb3_t = build_lbbest_stack(y)
    log(f"  LB-best 3-stack: tuned={tune_log_bias(lb3_o, y, prior)[1]:.5f}")

    # 4-stack = LB-best 3-stack + xgb_metastack__iso × α=0.30
    log("loading + iso_cal'ing xgb_metastack")
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)

    # log_blend at (0.7, 0.3)
    w = np.array([0.7, 0.3])
    s4_o = np.exp(w[0] * safelog(lb3_o) + w[1] * safelog(meta_o_iso))
    s4_o = s4_o / s4_o.sum(1, keepdims=True)
    s4_t = np.exp(w[0] * safelog(lb3_t) + w[1] * safelog(meta_t_iso))
    s4_t = s4_t / s4_t.sum(1, keepdims=True)

    bias_4s, tuned_4s = tune_log_bias(s4_o, y, prior)
    log(f"  4-stack reconstructed: tuned={tuned_4s:.5f}  bias={bias_4s.round(4).tolist()}")
    log(f"    drift={(bias_4s - neg_log_prior).round(4).tolist()}")

    log("loading RF natural (LB 0.98129)")
    rf_o = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    rf_t = normed(np.load(ART / "test_sklearn_rf_meta_natural.npy").astype(np.float32))
    bias_rf, tuned_rf = tune_log_bias(rf_o, y, prior)
    log(f"  RF natural: tuned={tuned_rf:.5f}  bias={bias_rf.round(4).tolist()}")
    pred_rf = (safelog(rf_o) + bias_rf).argmax(1)
    pcr_rf = per_class_recall(y, pred_rf)
    errs_rf = int((pred_rf != y).sum())

    log("=== α-sweep RF × 4-stack (α = RF weight) ===")
    sweep = []
    for alpha in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        log_blend_oof = alpha * safelog(rf_o) + (1 - alpha) * safelog(s4_o)
        blend_oof = np.exp(log_blend_oof - log_blend_oof.max(1, keepdims=True))
        blend_oof = blend_oof / blend_oof.sum(1, keepdims=True)

        log_blend_te = alpha * safelog(rf_t) + (1 - alpha) * safelog(s4_t)
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
        test_pred_rf = (safelog(rf_t) + bias_rf).argmax(1)
        rows_diff = int((test_pred != test_pred_rf).sum())
        net_h = int(((test_pred == 2) & (test_pred_rf != 2)).sum() -
                    ((test_pred_rf == 2) & (test_pred != 2)).sum())
        churn_h = int(((test_pred == 2) ^ (test_pred_rf == 2)).sum())
        asym = abs(net_h) / max(churn_h, 1)

        g1 = delta_tuned >= 2e-4
        g2 = all(d >= -5e-4 for d in delta_pcr)
        g4_dir = net_h > 0
        g4_asym = asym >= 0.5

        log(f"  α_RF={alpha:.2f}: tuned={tuned:.5f} Δ={delta_tuned:+.5f}  "
            f"errs={errs} (Δ{delta_errs:+d})  drift_max={max_drift:.3f}  "
            f"PCR Δ=[{delta_pcr[0]:+.5f},{delta_pcr[1]:+.5f},{delta_pcr[2]:+.5f}]  "
            f"diff={rows_diff} net_H={net_h:+d} asym={asym:.2f}  "
            f"{'G1+' if g1 else 'G1-'}{'G2+' if g2 else 'G2-'}"
            f"{'G4+' if (g4_dir and g4_asym) else 'G4-'}")

        sweep.append(dict(
            alpha=alpha, tuned=float(tuned), delta_tuned=float(delta_tuned),
            bias=bias.tolist(), drift=drift.tolist(),
            max_abs_drift=max_drift,
            pcr=pcr.tolist(), delta_pcr=delta_pcr,
            errs=errs, delta_errs=delta_errs,
            test_rows_diff=rows_diff, net_h=net_h, churn_h=churn_h, asym=asym,
            g1=g1, g2=g2, g4_direction=g4_dir, g4_asymmetry=g4_asym,
        ))

    # Find best gate-pass
    best = None
    for s in sweep:
        if s["g1"] and s["g2"] and s["g4_direction"] and s["g4_asymmetry"]:
            if best is None or s["delta_tuned"] > best["delta_tuned"]:
                best = s
    if best is None:
        best = max(sweep, key=lambda s: s["delta_tuned"])
        log(f"\n  NO gate-passing variant; reporting best by Δ tuned (α={best['alpha']:.2f})")
    else:
        log(f"\n  best gate-passing α={best['alpha']:.2f}  Δ={best['delta_tuned']:+.5f}")

    summary = dict(
        rf_tuned=float(tuned_rf), rf_bias=bias_rf.tolist(),
        tier1b_4stack_tuned=float(tuned_4s), tier1b_4stack_bias=bias_4s.tolist(),
        sweep=sweep, best_alpha=best["alpha"],
        best_delta_tuned=best["delta_tuned"],
        best_passes_all_gates=(best["g1"] and best["g2"] and
                                best["g4_direction"] and best["g4_asymmetry"]),
    )
    out_p = ART / "r9_rf_x_tier1b_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"\nwrote {out_p}")

    # Save best blend OOF + test
    a = best["alpha"]
    log_blend_oof = a * safelog(rf_o) + (1 - a) * safelog(s4_o)
    blend_oof = np.exp(log_blend_oof - log_blend_oof.max(1, keepdims=True))
    blend_oof = blend_oof / blend_oof.sum(1, keepdims=True)
    log_blend_te = a * safelog(rf_t) + (1 - a) * safelog(s4_t)
    blend_te = np.exp(log_blend_te - log_blend_te.max(1, keepdims=True))
    blend_te = blend_te / blend_te.sum(1, keepdims=True)
    suffix = f"_a{int(a * 100):03d}"
    np.save(ART / f"oof_rf_x_tier1b_blend{suffix}.npy", blend_oof.astype(np.float32))
    np.save(ART / f"test_rf_x_tier1b_blend{suffix}.npy", blend_te.astype(np.float32))
    log(f"saved best blend OOF + test (α={a:.2f}, suffix={suffix})")

    if best["g1"] and best["g2"] and best["g4_direction"] and best["g4_asymmetry"]:
        bias = np.array(best["bias"])
        test_pred = (safelog(blend_te) + bias).argmax(1)
        out_csv = SUB / f"submission_rf_x_tier1b{suffix}.csv"
        sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in test_pred]})
        sub.to_csv(out_csv, index=False)
        log(f"GATE-PASS candidate emitted: {out_csv}")
        log(f"  class counts: {sub[TARGET].value_counts().to_dict()}")
    else:
        log("NO gate-pass candidate")


if __name__ == "__main__":
    main()
