"""Full blend-gate + LB-projection diagnostic for RF natural.

Beyond Phase 3's monotone-α sweep at fixed argmax (no retune), this
script:
  1. Tunes bias on the blend OOF for each (anchor × α). The tuned
     blend metric is the apples-to-apples comparison vs anchor's
     own tuned standalone.
  2. Reports per-class recall delta + Jaccard + test disagreement
     count vs every relevant anchor: rawashishsin v3, Phase 2
     geomean, LB-best primary (4-stack), recipe XGB.
  3. Emits candidate CSV for RF natural STANDALONE at its own
     tuned bias.
  4. Emits candidate CSV for any blend (anchor × RF) that clears
     +2e-4 OOF lift AND PCR ≥ -5e-4 each class.

LB-projection uses the documented gap pattern for each anchor
family. NOT a hard prediction — just a sanity check on whether
the candidate lands plausibly above LB-best primary 0.98094.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, log_blend, tune_log_bias  # noqa: E402
from tier1b_helpers import build_lbbest_stack, iso_cal  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

META_SUFFIX = os.environ.get("META_SUFFIX", "")  # "" = LB-validated; "_a1lgbm" = extended bank


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
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    log(f"loading RF natural{META_SUFFIX} + anchors")
    rf_oof = np.load(ART / f"oof_sklearn_rf_meta_natural{META_SUFFIX}.npy").astype(np.float32)
    rf_test = np.load(ART / f"test_sklearn_rf_meta_natural{META_SUFFIX}.npy").astype(np.float32)
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_test = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)

    # Phase 2 geomean (natural CB variant suffix)
    geom_p = ART / "oof_blend_natural_geomean_natural.npy"
    geom_t = ART / "test_blend_natural_geomean_natural.npy"
    if geom_p.exists():
        geom_oof = np.load(geom_p).astype(np.float32)
        geom_test = np.load(geom_t).astype(np.float32)
    else:
        geom_oof = geom_test = None

    # Reconstruct LB-best primary (4-stack: lb3 + meta_iso α=0.30)
    log("reconstructing LB-best primary (4-stack)")
    lb3_oof, lb3_test = build_lbbest_stack(y)
    meta_o = np.load(ART / "oof_xgb_metastack.npy").astype(np.float32)
    meta_t = np.load(ART / "test_xgb_metastack.npy").astype(np.float32)
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    primary_oof = log_blend([lb3_oof, meta_o_iso], np.array([0.7, 0.3]))
    primary_test = log_blend([lb3_test, meta_t_iso], np.array([0.7, 0.3]))

    # Tuned standalone bal_accs
    rf_bias, rf_tuned = tune_log_bias(rf_oof, y, prior)
    raw_bias, raw_tuned = tune_log_bias(raw_oof, y, prior)
    primary_bias, primary_tuned = tune_log_bias(primary_oof, y, prior)
    if geom_oof is not None:
        geom_bias, geom_tuned = tune_log_bias(geom_oof, y, prior)

    log(f"standalone OOF tuned bal_acc:")
    log(f"  RF natural:        {rf_tuned:.5f}  bias={rf_bias.round(4).tolist()}")
    log(f"  rawashishsin v3:   {raw_tuned:.5f}  bias={raw_bias.round(4).tolist()}")
    log(f"  LB-best primary:   {primary_tuned:.5f}  bias={primary_bias.round(4).tolist()}")
    if geom_oof is not None:
        log(f"  Phase 2 geomean:   {geom_tuned:.5f}  bias={geom_bias.round(4).tolist()}")

    # Standalone PCR
    rf_pred = (safelog(rf_oof) + rf_bias).argmax(1)
    rf_pcr = per_class_recall(y, rf_pred)
    raw_pred = (safelog(raw_oof) + raw_bias).argmax(1)
    raw_pcr = per_class_recall(y, raw_pred)
    primary_pred = (safelog(primary_oof) + primary_bias).argmax(1)
    primary_pcr = per_class_recall(y, primary_pred)

    log(f"standalone PCR:")
    log(f"  RF natural:      L={rf_pcr[0]:.4f}  M={rf_pcr[1]:.4f}  H={rf_pcr[2]:.4f}")
    log(f"  rawashishsin:    L={raw_pcr[0]:.4f}  M={raw_pcr[1]:.4f}  H={raw_pcr[2]:.4f}")
    log(f"  LB-best primary: L={primary_pcr[0]:.4f}  M={primary_pcr[1]:.4f}  H={primary_pcr[2]:.4f}")

    rf_err_count = int((rf_pred != y).sum())
    raw_err_count = int((raw_pred != y).sum())
    primary_err_count = int((primary_pred != y).sum())
    log(f"standalone error counts:  RF={rf_err_count}  raw={raw_err_count}  primary={primary_err_count}")

    # 4-gate analysis: tuned blend vs each anchor
    results = {}
    anchors = [
        ("rawashishsin", raw_oof, raw_test, raw_pred, raw_pcr, raw_tuned),
        ("primary",      primary_oof, primary_test, primary_pred, primary_pcr, primary_tuned),
    ]
    if geom_oof is not None:
        geom_pred = (safelog(geom_oof) + geom_bias).argmax(1)
        geom_pcr = per_class_recall(y, geom_pred)
        anchors.append(("geomean", geom_oof, geom_test, geom_pred, geom_pcr, geom_tuned))

    for name, a_oof, a_test, a_pred, a_pcr, a_tuned in anchors:
        log(f"=== blend RF natural × {name} (tuned bias per blend) ===")
        sweep = []
        for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            b_oof = log_blend([a_oof, rf_oof], np.array([1.0 - alpha, alpha]))
            b_bias, b_tuned = tune_log_bias(b_oof, y, prior)
            b_pred = (safelog(b_oof) + b_bias).argmax(1)
            b_pcr = per_class_recall(y, b_pred)
            d_class = (b_pcr - a_pcr).tolist()
            d_total = float(b_tuned - a_tuned)
            errs = int((b_pred != y).sum())
            sweep.append({
                "alpha": alpha,
                "tuned": float(b_tuned),
                "bias": b_bias.tolist(),
                "delta_vs_anchor": d_total,
                "pcr_delta": d_class,
                "errs": errs,
            })
            mark = "*" if d_total >= 2e-4 and all(d >= -5e-4 for d in d_class) else " "
            log(f"  α={alpha:.2f} {mark} tuned={b_tuned:.5f} Δ={d_total:+.5f} "
                f"pcr_delta=[{d_class[0]:+.4f} {d_class[1]:+.4f} {d_class[2]:+.4f}] "
                f"errs={errs}")
        # Find best PASS-gate
        passing = [s for s in sweep if s["delta_vs_anchor"] >= 2e-4
                   and all(d >= -5e-4 for d in s["pcr_delta"])]
        if passing:
            best = max(passing, key=lambda s: s["tuned"])
            log(f"  ✓ best PASS: α={best['alpha']} Δ={best['delta_vs_anchor']:+.5f}")
            # Net rare-class flips on test
            b_test = log_blend([a_test, rf_test],
                                np.array([1.0 - best["alpha"], best["alpha"]]))
            b_test_pred = (safelog(b_test) + np.array(best["bias"])).argmax(1)
            a_test_pred = (safelog(a_test) + (raw_bias if name == "rawashishsin"
                            else primary_bias if name == "primary"
                            else geom_bias)).argmax(1)
            net_h = int(((b_test_pred == 2) & (a_test_pred != 2)).sum() -
                        ((a_test_pred == 2) & (b_test_pred != 2)).sum())
            churn_h = int(((b_test_pred == 2) ^ (a_test_pred == 2)).sum())
            disagree = int((b_test_pred != a_test_pred).sum())
            log(f"  test diff vs {name}: {disagree}, net_H={net_h:+d}  churn_H={churn_h}")
            # Emit candidate
            sub_path = SUB / f"submission_rf_natural{META_SUFFIX}_blend_{name}_a{int(best['alpha']*100):03d}.csv"
            sub = pd.DataFrame({
                "id": test_ids,
                TARGET: [IDX2CLS[i] for i in b_test_pred],
            })
            sub.to_csv(sub_path, index=False)
            log(f"  wrote {sub_path}")
            results[name] = dict(
                anchor_tuned=float(a_tuned), best=best,
                net_H=net_h, churn_H=churn_h, disagree=disagree,
                sub_path=str(sub_path),
            )
        else:
            log(f"  no PASS-gate alpha")
            results[name] = dict(anchor_tuned=float(a_tuned),
                                  best=None, sweep=sweep)

    # Standalone RF natural candidate (its own tuned bias)
    rf_test_pred_idx = (safelog(rf_test) + rf_bias).argmax(1)
    primary_test_pred = (safelog(primary_test) + primary_bias).argmax(1)
    raw_test_pred = (safelog(raw_test) + raw_bias).argmax(1)
    rf_diff_vs_primary = int((rf_test_pred_idx != primary_test_pred).sum())
    rf_diff_vs_raw = int((rf_test_pred_idx != raw_test_pred).sum())
    log(f"=== RF natural STANDALONE (own tuned bias [{', '.join(f'{b:.3f}' for b in rf_bias)}]) ===")
    log(f"  OOF tuned: {rf_tuned:.5f}")
    log(f"  test diff vs primary: {rf_diff_vs_primary}")
    log(f"  test diff vs rawashishsin: {rf_diff_vs_raw}")

    sub_path = SUB / f"submission_sklearn_rf_meta_natural{META_SUFFIX}_standalone.csv"
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in rf_test_pred_idx],
    })
    sub.to_csv(sub_path, index=False)
    log(f"  wrote {sub_path}")

    summary = dict(
        rf_natural_standalone=dict(
            oof_tuned=float(rf_tuned), bias=rf_bias.tolist(),
            errs=rf_err_count, pcr=rf_pcr.tolist(),
            test_diff_vs_primary=rf_diff_vs_primary,
            test_diff_vs_rawashishsin=rf_diff_vs_raw,
            sub_path=f"submissions/submission_sklearn_rf_meta_natural{META_SUFFIX}_standalone.csv",
        ),
        anchors=dict(
            rawashishsin_tuned=float(raw_tuned),
            primary_tuned=float(primary_tuned),
            geomean_tuned=float(geom_tuned) if geom_oof is not None else None,
        ),
        blend_gate_results=results,
    )
    out_p = ART / f"blend_gate_rf_natural_full{META_SUFFIX}_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
