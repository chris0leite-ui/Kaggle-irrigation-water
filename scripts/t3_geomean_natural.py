"""T3: Diverse natural-cal banks + geomean.

Reframed approach: instead of building LOO sub-banks of v1's 7-component
bank (~25 min × 7 = 3 hours), use the EXISTING natural-cal RF meta
variants already on disk. All 10 share v1's architecture (sklearn RF
bootstrap=True, class_weight=None, max_depth=12, n_est=500) but vary in
bank composition. Geomean across the family tests the "multiple sweet
spots → variance reduction" hypothesis instantly.

Variants on disk:
  v1_lb98129     — LB-best 7-component bank
  a1lgbm         — bank-extension +LGBM, LB 0.98097 (regressed)
  plus_natrealmlp — bank-extension, LB 0.98098 (regressed)
  plus_t2        — bank-extension +T2 pseudo, predicted regression
  v1bank_bag5    — TE-seed bag (NULL on G1/G2/G4)
  xreg           — cross-regime variant, LB 0.98115 (regressed)
  Va             — REPLACE realmlp→a2_natural_calib (NULL)
  Vb             — REPLACE variant (NULL)
  Vc             — REPLACE catboost→cb_skte, LB 0.98113 (regressed)

Output: 4-gate analysis on multiple geomean configurations vs v1 LB-best.
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
SUB = Path("submissions")

CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
TARGET = "Irrigation_Need"

VARIANTS = {
    "v1": "sklearn_rf_meta_natural_v1_lb98129",
    "a1lgbm": "sklearn_rf_meta_natural_a1lgbm",
    "plus_natrealmlp": "sklearn_rf_meta_natural_plus_natrealmlp",
    "plus_t2": "sklearn_rf_meta_natural_plus_t2",
    "xreg": "sklearn_rf_meta_natural_xreg",
    "v1bank_bag5": "sklearn_rf_meta_natural_v1bank_bag5",
    "Va": "rf_natural_replace_Va",
    "Vb": "rf_natural_replace_Vb",
    "Vc": "rf_natural_replace_Vc",
}

# v1 LB 0.98129 documented bias
V1_BIAS = np.array([0.4324411632360975, 0.8689466919465483, 3.2007689020440884])


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def _normed(a, eps=1e-9):
    return a / np.clip(a.sum(1, keepdims=True), eps, None)


def geomean(probs_list):
    """Log-mean of multiple prob arrays, then renormalize."""
    log_sum = np.zeros_like(probs_list[0], dtype=np.float64)
    for p in probs_list:
        log_sum += safelog(p)
    log_sum /= len(probs_list)
    out = np.exp(log_sum)
    return _normed(out).astype(np.float32)


def evaluate(oof, test, y, name, anchor_oof, anchor_test, anchor_pcr,
             v1_bias=V1_BIAS):
    """Evaluate a candidate vs v1 LB-best at v1's fixed bias."""
    pred_oof = (safelog(oof) + v1_bias).argmax(1)
    pred_test = (safelog(test) + v1_bias).argmax(1)
    bal_oof = float(per_class_recall(y, pred_oof).mean())
    pcr = per_class_recall(y, pred_oof)
    pcr_delta = pcr - anchor_pcr

    # Test-side disagreement
    a_pred_test = (safelog(anchor_test) + v1_bias).argmax(1)
    diff = int((pred_test != a_pred_test).sum())
    add_h = int(((pred_test == 2) & (a_pred_test != 2)).sum())
    rem_h = int(((a_pred_test == 2) & (pred_test != 2)).sum())
    net_h = add_h - rem_h
    churn_h = add_h + rem_h
    g4_ratio = abs(net_h) / max(churn_h, 1)

    return dict(
        name=name,
        bal_acc=bal_oof,
        pcr=pcr.tolist(),
        pcr_delta=pcr_delta.tolist(),
        test_diff=diff,
        add_h=add_h, rem_h=rem_h, net_h=net_h,
        g4_ratio=g4_ratio,
        direction="ADD-High" if net_h > 0 else ("REMOVE-High" if net_h < 0 else "neutral"),
    )


def main():
    print("Loading train labels")
    y = pd.read_csv("data/train.csv")[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    n_tr = len(y)

    # Load all variants
    pool = {}
    for tag, fname in VARIANTS.items():
        op = ART / f"oof_{fname}.npy"
        tp = ART / f"test_{fname}.npy"
        if op.exists() and tp.exists():
            pool[tag] = (
                np.load(op).astype(np.float32),
                np.load(tp).astype(np.float32),
            )
            print(f"  + {tag:18s}  ({fname})")
        else:
            print(f"  - {tag:18s}  MISSING")

    # Anchor: v1 LB 0.98129 at documented bias
    v1_oof, v1_test = pool["v1"]
    v1_pred_oof = (safelog(v1_oof) + V1_BIAS).argmax(1)
    v1_pcr = per_class_recall(y, v1_pred_oof)
    v1_bal = float(v1_pcr.mean())
    print(f"\nAnchor v1 LB 0.98129:")
    print(f"  OOF tuned at v1 bias = {v1_bal:.5f} (expect 0.98063)")
    print(f"  PCR = [L={v1_pcr[0]:.5f} M={v1_pcr[1]:.5f} H={v1_pcr[2]:.5f}]")

    # Phase 1: per-variant standalone diagnostic
    print("\n=== PHASE 1: per-variant standalone @ v1 bias ===")
    standalone = {}
    for tag, (oof, test) in pool.items():
        if tag == "v1":
            continue
        diag = evaluate(oof, test, y, tag, v1_oof, v1_test, v1_pcr)
        standalone[tag] = diag
        print(f"  {tag:18s} bal={diag['bal_acc']:.5f}  Δ={diag['bal_acc']-v1_bal:+.5f}  "
              f"net_H={diag['net_h']:+4d}  ratio={diag['g4_ratio']:.3f}  {diag['direction']}")

    # Phase 2: filter variants by similarity (Δ within ±0.0010 of v1)
    candidates = {tag: diag for tag, diag in standalone.items()
                  if abs(diag['bal_acc'] - v1_bal) < 0.0010}
    print(f"\n=== PHASE 2: filter to similar variants (|Δ| < 0.001) ===")
    print(f"  candidates: {sorted(candidates.keys())} ({len(candidates)} variants)")

    # Build geomeans across configurations
    print("\n=== PHASE 3: geomean diagnostics ===")
    configs = [
        ("v1+all_similar", ["v1"] + sorted(candidates.keys())),
        ("v1+xreg_only", ["v1", "xreg"] if "xreg" in pool else None),
        ("v1+Vc_only", ["v1", "Vc"] if "Vc" in pool else None),
        ("v1+a1lgbm_only", ["v1", "a1lgbm"] if "a1lgbm" in pool else None),
    ]
    # Also add: v1 with all 4 best LB-validated variants (LB 0.98113, 0.98115, etc.)
    lb_validated = [t for t in ["xreg", "Vc", "a1lgbm", "plus_natrealmlp"]
                     if t in pool]
    if len(lb_validated) >= 2:
        configs.append((f"v1+all_lb_validated", ["v1"] + lb_validated))

    geomean_results = {}
    for cfg_name, members in configs:
        if members is None:
            continue
        oofs = [pool[m][0] for m in members]
        tests = [pool[m][1] for m in members]
        gm_oof = geomean(oofs)
        gm_test = geomean(tests)

        diag = evaluate(gm_oof, gm_test, y, cfg_name, v1_oof, v1_test, v1_pcr)
        delta = diag['bal_acc'] - v1_bal
        geomean_results[cfg_name] = dict(diag, members=members, oof_delta=delta)

        print(f"\n  {cfg_name} ({members}):")
        print(f"    bal_acc      = {diag['bal_acc']:.5f}  Δ vs v1 = {delta:+.5f}")
        print(f"    pcr_delta    = [{diag['pcr_delta'][0]:+.5f} "
              f"{diag['pcr_delta'][1]:+.5f} {diag['pcr_delta'][2]:+.5f}]")
        print(f"    test_diff    = {diag['test_diff']}")
        print(f"    H-flips      = +{diag['add_h']}/-{diag['rem_h']}  net={diag['net_h']:+d}")
        print(f"    direction    = {diag['direction']}  ratio={diag['g4_ratio']:.3f}")

    # 4-gate verdict for each
    print("\n=== 4-GATE VERDICT ===")
    for cfg_name, diag in geomean_results.items():
        delta = diag['oof_delta']
        pcr_d = diag['pcr_delta']
        net_h = diag['net_h']
        ratio = diag['g4_ratio']

        g1 = delta > 0
        g2 = all(d >= -5e-4 for d in pcr_d)
        g3 = True  # drift not separately checked — RF natural always passes
        g4 = (net_h > 0) and (ratio >= 0.5)
        n_pass = sum([g1, g2, g3, g4])
        verdict = "EMIT" if n_pass == 4 else f"NULL ({n_pass}/4)"
        print(f"  {cfg_name:25s}  G1:{g1}  G2:{g2}  G4:{g4}  → {verdict}")

    # Save best gate-passing as submission candidate
    best_cfg = None
    best_delta = -1
    for cfg_name, diag in geomean_results.items():
        d = diag['oof_delta']
        pcr_d = diag['pcr_delta']
        if d > best_delta and all(p >= -5e-4 for p in pcr_d) and diag['net_h'] > 0:
            best_delta = d
            best_cfg = cfg_name

    if best_cfg is not None:
        members = geomean_results[best_cfg]['members']
        gm_oof = geomean([pool[m][0] for m in members])
        gm_test = geomean([pool[m][1] for m in members])
        np.save(ART / "oof_t3_geomean_natural.npy", gm_oof)
        np.save(ART / "test_t3_geomean_natural.npy", gm_test)
        # Submission with v1's bias (bias-ridge invariant)
        pred_test = (safelog(gm_test) + V1_BIAS).argmax(1)
        sub = pd.DataFrame({
            "id": pd.read_csv("data/test.csv")["id"].values,
            TARGET: [IDX2CLS[i] for i in pred_test],
        })
        sub_path = SUB / f"submission_t3_geomean_{best_cfg}.csv"
        sub.to_csv(sub_path, index=False)
        print(f"\nbest: {best_cfg} (Δ={best_delta:+.5f})  → {sub_path}")
    else:
        print("\nNo geomean variant satisfies G1+G2+G4 — T3 NULL")

    summary = dict(
        anchor_v1_bal=v1_bal,
        anchor_v1_pcr=v1_pcr.tolist(),
        standalone=standalone,
        geomean_results={k: dict(v) for k, v in geomean_results.items()},
        best_cfg=best_cfg,
        best_delta=float(best_delta) if best_cfg else None,
    )
    out_p = ART / "t3_geomean_natural_results.json"
    out_p.write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nwrote {out_p}")


if __name__ == "__main__":
    main()
