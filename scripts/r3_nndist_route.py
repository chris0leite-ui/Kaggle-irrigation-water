"""R3 — Confident-row asymmetric override using NN-dist standalone.

Replaces Option 2's sparse `frac_high_neighbors` gate with NN-dist's
STANDALONE OOF max_prob (which is dense, well-calibrated, and uses
all 5 distance features through the trained recipe XGB).

Mechanism (per row):
  if nndist_max_prob[i] > τ AND primary_argmax[i] != nndist_argmax[i]:
    → flip primary[i] to nndist_argmax[i]

Tune τ on OOF over a coarse grid; pick the gate-passing config with
the maximum OOF Δ. Per-class recall guardrail: each class ≥ anchor
floor − 5e-4. Emit submission for the best config (and a few near-
best configs at different τ values for an LB probe-set).

Distinct from prior Option 2 (which used the 16-NN frac_high feature
directly — too sparse, p95 = 0.125). NN-dist's softprob through the
recipe XGB is dense and learned, encoding all 5 raw distance features
plus interaction with recipe's other 440 features.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, SUB, build_lbbest_stack, iso_cal, load_y, log, normed,
)

GUARDRAIL = 5e-4
EMIT_DELTA = 1e-4
TAU_GRID = np.array([0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94,
                     0.95, 0.96, 0.97, 0.98, 0.99, 0.995])


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return np.diag(cm).astype(float) / cm.sum(axis=1).clip(1)


def predict_with_bias(p):
    return (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)


def apply_override(primary_pred: np.ndarray, nndist_p: np.ndarray, tau: float):
    """Flip primary[i] to nndist_argmax[i] where nndist_max_prob > tau AND
    they disagree. Returns the new prediction array + override count + per-direction breakdown."""
    nnd_argmax = predict_with_bias(nndist_p)
    nnd_max = nndist_p.max(axis=1)
    mask = (nnd_max > tau) & (primary_pred != nnd_argmax)
    pred = primary_pred.copy()
    pred[mask] = nnd_argmax[mask]
    # breakdown by direction
    direction = {}
    for src in (0, 1, 2):
        for dst in (0, 1, 2):
            if src == dst: continue
            n = int(((primary_pred == src) & (nnd_argmax == dst) & mask).sum())
            direction[f"{src}->{dst}"] = n
    return pred, int(mask.sum()), direction


def sweep(primary_o, nndist_o, y, anchor_bal, anchor_pcr):
    rows = []
    primary_pred = predict_with_bias(primary_o)
    for tau in TAU_GRID:
        new_pred, n_ov, dir_ = apply_override(primary_pred, nndist_o, float(tau))
        bal = balanced_accuracy_score(y, new_pred)
        pcr = per_class_recall(y, new_pred)
        guard = all(pcr[c] >= anchor_pcr[c] - GUARDRAIL for c in range(3))
        rows.append(dict(
            tau=float(tau),
            n_overrides=n_ov,
            direction=dir_,
            bal=float(bal),
            delta=float(bal - anchor_bal),
            pcr=pcr.tolist(),
            pcr_pass=bool(guard),
        ))
    return rows


def emit_submission(tau, primary_t, nndist_t, ids):
    primary_pred_t = predict_with_bias(primary_t)
    new_pred_t, n_ov, dir_ = apply_override(primary_pred_t, nndist_t, tau)
    cls_map = {0: "Low", 1: "Medium", 2: "High"}
    sub = pd.DataFrame({"id": ids, "Irrigation_Need": [cls_map[i] for i in new_pred_t]})
    p = SUB / f"submission_r3_nndist_route_t{int(tau*1000):03d}.csv"
    sub.to_csv(p, index=False)
    log(f"  EMIT τ={tau:.3f}  test overrides={n_ov}  "
        f"dir={dir_}  pred dist: {dict(sub['Irrigation_Need'].value_counts())}")
    return str(p), n_ov, dir_


def main():
    t0 = time.time()
    y = load_y()
    log("loading LB-best 4-stack PRIMARY (3-stack + meta_v1_iso α=0.30)")
    lb3_o, lb3_t = build_lbbest_stack(y)
    m_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    m_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    m_o_iso, m_t_iso = iso_cal(m_o, m_t, y)
    primary_o = log_blend([lb3_o, m_o_iso], np.array([0.70, 0.30]))
    primary_t = log_blend([lb3_t, m_t_iso], np.array([0.70, 0.30]))

    primary_pred = predict_with_bias(primary_o)
    anchor_bal = balanced_accuracy_score(y, primary_pred)
    anchor_pcr = per_class_recall(y, primary_pred)
    log(f"  primary OOF = {anchor_bal:.5f}  PCR L={anchor_pcr[0]:.4f} M={anchor_pcr[1]:.4f} H={anchor_pcr[2]:.4f}")

    log("loading nndist standalone OOF/test")
    nnd_o = normed(np.load(ART / "oof_recipe_full_te_nndist.npy").astype(np.float32))
    nnd_t = normed(np.load(ART / "test_recipe_full_te_nndist.npy").astype(np.float32))
    nnd_max_oof = nnd_o.max(axis=1)
    nnd_max_test = nnd_t.max(axis=1)
    log(f"  nndist max_prob percentiles (OOF): "
        f"50={np.percentile(nnd_max_oof,50):.4f} "
        f"75={np.percentile(nnd_max_oof,75):.4f} "
        f"90={np.percentile(nnd_max_oof,90):.4f} "
        f"95={np.percentile(nnd_max_oof,95):.4f} "
        f"99={np.percentile(nnd_max_oof,99):.4f}")
    log(f"  nndist max_prob percentiles (test): "
        f"50={np.percentile(nnd_max_test,50):.4f} "
        f"90={np.percentile(nnd_max_test,90):.4f} "
        f"99={np.percentile(nnd_max_test,99):.4f}")

    # Disagreement count at τ=0 (just AND on argmax-disagree)
    nnd_argmax = predict_with_bias(nnd_o)
    n_disagree = int((primary_pred != nnd_argmax).sum())
    log(f"  primary != nndist argmax on {n_disagree:,} OOF rows ({n_disagree/len(y)*100:.2f}%)")

    log("=== sweep ===")
    rows = sweep(primary_o, nnd_o, y, anchor_bal, anchor_pcr)
    log(f"{'tau':>6} {'n_ov':>6} {'bal':>9} {'Δ':>9} {'PCR_L':>7} {'PCR_M':>7} {'PCR_H':>7} {'guard':>6}")
    for r in rows:
        log(f"{r['tau']:>6.3f} {r['n_overrides']:>6d} {r['bal']:>9.5f} "
            f"{r['delta']:>+9.5f} {r['pcr'][0]:>7.4f} {r['pcr'][1]:>7.4f} {r['pcr'][2]:>7.4f} "
            f"{'PASS' if r['pcr_pass'] else 'FAIL':>6}")

    # Find best gate-passing config + a couple at different τ for LB probe set
    passing = [r for r in rows if r["pcr_pass"]]
    passing.sort(key=lambda r: r["delta"], reverse=True)

    summary = dict(
        anchor_bal=float(anchor_bal),
        anchor_pcr=anchor_pcr.tolist(),
        n_disagree_oof=n_disagree,
        sweep=rows,
        emitted=[],
    )

    ids = pd.read_csv("data/test.csv")["id"].values
    if passing:
        log(f"top 3 gate-passing:")
        seen_n_ov = set()
        for r in passing[:5]:
            log(f"  τ={r['tau']:.3f}  Δ={r['delta']:+.5f}  n_ov={r['n_overrides']}")

        # Emit best Δ and one larger-override config (more LB-test surface)
        best = passing[0]
        if best["delta"] >= EMIT_DELTA:
            path, n_ov, dir_ = emit_submission(best["tau"], primary_t, nnd_t, ids)
            summary["emitted"].append(dict(tau=best["tau"], delta=best["delta"],
                                           n_overrides_test=n_ov, direction=dir_,
                                           submission=path))
        # Also emit the largest-override gate-passing config (more aggressive)
        big = max(passing, key=lambda r: r["n_overrides"])
        if big != best and big["delta"] >= 0:
            path, n_ov, dir_ = emit_submission(big["tau"], primary_t, nnd_t, ids)
            summary["emitted"].append(dict(tau=big["tau"], delta=big["delta"],
                                           n_overrides_test=n_ov, direction=dir_,
                                           submission=path,
                                           note="larger-override variant"))
    else:
        log("no gate-passing configs")

    out = ART / "r3_nndist_route_results.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    log(f"wrote {out}")
    log(f"total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
