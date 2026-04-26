"""Option 2 — NN-dist conditional override on rule-disagreement rows.

Use the 5 FAISS distance features from `oof_nn_dist_features.npy`
(scripts/nn_distance_features.py): [dist_min, dist_mean, frac_low,
frac_med, frac_high] of k=16 NN to the 10k rule-perfect original.

Diagnostic from the build step (CLAUDE.md 2026-04-26):
  Low rows    72.5% same-class neighbors in orig
  Medium rows 46.1% same-class
  High rows   16.3% same-class    ← rule rarely says High near these
                                    rows; the model has to LEARN the
                                    flip from non-rule features.

Override mechanism (per row):
  Branch HtoNotH: anchor_argmax = High AND frac_high_neighbors < τL
                  → flip to argmax(anchor.softmax with High masked).
                  Says: "model says High but rule-perfect orig has no
                        High neighbors near here — likely Medium".
  Branch NotHtoH: anchor_argmax != High AND frac_high_neighbors > τH
                  AND nndist_oof.argmax == High
                  → flip to High. Says: "model says Med/Low but
                        rule-perfect neighbors are mostly High AND
                        the NN-dist standalone agrees".

Per-row hard override (NOT probability blend) — sidesteps the
Pareto-frontier per-class trade by surgically targeting boundary
rows. If the override count is small (~few hundred test rows), the
LB delta is bounded above ~+0.0005 best case, below ~−0.0003 worst.

Tune (τL, τH) on OOF over a coarse grid; emit submission for the
best gate-passing config (Δ ≥ +1e-4 AND per-class recall preserved
within 5e-4 each class).
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

# Coarse grid; finer probe after we see what wins.
TAU_L_GRID = np.arange(0.00, 0.40, 0.05)  # τL: max frac_high to flip H→¬H
TAU_H_GRID = np.arange(0.30, 1.01, 0.10)  # τH: min frac_high to flip ¬H→H


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return np.diag(cm).astype(float) / cm.sum(axis=1).clip(1)


def predict_anchor(anchor_p):
    return (np.log(np.clip(anchor_p, 1e-12, 1)) + BIAS).argmax(1)


def apply_override(anchor_p, frac_high, nndist_argmax,
                   tauL: float | None, tauH: float | None):
    """Return predictions with overrides applied.

    tauL = None → branch HtoNotH disabled.
    tauH = None → branch NotHtoH disabled.
    """
    pred = predict_anchor(anchor_p)
    pred = pred.copy()
    overrides = dict(HtoNotH=0, NotHtoH=0)

    if tauL is not None:
        # Anchor says High AND frac_high is low → flip to next-best non-High.
        h_mask = (pred == 2) & (frac_high < tauL)
        if h_mask.any():
            # argmax over Low/Med columns of anchor (with bias added).
            biased = np.log(np.clip(anchor_p, 1e-12, 1)) + BIAS
            biased_no_h = biased[h_mask].copy()
            biased_no_h[:, 2] = -np.inf
            new_pred = biased_no_h.argmax(1)
            pred[h_mask] = new_pred
            overrides["HtoNotH"] = int(h_mask.sum())

    if tauH is not None:
        # Anchor says Med/Low AND frac_high is high AND nndist agrees High.
        nh_mask = (pred != 2) & (frac_high > tauH) & (nndist_argmax == 2)
        if nh_mask.any():
            pred[nh_mask] = 2
            overrides["NotHtoH"] = int(nh_mask.sum())

    return pred, overrides


def sweep_grid(anchor_o, frac_high_oof, nndist_argmax_oof, y,
               anchor_bal, anchor_pcr):
    """Joint grid sweep over (τL, τH). Each cell evaluates the
    override in BOTH directions simultaneously; we also probe each
    direction alone for clean attribution."""
    rows = []
    log(f"sweeping {len(TAU_L_GRID)} τL × {len(TAU_H_GRID)} τH = "
        f"{len(TAU_L_GRID)*len(TAU_H_GRID)} cells (+ singletons)")

    # Direction A only
    for tL in TAU_L_GRID:
        if tL == 0:
            continue  # tL=0 means never override
        pred, ov = apply_override(anchor_o, frac_high_oof, nndist_argmax_oof,
                                  float(tL), None)
        bal = balanced_accuracy_score(y, pred)
        pcr = per_class_recall(y, pred)
        rows.append(dict(mode="HtoNotH-only", tauL=float(tL), tauH=None,
                         bal=float(bal), delta=float(bal - anchor_bal),
                         pcr=pcr.tolist(),
                         pcr_pass=bool(all(pcr[c] >= anchor_pcr[c] - GUARDRAIL
                                           for c in range(3))),
                         overrides=ov))

    # Direction B only
    for tH in TAU_H_GRID:
        if tH >= 1.0:
            continue
        pred, ov = apply_override(anchor_o, frac_high_oof, nndist_argmax_oof,
                                  None, float(tH))
        bal = balanced_accuracy_score(y, pred)
        pcr = per_class_recall(y, pred)
        rows.append(dict(mode="NotHtoH-only", tauL=None, tauH=float(tH),
                         bal=float(bal), delta=float(bal - anchor_bal),
                         pcr=pcr.tolist(),
                         pcr_pass=bool(all(pcr[c] >= anchor_pcr[c] - GUARDRAIL
                                           for c in range(3))),
                         overrides=ov))

    # Joint sweep
    for tL in TAU_L_GRID:
        for tH in TAU_H_GRID:
            if tL == 0 and tH >= 1.0:
                continue
            pred, ov = apply_override(anchor_o, frac_high_oof,
                                      nndist_argmax_oof, float(tL), float(tH))
            bal = balanced_accuracy_score(y, pred)
            pcr = per_class_recall(y, pred)
            rows.append(dict(mode="joint", tauL=float(tL), tauH=float(tH),
                             bal=float(bal), delta=float(bal - anchor_bal),
                             pcr=pcr.tolist(),
                             pcr_pass=bool(all(pcr[c] >= anchor_pcr[c] - GUARDRAIL
                                               for c in range(3))),
                             overrides=ov))

    return rows


def emit_submission(tag, anchor_t, frac_high_test, nndist_argmax_test,
                    tauL, tauH, ids):
    pred, ov = apply_override(anchor_t, frac_high_test, nndist_argmax_test,
                              tauL, tauH)
    cls_map = {0: "Low", 1: "Medium", 2: "High"}
    sub = pd.DataFrame({"id": ids, "Irrigation_Need": [cls_map[i] for i in pred]})
    name = f"opt2_route_{tag}"
    if tauL is not None:
        name += f"_tL{int(tauL*100):02d}"
    if tauH is not None:
        name += f"_tH{int(tauH*100):02d}"
    p = SUB / f"submission_{name}.csv"
    sub.to_csv(p, index=False)
    log(f"  EMITTED {p}  test overrides {ov}  pred dist: "
        f"{dict(sub['Irrigation_Need'].value_counts())}")
    return str(p)


def main():
    t0 = time.time()
    y = load_y()
    log("loading LB-best 4-stack anchor (3-stack ⊗ xgb_metastack_iso α=0.30)")
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    anchor_o = log_blend([lb3_o, meta_o_iso], np.array([0.70, 0.30]))
    anchor_t = log_blend([lb3_t, meta_t_iso], np.array([0.70, 0.30]))

    pred_anc = predict_anchor(anchor_o)
    anchor_bal = balanced_accuracy_score(y, pred_anc)
    anchor_pcr = per_class_recall(y, pred_anc)
    log(f"  anchor OOF @ recipe bias = {anchor_bal:.5f}  "
        f"PCR L={anchor_pcr[0]:.4f} M={anchor_pcr[1]:.4f} H={anchor_pcr[2]:.4f}")

    log("loading NN-dist features")
    nnd_oof = np.load(ART / "oof_nn_dist_features.npy").astype(np.float32)
    nnd_test = np.load(ART / "test_nn_dist_features.npy").astype(np.float32)
    frac_high_oof = nnd_oof[:, 4]   # frac of k=16 NN with class High
    frac_high_test = nnd_test[:, 4]
    log(f"  oof frac_high pct: 25={np.percentile(frac_high_oof,25):.3f} "
        f"50={np.percentile(frac_high_oof,50):.3f} "
        f"75={np.percentile(frac_high_oof,75):.3f} "
        f"95={np.percentile(frac_high_oof,95):.3f}")

    # NN-dist standalone OOF (for 'and nndist agrees' branch)
    log("loading nndist standalone OOF for argmax-agreement gating")
    nnd_pred_oof = normed(np.load(ART / "oof_recipe_full_te_nndist.npy"))
    nnd_pred_test = normed(np.load(ART / "test_recipe_full_te_nndist.npy"))
    nnd_argmax_oof = predict_anchor(nnd_pred_oof)
    nnd_argmax_test = predict_anchor(nnd_pred_test)

    log("=== sweep ===")
    rows = sweep_grid(anchor_o, frac_high_oof, nnd_argmax_oof, y,
                      anchor_bal, anchor_pcr)

    # Print top-10 gate-passing rows by Δ
    passing = [r for r in rows if r["pcr_pass"]]
    log(f"\n{len(passing)}/{len(rows)} configs pass per-class guardrail")
    passing.sort(key=lambda r: r["delta"], reverse=True)
    log("top 12 gate-passing configs by Δ:")
    for r in passing[:12]:
        ov = r["overrides"]
        log(f"  mode={r['mode']:<14} tauL={r['tauL']!s:<6} "
            f"tauH={r['tauH']!s:<6} Δ={r['delta']:+.5f} "
            f"PCR=L{r['pcr'][0]:.4f}/M{r['pcr'][1]:.4f}/H{r['pcr'][2]:.4f} "
            f"ov_HtoNotH={ov['HtoNotH']} ov_NotHtoH={ov['NotHtoH']}")

    summary = dict(
        anchor_bal=float(anchor_bal),
        anchor_pcr=anchor_pcr.tolist(),
        n_rows=len(rows),
        n_pass_guardrail=len(passing),
        rows=rows,
    )

    # Emit submissions for any gate-passing config with Δ ≥ EMIT_DELTA.
    # Cap at 5 distinct configs to avoid spamming the submissions/ dir.
    ids = pd.read_csv("submissions/submission_recipe_full_te.csv")["id"].values
    emitted = []
    seen_keys = set()
    for r in passing:
        if r["delta"] < EMIT_DELTA:
            break
        key = (r["mode"], r["tauL"], r["tauH"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        path = emit_submission(r["mode"], anchor_t, frac_high_test,
                               nnd_argmax_test, r["tauL"], r["tauH"], ids)
        emitted.append(dict(row=r, path=path))
        if len(emitted) >= 5:
            break
    summary["emitted"] = emitted

    out = ART / "opt2_nn_route_results.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out}")
    log(f"total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
