"""Option 2 v2 — NN-dist route override with relaxed gates.

The original lb_ready_emit's Option 2 found 2 OOF configs that passed
the guardrail but they had 0 test overrides — the gates rejected
every test row. This version sweeps a wider, more permissive grid to
find configs that:
  - PASS the OOF per-class recall guardrail
  - produce ≥30 test-row overrides (so the LB probe is meaningful)

Mechanisms:
  A: HtoNotH bare — anchor=High AND frac_high_neighbors < τL → flip to next-best
  C: NotHtoH bare — anchor!=High AND frac_high_neighbors > τH → flip to High

These are pure FAISS-distance gates, no NN-dist standalone agreement
required (the original 'and nndist agrees' branch was too strict on
test).
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
TARGET_TEST_OVERRIDES = 30


def per_class_recall(y, pred, n=3):
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    return np.diag(cm).astype(float) / cm.sum(axis=1).clip(1)


def predict_anchor(p):
    return (np.log(np.clip(p, 1e-12, 1)) + BIAS).argmax(1)


def main():
    t0 = time.time()
    y = load_y()
    log("loading LB-best 4-stack anchor")
    lb3_o, lb3_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    anchor_o = log_blend([lb3_o, meta_o_iso], np.array([0.70, 0.30]))
    anchor_t = log_blend([lb3_t, meta_t_iso], np.array([0.70, 0.30]))
    pred_anc_o = predict_anchor(anchor_o)
    pred_anc_t = predict_anchor(anchor_t)
    bal_anc = balanced_accuracy_score(y, pred_anc_o)
    pcr_anc = per_class_recall(y, pred_anc_o)
    log(f"  anchor OOF = {bal_anc:.5f}  PCR={pcr_anc.tolist()}")

    nnd_oof = np.load(ART / "oof_nn_dist_features.npy").astype(np.float32)
    nnd_test = np.load(ART / "test_nn_dist_features.npy").astype(np.float32)
    fh_o = nnd_oof[:, 4]
    fh_t = nnd_test[:, 4]
    log(f"  test frac_high pct: 25={np.percentile(fh_t,25):.3f} "
        f"50={np.percentile(fh_t,50):.3f} 75={np.percentile(fh_t,75):.3f} "
        f"90={np.percentile(fh_t,90):.3f} 95={np.percentile(fh_t,95):.3f}")

    biased_o = np.log(np.clip(anchor_o, 1e-12, 1)) + BIAS
    biased_t = np.log(np.clip(anchor_t, 1e-12, 1)) + BIAS

    # Branch A: HtoNotH bare. anchor=High AND frac_high < τL → flip to next-best
    # Wider, more aggressive grid.
    rows = []
    for tL in np.arange(0.00, 0.50, 0.025):
        m_o = (pred_anc_o == 2) & (fh_o < tL)
        if m_o.sum() == 0:
            continue
        biased_no_h_o = biased_o[m_o].copy()
        biased_no_h_o[:, 2] = -np.inf
        new_o = pred_anc_o.copy()
        new_o[m_o] = biased_no_h_o.argmax(1)
        bal = balanced_accuracy_score(y, new_o)
        pcr = per_class_recall(y, new_o)
        if not all(pcr[c] >= pcr_anc[c] - GUARDRAIL for c in range(3)):
            continue
        # test version
        m_t = (pred_anc_t == 2) & (fh_t < tL)
        biased_no_h_t = biased_t[m_t].copy()
        biased_no_h_t[:, 2] = -np.inf
        new_t = pred_anc_t.copy()
        if m_t.sum() > 0:
            new_t[m_t] = biased_no_h_t.argmax(1)
        rows.append(dict(
            mode="A_HtoNotH_bare", tau=float(tL),
            oof_bal=float(bal), oof_delta=float(bal - bal_anc),
            ov_oof=int(m_o.sum()), ov_test=int(m_t.sum()),
            pred_test=new_t,
            pcr=pcr.tolist(),
        ))

    # Branch C: NotHtoH bare. anchor!=High AND frac_high > τH → flip to High.
    for tH in np.arange(0.05, 0.50, 0.025):
        m_o = (pred_anc_o != 2) & (fh_o > tH)
        if m_o.sum() == 0:
            continue
        new_o = pred_anc_o.copy()
        new_o[m_o] = 2
        bal = balanced_accuracy_score(y, new_o)
        pcr = per_class_recall(y, new_o)
        if not all(pcr[c] >= pcr_anc[c] - GUARDRAIL for c in range(3)):
            continue
        m_t = (pred_anc_t != 2) & (fh_t > tH)
        new_t = pred_anc_t.copy()
        if m_t.sum() > 0:
            new_t[m_t] = 2
        rows.append(dict(
            mode="C_NotHtoH_bare", tau=float(tH),
            oof_bal=float(bal), oof_delta=float(bal - bal_anc),
            ov_oof=int(m_o.sum()), ov_test=int(m_t.sum()),
            pred_test=new_t,
            pcr=pcr.tolist(),
        ))

    log(f"\n{len(rows)} configs pass OOF guardrail")
    rows.sort(key=lambda r: r["ov_test"], reverse=True)
    log("top 10 by test-override count:")
    for r in rows[:10]:
        log(f"  {r['mode']:20s} τ={r['tau']:.3f}  ov_test={r['ov_test']:>4}  "
            f"ov_oof={r['ov_oof']:>5}  Δ={r['oof_delta']:+.5f}  "
            f"PCR=L{r['pcr'][0]:.4f}/M{r['pcr'][1]:.4f}/H{r['pcr'][2]:.4f}")

    # Emit top configs that have ≥ TARGET_TEST_OVERRIDES.
    ids = pd.read_csv("submissions/submission_recipe_full_te.csv")["id"].values
    cls_map = {0: "Low", 1: "Medium", 2: "High"}
    primary_path = "submissions/submission_tier1b_greedy_meta.csv"
    primary_pred = pd.read_csv(primary_path)["Irrigation_Need"].values
    summary = {"anchor_oof": float(bal_anc), "emitted": []}
    seen_taus = set()
    for r in rows:
        if r["ov_test"] < TARGET_TEST_OVERRIDES:
            continue
        # 1 emission per (mode, tau-bucket) to avoid spam
        bucket = (r["mode"], round(r["tau"], 2))
        if bucket in seen_taus:
            continue
        seen_taus.add(bucket)
        sub = pd.DataFrame({"id": ids,
                            "Irrigation_Need": [cls_map[i] for i in r["pred_test"]]})
        diff_vs_primary = int((sub["Irrigation_Need"].values != primary_pred).sum())
        name = f"opt2v2_{r['mode']}_t{int(r['tau']*1000):03d}"
        p = SUB / f"submission_{name}.csv"
        sub.to_csv(p, index=False)
        log(f"  emitted {p}  ov_test={r['ov_test']}  diff_vs_primary={diff_vs_primary}")
        summary["emitted"].append(dict(
            name=name, path=str(p),
            mode=r["mode"], tau=r["tau"],
            ov_test=r["ov_test"], ov_oof=r["ov_oof"],
            oof_delta=r["oof_delta"],
            pcr=r["pcr"],
            test_diff_vs_primary=diff_vs_primary,
        ))
        if len(summary["emitted"]) >= 4:
            break

    out = ART / "option2_route_v2_results.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"wrote {out}")
    log(f"total wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
