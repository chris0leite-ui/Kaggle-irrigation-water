"""Unified blend gate for N5b deployments #1 and #3.

Usage:
  CAND=ood    python scripts/n5b_blend_gate.py
  CAND=knn10k python scripts/n5b_blend_gate.py

Loads the candidate (recipe_full_te + extra FE block) and runs the
standard fixed-bias blend gate vs three anchors:
  - recipe_full_te (OOF 0.97967)
  - LB-best 3-stack (OOF 0.98061)
  - LB-best 4-stack PRIMARY (OOF 0.98084, LB 0.98094)

Reports Jaccard + errs vs each anchor, alpha-sweep, per-class recall delta
+ macro guardrail. Emits submission CSV iff alpha-peak Δ ≥ +2e-4 AND
all per-class Δ ≥ -5e-4.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(parents=True, exist_ok=True)
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
LABELS = ["Low", "Medium", "High"]


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def build_primary_4stack(y):
    s3_o, s3_t = build_lbbest_stack(y)
    ms_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_o_iso, ms_t_iso = iso_cal(ms_o, ms_t, y)
    p4_o = log_blend([s3_o, ms_o_iso], np.array([0.70, 0.30]))
    p4_t = log_blend([s3_t, ms_t_iso], np.array([0.70, 0.30]))
    return p4_o, p4_t


def err_jaccard(a_pred, b_pred, y):
    ea = a_pred != y; eb = b_pred != y
    inter = int((ea & eb).sum()); union = int((ea | eb).sum())
    return inter / max(1, union)


def main() -> None:
    cand = os.environ.get("CAND", "")
    assert cand in ("ood", "knn10k", "ood_knn10k"), f"CAND must be 'ood'|'knn10k'|'ood_knn10k', got {cand!r}"
    suffix = {"ood": "_ood", "knn10k": "_knn10k", "ood_knn10k": "_ood_knn10k"}[cand]
    cand_oof_path = ART / f"oof_recipe_full_te{suffix}.npy"
    cand_test_path = ART / f"test_recipe_full_te{suffix}.npy"
    assert cand_oof_path.exists(), f"missing {cand_oof_path}"
    assert cand_test_path.exists(), f"missing {cand_test_path}"

    print(f"[1] Loading candidate {cand!r}...")
    y = load_y()
    p_cand_o = normed(np.load(cand_oof_path).astype(np.float32))
    p_cand_t = normed(np.load(cand_test_path).astype(np.float32))

    # Standalone tuned bias from the recipe pipeline's results JSON.
    res_path = ART / f"recipe_full_te{suffix}_results.json"
    if res_path.exists():
        res = json.loads(res_path.read_text())
        own_bias = np.array(res.get("tuned_bias", BIAS.tolist()), dtype=np.float32)
        own_oof = res.get("tuned_oof", None)
        print(f"    own tuned: OOF={own_oof}  bias={own_bias.round(4).tolist()}")

    print(f"[2] Building anchors...")
    # Anchor A: recipe_full_te
    a_o = normed(np.load(ART / "oof_recipe_full_te.npy").astype(np.float32))
    a_t = normed(np.load(ART / "test_recipe_full_te.npy").astype(np.float32))
    # Anchor B: LB-best 3-stack
    b_o, b_t = build_lbbest_stack(y)
    # Anchor C: LB-best 4-stack PRIMARY
    c_o, c_t = build_primary_4stack(y)

    anchors = {
        "recipe":  (a_o, a_t),
        "lb3":     (b_o, b_t),
        "lb4_PRIMARY": (c_o, c_t),
    }

    print("[3] Anchor diagnostics @ recipe bias:")
    print(f"  {'name':18s}  {'OOF':>7s}  {'errs':>5s}  {'recL':>6s}  {'recM':>6s}  {'recH':>6s}")
    for name, (po, _) in anchors.items():
        pred = (np.log(np.clip(po, 1e-12, 1)) + BIAS).argmax(1)
        m = balanced_accuracy_score(y, pred)
        recs = recall_score(y, pred, average=None)
        errs = int((pred != y).sum())
        print(f"  {name:18s}  {m:.5f}  {errs:5d}  {recs[0]:.4f}  {recs[1]:.4f}  {recs[2]:.4f}")
    pred_cand = (np.log(np.clip(p_cand_o, 1e-12, 1)) + BIAS).argmax(1)
    m_cand = balanced_accuracy_score(y, pred_cand)
    recs_cand = recall_score(y, pred_cand, average=None)
    errs_cand = int((pred_cand != y).sum())
    print(f"  {cand+'_CAND':18s}  {m_cand:.5f}  {errs_cand:5d}  "
          f"{recs_cand[0]:.4f}  {recs_cand[1]:.4f}  {recs_cand[2]:.4f}")

    print("\n[4] Jaccard(cand-errs vs anchor-errs) @ recipe bias:")
    for name, (po, _) in anchors.items():
        pred = (np.log(np.clip(po, 1e-12, 1)) + BIAS).argmax(1)
        j = err_jaccard(pred_cand, pred, y)
        print(f"  Jaccard vs {name:18s} = {j:.4f}")

    print("\n[5] Blend gate vs each anchor (fixed BIAS, alpha sweep)...")
    alphas = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    out = {"candidate": cand, "anchors": {}, "best_emit": None}
    best_overall = None
    for name, (po, pt) in anchors.items():
        pred_a = (np.log(np.clip(po, 1e-12, 1)) + BIAS).argmax(1)
        m_a = balanced_accuracy_score(y, pred_a)
        recs_a = recall_score(y, pred_a, average=None)
        rows = []
        peak = (-1, -1, None, None)
        for a in alphas:
            blend_o = log_blend([po, p_cand_o], np.array([1 - a, a]))
            pred_b = (np.log(np.clip(blend_o, 1e-12, 1)) + BIAS).argmax(1)
            m_b = balanced_accuracy_score(y, pred_b)
            d = m_b - m_a
            recs_b = recall_score(y, pred_b, average=None)
            d_rec = (recs_b - recs_a).round(5)
            guard = bool((d_rec >= -5e-4).all())
            row = dict(alpha=a, oof=round(m_b, 6), d_macro=round(d, 6),
                       drec=d_rec.tolist(), guard=guard)
            rows.append(row)
            if d > peak[0] and guard:
                peak = (d, m_b, a, d_rec.tolist())
        out["anchors"][name] = {"baseline": round(m_a, 6), "rows": rows,
                                "peak": {"d_macro": peak[0], "oof": peak[1],
                                         "alpha": peak[2], "drec": peak[3]}}
        print(f"  vs {name:18s} baseline={m_a:.5f}  peak: a={peak[2]} "
              f"OOF={peak[1]:.5f} dM={peak[0]:+.5f} drec={peak[3]}")
        if name == "lb4_PRIMARY" and peak[2] is not None and peak[0] >= 2e-4:
            best_overall = (peak[2], peak[1], peak[0])

    print("\n[6] EMIT DECISION (vs PRIMARY anchor)")
    if best_overall is None:
        print("  No alpha clears +2e-4 OR per-class guardrail. NULL.")
    else:
        a, oof, d = best_overall
        # Build test-side blend
        pt_anchor = anchors["lb4_PRIMARY"][1]
        blend_t = log_blend([pt_anchor, p_cand_t], np.array([1 - a, a]))
        pred_test = (np.log(np.clip(blend_t, 1e-12, 1)) + BIAS).argmax(1)
        # Compare to current LB-best primary submission count differences
        primary_pred_test = (np.log(np.clip(pt_anchor, 1e-12, 1)) + BIAS).argmax(1)
        n_diff = int((pred_test != primary_pred_test).sum())
        test = pd.read_csv("data/test.csv")
        sub = pd.DataFrame({"id": test["id"].values,
                             "Irrigation_Need": [LABELS[i] for i in pred_test]})
        fname = f"submission_n5b_d_{cand}_a{int(a*1000):03d}.csv"
        sub.to_csv(SUB / fname, index=False)
        print(f"  PASSING. alpha={a}  OOF={oof:.5f}  dM={d:+.5f}  "
              f"test_diff={n_diff}")
        print(f"  -> SAVED {fname} (AWAITING USER APPROVAL FOR LB)")
        out["best_emit"] = {"alpha": a, "oof": oof, "d_macro": d,
                             "test_diff": n_diff, "submission": fname}

    out_path = ART / f"n5b_blend_gate_{cand}_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
