"""Bank-add follow-up for N5b deployments #1 and #3.

If the direct blend gate (n5b_blend_gate.py) nulls, this script tests the
deeper insertion point: include the new component as a meta-stacker BANK
INPUT, retrain the metastack with the same heavy-reg HPs, then test the
LB-best primary architecture with the new meta swapped in.

Usage:
  CAND=ood    python scripts/n5b_bank_add_test.py
  CAND=knn10k python scripts/n5b_bank_add_test.py
  CAND=both   python scripts/n5b_bank_add_test.py

Steps:
  1) Verify oof_recipe_full_te_<suffix>.npy exists for the candidate.
  2) Re-run tier1b_xgb_metastack with META_OUT_SUFFIX="_n5b_<cand>"
     (auto-includes the new component since load_pool() scans oof_*.npy).
  3) Iso-cal the new metastack OOF/test.
  4) Build primary' = 0.7 × LB-best-3-stack + 0.3 × new_meta_iso.
  5) Score @ fixed BIAS, compare to LB-best 4-stack PRIMARY (0.98094).
  6) If Δ ≥ +2e-4 AND per-class guardrail passes, emit submission.

NOTE: re-running tier1b_xgb_metastack ensures EVERY current oof_*.npy
ends up in the bank, not just the named candidate. To restrict to a
specific candidate, temporarily move other new oof files out, run, then
move them back.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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


def main() -> None:
    cand = os.environ.get("CAND", "")
    assert cand in ("ood", "knn10k", "both"), f"CAND must be ood|knn10k|both, got {cand!r}"

    # Verify candidate component(s) exist on disk.
    needed = []
    if cand in ("ood", "both"):
        needed.append("recipe_full_te_ood")
    if cand in ("knn10k", "both"):
        needed.append("recipe_full_te_knn10k")
    for n in needed:
        p = ART / f"oof_{n}.npy"
        assert p.exists(), f"missing {p} — wait for D1/D3 production to finish"

    suffix = f"_n5b_{cand}"
    print(f"[1] Re-training meta-stacker with META_OUT_SUFFIX={suffix!r}...")
    env = os.environ.copy()
    env["META_OUT_SUFFIX"] = suffix
    cp = subprocess.run([sys.executable, "scripts/tier1b_xgb_metastack.py"],
                         env=env, check=False)
    if cp.returncode != 0:
        print(f"meta-stacker failed (rc={cp.returncode})"); return
    new_oof = np.load(ART / f"oof_xgb_metastack{suffix}.npy")
    new_test = np.load(ART / f"test_xgb_metastack{suffix}.npy")
    print(f"    new meta shape: {new_oof.shape}")

    print("[2] Iso-cal new meta + build primary'...")
    y = load_y()
    new_iso_o, new_iso_t = iso_cal(normed(new_oof), normed(new_test), y)
    s3_o, s3_t = build_lbbest_stack(y)
    p_new_o = log_blend([s3_o, new_iso_o], np.array([0.70, 0.30]))
    p_new_t = log_blend([s3_t, new_iso_t], np.array([0.70, 0.30]))

    print("[3] Build current PRIMARY for baseline comparison...")
    v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    v1_iso_o, v1_iso_t = iso_cal(v1_o, v1_t, y)
    p_v1_o = log_blend([s3_o, v1_iso_o], np.array([0.70, 0.30]))
    p_v1_t = log_blend([s3_t, v1_iso_t], np.array([0.70, 0.30]))

    pred_v1 = (np.log(np.clip(p_v1_o, 1e-12, 1)) + BIAS).argmax(1)
    pred_new = (np.log(np.clip(p_new_o, 1e-12, 1)) + BIAS).argmax(1)
    m_v1 = balanced_accuracy_score(y, pred_v1)
    m_new = balanced_accuracy_score(y, pred_new)
    rec_v1 = recall_score(y, pred_v1, average=None)
    rec_new = recall_score(y, pred_new, average=None)
    d_macro = m_new - m_v1
    d_rec = rec_new - rec_v1

    print("\n[4] Comparison @ fixed BIAS:")
    print(f"  v1 PRIMARY (LB 0.98094): OOF={m_v1:.5f} recall={rec_v1.round(5)}")
    print(f"  new PRIMARY (bank+{cand}): OOF={m_new:.5f} recall={rec_new.round(5)}")
    print(f"  Δ macro={d_macro:+.6f}  Δ recall={d_rec.round(6)}")
    guard = bool((d_rec >= -5e-4).all())
    gate = d_macro >= 2e-4
    print(f"  per-class guardrail: {'PASS' if guard else 'FAIL'}")
    print(f"  +2e-4 gate: {'PASS' if gate else 'FAIL'}")

    if gate and guard:
        # Compare test argmax differences vs current primary submission
        pred_t_v1 = (np.log(np.clip(p_v1_t, 1e-12, 1)) + BIAS).argmax(1)
        pred_t_new = (np.log(np.clip(p_new_t, 1e-12, 1)) + BIAS).argmax(1)
        n_diff = int((pred_t_new != pred_t_v1).sum())
        test = pd.read_csv("data/test.csv")
        sub = pd.DataFrame({"id": test["id"].values,
                             "Irrigation_Need": [LABELS[i] for i in pred_t_new]})
        fname = f"submission_n5b_bankadd_{cand}.csv"
        sub.to_csv(SUB / fname, index=False)
        print(f"  test_argmax_diff_vs_primary={n_diff}")
        print(f"  -> SAVED {fname} (AWAITING USER APPROVAL FOR LB)")
    else:
        print("  no submission emitted")

    out = {"candidate": cand, "v1_oof": float(m_v1), "new_oof": float(m_new),
           "d_macro": float(d_macro), "d_rec": d_rec.tolist(),
           "gate_pass": bool(gate and guard)}
    with open(ART / f"n5b_bankadd_{cand}_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {ART}/n5b_bankadd_{cand}_results.json")


if __name__ == "__main__":
    main()
