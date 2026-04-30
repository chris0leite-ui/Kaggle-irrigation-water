"""#11 Per-class LB-weighted Bayesian Model Average.

Take 6 LB-validated base submissions, compute per-class precision on OOF,
weight each model's per-class probability by its per-class precision /
recall power. Geomean per-row, take argmax.

Different from log-blend (one weight per model): allows model A to
dominate Low predictions while model B dominates High.

Inputs:
  v1_rf       LB 0.98129 (RF natural meta)
  raw         LB 0.98109 (rawashishsin XGB+sklearn TE)
  tier1b      LB 0.98094 (XGB-meta on 63-comp bank)
  3way        LB 0.98005 (recipe multi-seed log-blend)
  recipe      LB 0.97939 (recipe XGB)
  cb          LB 0.97935 (recipe CatBoost)

Mechanism:
  For each model m and class k, compute per-class OOF recall: rec_m[k]
  Per-class weight: w_m[k] = LB_m^β * rec_m[k]^γ, β=γ=1
  Test prob: p_blend[i,k] ∝ Π_m p_m[i,k]^(w_m[k] / Σ_m' w_m'[k])

Per CLAUDE.md: emit candidate CSV; user approves probe.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS = ("Low", "Medium", "High")
CLS2IDX = {c: i for i, c in enumerate(CLS)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def _normed(a, eps=1e-9):
    a = np.clip(a, eps, 1.0)
    return a / a.sum(axis=1, keepdims=True)


def per_class_recall(y, pred, n=3):
    out = np.zeros(n, dtype=np.float64)
    for k in range(n):
        m = y == k
        out[k] = (pred[m] == k).sum() / max(m.sum(), 1)
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    prior = np.bincount(y, minlength=3) / len(y)

    # ===== Load 6 LB-validated bases =====
    bases = [
        # (label, oof_path, test_path, LB)
        ("v1",     "oof_sklearn_rf_meta_natural_v1_lb98129.npy",
                   "test_sklearn_rf_meta_natural_v1_lb98129.npy", 0.98129),
        ("raw",    "oof_rawashishsin_2600.npy",
                   "test_rawashishsin_2600.npy", 0.98109),
        ("t1b",    "oof_tier1b_greedy_meta.npy",
                   "test_tier1b_greedy_meta.npy", 0.98094),
        ("recipe", "oof_recipe_full_te.npy",
                   "test_recipe_full_te.npy", 0.97939),
        ("cb",     "oof_recipe_full_te_catboost.npy",
                   "test_recipe_full_te_catboost.npy", 0.97935),
    ]
    pool = []
    for label, oof_p, test_p, lb in bases:
        oof_path = ART / oof_p
        test_path = ART / test_p
        if not oof_path.exists() or not test_path.exists():
            print(f"  SKIP {label}: missing")
            continue
        oof = _normed(np.load(oof_path).astype(np.float32))
        tst = _normed(np.load(test_path).astype(np.float32))
        bias, tuned = tune_log_bias(oof, y, prior)
        oof_b = (np.log(np.clip(oof, 1e-9, 1.0)) + bias)
        # Compute per-class recall at tuned bias
        argm = oof_b.argmax(1)
        pcr = per_class_recall(y, argm)
        # Test biased probs (after bias-adjusted softmax)
        oof_b -= oof_b.max(axis=1, keepdims=True)
        oof_p_biased = np.exp(oof_b) / np.exp(oof_b).sum(axis=1, keepdims=True)
        test_log = np.log(np.clip(tst, 1e-9, 1.0)) + bias
        test_log -= test_log.max(axis=1, keepdims=True)
        test_p_biased = np.exp(test_log) / np.exp(test_log).sum(axis=1, keepdims=True)
        pool.append(dict(
            label=label, lb=lb, tuned=tuned, bias=bias,
            pcr=pcr, oof_argmax=argm,
            oof_biased=oof_p_biased, test_biased=test_p_biased,
        ))
        print(f"  {label} (LB {lb:.5f}): tuned OOF {tuned:.5f}  "
              f"PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # ===== Per-class weights: w_m[k] = LB_m * rec_m[k] =====
    # (more weight to LB-stronger models AND models that recall class k well)
    print(f"\n=== Per-class weight grid ===")
    print(f"{'model':<7}{'class':<8}{'LB':<10}{'rec':<10}{'w=LB*rec':<10}")
    for p in pool:
        for k in range(3):
            w = p["lb"] * p["pcr"][k]
            print(f"  {p['label']:<7}{IDX2CLS[k]:<8}{p['lb']:<10.5f}{p['pcr'][k]:<10.4f}{w:<10.5f}")

    # Compute normalized per-class weights (sum over models = 1)
    n_models = len(pool)
    w = np.zeros((n_models, 3))
    for i, p in enumerate(pool):
        for k in range(3):
            w[i, k] = p["lb"] * p["pcr"][k]
    # Normalize per class
    w_norm = w / w.sum(axis=0, keepdims=True)
    print(f"\nNormalized per-class weights (rows=model, cols=class):")
    print(f"{'':>9}{'Low':>10}{'Medium':>10}{'High':>10}")
    for i, p in enumerate(pool):
        print(f"  {p['label']:<7}" + "".join(f"{w_norm[i, k]:10.4f}" for k in range(3)))

    # ===== Blend OOF =====
    # log p_blend[i, k] = sum_m w_norm[m, k] * log p_m[i, k]
    n_oof = len(y)
    log_blend_oof = np.zeros((n_oof, 3), dtype=np.float64)
    for i, p in enumerate(pool):
        log_p = np.log(np.clip(p["oof_biased"], 1e-9, 1.0))
        for k in range(3):
            log_blend_oof[:, k] += w_norm[i, k] * log_p[:, k]
    # Normalize
    log_blend_oof -= log_blend_oof.max(axis=1, keepdims=True)
    p_blend_oof = np.exp(log_blend_oof)
    p_blend_oof /= p_blend_oof.sum(axis=1, keepdims=True)

    # Tune log-bias on blend (just to check; we have post-class weights already)
    bias_b, tuned_b = tune_log_bias(p_blend_oof, y, prior)
    pred_blend_oof = (np.log(np.clip(p_blend_oof, 1e-9, 1.0)) + bias_b).argmax(1)
    pcr_blend = per_class_recall(y, pred_blend_oof)
    bal_blend = balanced_accuracy_score(y, pred_blend_oof)
    print(f"\n=== OOF blend results ===")
    print(f"  Per-class BMA (tuned bias {bias_b.round(3).tolist()}): {bal_blend:.5f}")
    print(f"  PCR=[L={pcr_blend[0]:.4f} M={pcr_blend[1]:.4f} H={pcr_blend[2]:.4f}]")
    # Compare to v1 baseline (=0.98063 tuned)
    print(f"  Δ vs v1 (LB-best base 0.98063): {bal_blend - 0.98063:+.5f}")

    # ===== Blend test =====
    n_te = len(test_ids)
    log_blend_test = np.zeros((n_te, 3), dtype=np.float64)
    for i, p in enumerate(pool):
        log_p = np.log(np.clip(p["test_biased"], 1e-9, 1.0))
        for k in range(3):
            log_blend_test[:, k] += w_norm[i, k] * log_p[:, k]
    log_blend_test -= log_blend_test.max(axis=1, keepdims=True)
    p_blend_test = np.exp(log_blend_test)
    p_blend_test /= p_blend_test.sum(axis=1, keepdims=True)
    test_pred = (np.log(np.clip(p_blend_test, 1e-9, 1.0)) + bias_b).argmax(1)

    # Compare to LB-best winner
    winner_pred = pd.read_csv(SUB / "submission_2other_raw_tier1b_k2.csv")[TARGET].map(CLS2IDX).to_numpy()
    diff_winner = int((test_pred != winner_pred).sum())
    v1_pred = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")[TARGET].map(CLS2IDX).to_numpy()
    diff_v1 = int((test_pred != v1_pred).sum())
    print(f"\nTest blend vs LB-best winner: {diff_winner} rows differ")
    print(f"Test blend vs v1: {diff_v1} rows differ")
    # Class dist
    cnt = np.bincount(test_pred, minlength=3)
    print(f"Test class distribution: L={cnt[0]} M={cnt[1]} H={cnt[2]}")
    cnt_w = np.bincount(winner_pred, minlength=3)
    print(f"  vs winner:             L={cnt_w[0]} M={cnt_w[1]} H={cnt_w[2]}")

    # Save
    path = SUB / "submission_n11_bma_per_class.csv"
    pd.DataFrame({"id": test_ids,
                  TARGET: [IDX2CLS[i] for i in test_pred]}).to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # ===== Layered: use BMA as new OTHER for override on top of winner =====
    # If BMA differs from winner, can use as additional consensus signal
    # Let's check: where BMA agrees with raw AND tier1b on a class != winner
    raw_test = _normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))
    t1b_test = _normed(np.load(ART / "test_tier1b_greedy_meta.npy").astype(np.float32))
    raw_b = tune_log_bias(_normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)), y, prior)[0]
    t1b_b = tune_log_bias(_normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32)), y, prior)[0]
    raw_arg = (np.log(np.clip(raw_test, 1e-9, 1.0)) + raw_b).argmax(1)
    t1b_arg = (np.log(np.clip(t1b_test, 1e-9, 1.0)) + t1b_b).argmax(1)
    consensus = (raw_arg == t1b_arg) & (raw_arg != winner_pred) & (test_pred == raw_arg)
    n_layered = consensus.sum()
    print(f"\n=== Layered: BMA + raw + t1b unanimous (3-way consensus on winner overrides) ===")
    print(f"  Overrides: {n_layered}")
    if n_layered > 0:
        layered = winner_pred.copy()
        layered[consensus] = raw_arg[consensus]
        for src in range(3):
            for dst in range(3):
                if src == dst: continue
                cnt = ((winner_pred == src) & (layered == dst)).sum()
                if cnt:
                    print(f"    {IDX2CLS[src]}->{IDX2CLS[dst]}: {cnt}")
        path2 = SUB / "submission_n11_bma_layered_on_winner.csv"
        pd.DataFrame({"id": test_ids,
                      TARGET: [IDX2CLS[i] for i in layered]}).to_csv(path2, index=False)
        print(f"  Saved: {path2}")

    # Save summary
    summary = {
        "BMA_OOF_balanced_acc": float(bal_blend),
        "BMA_test_diff_vs_winner": diff_winner,
        "BMA_test_diff_vs_v1": diff_v1,
        "BMA_class_dist": {IDX2CLS[k]: int(cnt[k]) if hasattr(cnt, '__len__') and len(cnt) > k else 0 for k in range(3)},
        "weights_per_class": {p["label"]: {IDX2CLS[k]: float(w_norm[i, k]) for k in range(3)}
                              for i, p in enumerate(pool)},
        "layered_n_overrides": int(n_layered),
        "submission": str(path),
    }
    with open(ART / "n11_bma_per_class_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: scripts/artifacts/n11_bma_per_class_results.json")


if __name__ == "__main__":
    main()
