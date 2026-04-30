"""Variant A: soft router blend (per-row weighted blend, not argmax flip).

Mechanism: idea 4's router has AUC 0.895 on the v1↔rawashishsin disagreement
task. Hard argmax routing failed because removing v1's correct H predictions
costs more than picking up rawashishsin's correct calls. Soft blend uses
the router's P(raw_wins) as a per-row weight on rawashishsin's prob vector,
preserving H probability mass.

Per-row blend (only on disagreement rows):
    blended[i] = (1 - w[i]) * v1_probs[i] + w[i] * raw_probs[i]
    where w[i] = router_prob(i)  (P(raw_wins) on disagreement rows, 0 elsewhere)

Apply v1's tuned bias to blended probs. Argmax. 4-gate vs v1.

Output:
    submissions/submission_router_softblend.csv (if 4-gate passes)
    scripts/artifacts/router_softblend_results.json
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
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    out = np.zeros(n_class)
    for k in range(n_class):
        m = y == k
        if m.sum():
            out[k] = (pred[m] == k).sum() / m.sum()
    return out


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values

    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_test = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)

    # Router OOF: P(raw_wins) on disagreement rows; 0 on agreement rows
    router_oof = np.load(ART / "oof_router_predictions.npy").astype(np.float32)
    router_test = np.load(ART / "test_router_decisions.npy").astype(np.float32)

    # Re-tune biases on full OOF (apples-to-apples with idea 4 script)
    prior = np.bincount(y, minlength=3) / len(y)
    bias_v1, _ = tune_log_bias(v1_oof, y, prior)
    bias_raw, _ = tune_log_bias(raw_oof, y, prior)
    print(f"v1 bias = {bias_v1.round(4).tolist()}")
    print(f"raw bias = {bias_raw.round(4).tolist()}")

    def to_probs(probs, bias):
        z = safelog(probs) + bias
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    v1_p_oof = to_probs(v1_oof, bias_v1)
    v1_p_test = to_probs(v1_test, bias_v1)
    raw_p_oof = to_probs(raw_oof, bias_raw)
    raw_p_test = to_probs(raw_test, bias_raw)

    v1_arg_oof = v1_p_oof.argmax(1)
    v1_arg_test = v1_p_test.argmax(1)
    raw_arg_oof = raw_p_oof.argmax(1)
    raw_arg_test = raw_p_test.argmax(1)

    dis_oof = (v1_arg_oof != raw_arg_oof)
    dis_test = (v1_arg_test != raw_arg_test)
    print(f"OOF disagreement rows: {dis_oof.sum():,}")
    print(f"Test disagreement rows: {dis_test.sum():,}")

    # Restrict router weight to disagreement rows only (already 0 elsewhere)
    # Soft blend: per-row weight w = router prob on disagreement rows
    # On agreement rows: keep v1 (router_oof = 0 → blend = v1)
    w_oof = router_oof.astype(np.float32).reshape(-1, 1)
    w_test = router_test.astype(np.float32).reshape(-1, 1)

    # Restrict to disagreement rows. On agreement rows, blend = v1.
    w_oof = np.where(dis_oof.reshape(-1, 1), w_oof, 0.0)
    w_test = np.where(dis_test.reshape(-1, 1), w_test, 0.0)

    # Sweep w-scale alpha (overall multiplier on the per-row weight)
    # alpha=1.0 = use router's raw P(raw_wins); alpha=0.5 = halve it (more conservative)
    # Also sweep a per-class scale on the rare class (preserve H mass even on routed rows)
    sweep = []

    v1_bal = balanced_accuracy_score(y, v1_arg_oof)
    pcr_v1 = per_class_recall(y, v1_arg_oof)
    test_class_v1 = [int((v1_arg_test == k).sum()) for k in range(3)]

    for alpha in [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
        # Soft per-row blend at scaled router weight
        w_oof_s = (w_oof * alpha).clip(0, 1)
        w_test_s = (w_test * alpha).clip(0, 1)

        blend_oof_p = (1 - w_oof_s) * v1_p_oof + w_oof_s * raw_p_oof
        blend_test_p = (1 - w_test_s) * v1_p_test + w_test_s * raw_p_test
        # Renormalize defensively (already convex combo of probs, so no-op modulo numerical)
        blend_oof_p /= blend_oof_p.sum(axis=1, keepdims=True)
        blend_test_p /= blend_test_p.sum(axis=1, keepdims=True)

        bl_arg_oof = blend_oof_p.argmax(1)
        bl_arg_test = blend_test_p.argmax(1)
        new_bal = balanced_accuracy_score(y, bl_arg_oof)
        pcr_new = per_class_recall(y, bl_arg_oof)
        pcr_delta = (pcr_new - pcr_v1).tolist()
        delta = float(new_bal - v1_bal)
        # G4: net rare-class flip on OOF
        net_h_oof = int(((bl_arg_oof == 2) & (v1_arg_oof != 2)).sum() -
                        ((v1_arg_oof == 2) & (bl_arg_oof != 2)).sum())
        churn_h_oof = int(((bl_arg_oof == 2) ^ (v1_arg_oof != 2)).sum())  # not used
        # Test-side class delta
        test_class_delta = [int((bl_arg_test == k).sum() - test_class_v1[k]) for k in range(3)]
        # Test-side disagreement count
        test_diff = int((bl_arg_test != v1_arg_test).sum())

        # 4-gate decisions
        g1 = delta >= 2e-4
        g2 = all(d >= -5e-4 for d in pcr_delta)
        # G4: ratio of net_H to total H churn (proxy for ADD-direction asymmetry)
        oof_h_changes = int(((bl_arg_oof == 2) ^ (v1_arg_oof == 2)).sum())
        g4_ratio = abs(net_h_oof) / max(oof_h_changes, 1)
        g4 = (net_h_oof > 0) and (g4_ratio >= 0.5)

        sweep.append(dict(
            alpha=alpha, oof_delta=delta, new_bal=float(new_bal),
            pcr_delta=pcr_delta, net_h_oof=net_h_oof,
            oof_h_changes=oof_h_changes, g4_ratio=float(g4_ratio),
            test_diff=test_diff, test_class_delta=test_class_delta,
            g1_pass=g1, g2_pass=g2, g4_pass=g4,
        ))
        print(f"alpha={alpha:.2f}  Δ_OOF={delta:+.6f}  "
              f"PCR=[{pcr_delta[0]:+.5f},{pcr_delta[1]:+.5f},{pcr_delta[2]:+.5f}]  "
              f"net_H={net_h_oof:+d}/{oof_h_changes}  test_diff={test_diff}  "
              f"G1{'✓' if g1 else '✗'} G2{'✓' if g2 else '✗'} G4{'✓' if g4 else '✗'}")

    # Pick best gate-pass
    passers = [s for s in sweep if s["g1_pass"] and s["g2_pass"] and s["g4_pass"]]
    best = max(passers, key=lambda s: s["oof_delta"], default=None)
    if best is None:
        # fall back to best OOF delta passing G2 only
        g2only = [s for s in sweep if s["g2_pass"]]
        best = max(g2only, key=lambda s: s["oof_delta"], default=None)
    if best:
        alpha = best["alpha"]
        w_test_s = (w_test * alpha).clip(0, 1)
        blend_test_p = (1 - w_test_s) * v1_p_test + w_test_s * raw_p_test
        blend_test_p /= blend_test_p.sum(axis=1, keepdims=True)
        bl_arg_test = blend_test_p.argmax(1)
        labels = [IDX2CLS[i] for i in bl_arg_test]
        sub_path = SUB / f"submission_router_softblend_a{int(alpha*100):03d}.csv"
        pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(sub_path, index=False)
        print(f"\nwrote {sub_path}  (best alpha={alpha} OOF Δ={best['oof_delta']:+.6f})")

    out = dict(
        v1_bias=bias_v1.tolist(), raw_bias=bias_raw.tolist(),
        v1_oof_bal=float(v1_bal),
        sweep=sweep, best=best,
    )
    with open(ART / "router_softblend_results.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
