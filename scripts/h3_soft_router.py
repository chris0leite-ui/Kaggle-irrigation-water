"""H3 SOFT router: per-row weighted blend of v1 and raw on disagreement rows.

For each disagreement row i:
  soft_probs[i] = w[i] * v1_probs[i] + (1 - w[i]) * raw_probs[i]
where w[i] = router_oof_extended[i] (router's P(v1 right)).

Agreement rows: keep v1 (no blending — both models agree).

Test: same scheme using router_test_p.

Compare to hard-switch H3 versions. Soft blending may avoid the
Pareto-frontier closure by mixing class probabilities continuously
rather than fully delegating to one or the other.
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


def safelog(p, eps=1e-9): return np.log(np.clip(p, eps, 1.0))
def _normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum() > 0:
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    raw_oof = _normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    raw_test = _normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))

    prior = np.bincount(y, minlength=3) / len(y)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    raw_bias, raw_tuned = tune_log_bias(raw_oof, y, prior)

    print(f"v1  OOF tuned={v1_tuned:.5f}  bias={v1_bias.round(4).tolist()}")
    print(f"raw OOF tuned={raw_tuned:.5f}  bias={raw_bias.round(4).tolist()}")

    v1_pred_oof = (safelog(v1_oof) + v1_bias).argmax(1)
    raw_pred_oof = (safelog(raw_oof) + raw_bias).argmax(1)
    disagree_oof = v1_pred_oof != raw_pred_oof

    v1_pred_test = (safelog(v1_test) + v1_bias).argmax(1)
    raw_pred_test = (safelog(raw_test) + raw_bias).argmax(1)
    disagree_test = v1_pred_test != raw_pred_test

    print(f"OOF disagree: {disagree_oof.sum()} / {n_tr}")
    print(f"test disagree: {disagree_test.sum()} / {n_te}")

    # Build full-train router_oof (0.5 default)
    router_oof = np.full(n_tr, 0.5, dtype=np.float32)
    one_right = np.load(ART / "h3_router_one_right_mask.npy").astype(bool)
    rou = np.load(ART / "h3_router_oof.npy").astype(np.float32)
    router_oof[one_right] = rou
    router_test = np.load(ART / "h3_router_test_p.npy").astype(np.float32)

    # Soft-router blend on disagreement rows only.
    # w controls the strength of switching toward raw on low-router rows.
    print("\n=== Soft-router blend: w=router_oof on disagreement rows ===")
    print("(higher router_oof = trust v1 more; lower = trust raw more)")

    # Apply v1's tuned bias post-blend as the decision rule (anchor approach)
    for blend_strength in [0.5, 0.7, 1.0, 1.5, 2.0]:
        # Soft probs on disagreement rows
        # w = clip(router_oof, 0.05, 0.95) ** blend_strength (sharpen)
        # Then probs = w * v1 + (1-w) * raw
        w = np.clip(router_oof, 0.05, 0.95) ** blend_strength
        # Sharpened: more confident routing, less middle-ground

        # Build OOF blended probs
        blend_oof = v1_oof.copy()
        # Disagreement rows get the soft blend
        d_idx = np.where(disagree_oof)[0]
        w_d = w[d_idx][:, None]
        blend_oof[d_idx] = _normed(w_d * v1_oof[d_idx] + (1 - w_d) * raw_oof[d_idx])

        bal = balanced_accuracy_score(y, (safelog(blend_oof) + v1_bias).argmax(1))
        delta = bal - v1_tuned
        pcr = per_class_recall(y, (safelog(blend_oof) + v1_bias).argmax(1))
        print(f"  strength={blend_strength}: bal={bal:.5f}  d={delta:+.5f}  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # Also try: anchor = v1, log-blend the soft on disagreement rows
    print("\n=== Soft-router LOG-blend on disagreement rows ===")
    for alpha in [0.10, 0.20, 0.30, 0.50]:
        # On disagreement rows, log-blend (1-alpha)*v1_log + alpha*soft_log
        # where soft = router_oof * v1 + (1-router_oof) * raw
        soft_oof = v1_oof.copy()
        soft_test = v1_test.copy()
        for arr_v1, arr_raw, arr_router, arr_dis, arr_out in [
            (v1_oof, raw_oof, router_oof, disagree_oof, soft_oof),
            (v1_test, raw_test, router_test, disagree_test, soft_test),
        ]:
            d_idx = np.where(arr_dis)[0]
            w_d = arr_router[d_idx][:, None]
            arr_out[d_idx] = _normed(w_d * arr_v1[d_idx] + (1 - w_d) * arr_raw[d_idx])

        # Now log-blend v1 with soft at alpha
        blend_oof = (1.0 - alpha) * safelog(v1_oof) + alpha * safelog(soft_oof)
        blend_oof = _normed(np.exp(blend_oof))
        bal = balanced_accuracy_score(y, (safelog(blend_oof) + v1_bias).argmax(1))
        delta = bal - v1_tuned
        pcr = per_class_recall(y, (safelog(blend_oof) + v1_bias).argmax(1))
        print(f"  alpha={alpha}: bal={bal:.5f}  d={delta:+.5f}  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # Best emit candidate: try strength=1.0 + standalone (no log-blend)
    print("\n=== Emit best candidate (strength=1.0 soft-router replace on disagreement) ===")
    w = np.clip(router_test, 0.05, 0.95)
    blend_test = v1_test.copy()
    d_idx = np.where(disagree_test)[0]
    w_d = w[d_idx][:, None]
    blend_test[d_idx] = _normed(w_d * v1_test[d_idx] + (1 - w_d) * raw_test[d_idx])
    blend_pred = (safelog(blend_test) + v1_bias).argmax(1)
    diff = int((blend_pred != v1_pred_test).sum())
    print(f"  test diff vs v1: {diff} / {n_te}")

    sub_path = SUB / "submission_h3_soft_router_v1bias.csv"
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in blend_pred],
    })
    sub.to_csv(sub_path, index=False)
    print(f"  wrote {sub_path}")


if __name__ == "__main__":
    main()
