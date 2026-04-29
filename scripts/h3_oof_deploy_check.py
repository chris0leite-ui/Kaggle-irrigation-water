"""H3 deployment OOF check: simulate router-gated v1<>raw on OOF.

The router was trained on disagreement-only rows (n=1858 OOF).
Per-fold OOF predictions saved at h3_router_oof.npy.

For each tau, simulate test deployment by switching v1->raw on
disagreement rows where router_p < tau, compute bal_acc + per-class
recall + macro lift vs v1 standalone.

This validates whether the deployment will lift LB before probing.
"""
from __future__ import annotations

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
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    n_tr = len(y)

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    raw_oof = _normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    prior = np.bincount(y, minlength=3) / len(y)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    raw_bias, _ = tune_log_bias(raw_oof, y, prior)

    v1_pred = (safelog(v1_oof) + v1_bias).argmax(1)
    raw_pred = (safelog(raw_oof) + raw_bias).argmax(1)
    disagree = v1_pred != raw_pred

    # Load router OOF (only fits disagreement+exactly-one-right rows)
    router_oof = np.load(ART / "h3_router_oof.npy").astype(np.float32)
    one_right_mask = np.load(ART / "h3_router_one_right_mask.npy").astype(bool)
    router_y = np.load(ART / "h3_router_y.npy").astype(np.int32)

    print(f"v1  OOF tuned={v1_tuned:.5f}  bias={v1_bias.round(4).tolist()}")
    print(f"OOF disagreement rows: {disagree.sum()} / {n_tr}")
    print(f"  exactly-one-right rows: {one_right_mask.sum()} (router train set)")
    print(f"router OOF AUC: {(router_oof.argsort().argsort() / len(router_oof) * (1 + 0))[router_y == 1].mean():.4f} (rank-AUC)")

    # Build a full-train router_oof_extended[i]:
    # - For one_right_mask rows: router_oof
    # - For other rows (disagreement but both right or both wrong): set to 0.5 (neutral)
    # - For agreement rows: not used
    router_full = np.full(n_tr, 0.5, dtype=np.float32)
    router_full[one_right_mask] = router_oof

    # Simulate deployment: at tau, switch v1->raw on disagreement rows
    # where router_p < tau
    print("\n=== OOF deployment simulation ===")
    print("(switch v1 -> raw on disagreement rows where router_oof < tau)")
    print(f"Baseline v1 OOF tuned: {v1_tuned:.5f}")
    print(f"v1 PCR=[L={per_class_recall(y, v1_pred)[0]:.4f} M={per_class_recall(y, v1_pred)[1]:.4f} H={per_class_recall(y, v1_pred)[2]:.4f}]")

    for tau in [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        switch_mask = disagree & (router_full < tau)
        n_sw = switch_mask.sum()
        new_pred = v1_pred.copy()
        new_pred[switch_mask] = raw_pred[switch_mask]
        bal = balanced_accuracy_score(y, new_pred)
        delta = bal - v1_tuned

        # Decompose: of the switched rows, how many were correct?
        sw_idx = np.where(switch_mask)[0]
        n_correct_sw = (raw_pred[sw_idx] == y[sw_idx]).sum()
        n_v1_was_correct = (v1_pred[sw_idx] == y[sw_idx]).sum()
        prec = n_correct_sw / max(1, n_sw)

        pcr = per_class_recall(y, new_pred)
        print(f"  tau={tau:.2f}: n_sw={n_sw:4d}  prec={prec:.3f}  v1_was_right={n_v1_was_correct:3d}  bal={bal:.5f}  d={delta:+.5f}  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # Also try the converse: switch v1 -> raw on rows where router > tau (router prefers raw)
    # NB: in our convention, target=1 means v1=right, so switching to raw should be WHEN router_p IS LOW.
    print("\n=== Reverse direction: switch v1 -> raw on disagreement where router > tau (gate prefers v1) ===")
    print("(this is wrong direction; included as sanity check)")
    for tau in [0.50, 0.60, 0.70, 0.80, 0.90]:
        switch_mask = disagree & (router_full > tau)
        n_sw = switch_mask.sum()
        new_pred = v1_pred.copy()
        new_pred[switch_mask] = raw_pred[switch_mask]
        bal = balanced_accuracy_score(y, new_pred)
        delta = bal - v1_tuned
        print(f"  tau={tau:.2f}: n_sw={n_sw:4d}  bal={bal:.5f}  d={delta:+.5f}")


if __name__ == "__main__":
    main()
