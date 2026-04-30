"""Build T4 pseudo-label CSV: filter v1 RF natural test predictions at tau=0.99
after applying v1's tuned bias. Output: data/t4_pseudo_labels.csv with
columns (id, Irrigation_Need_pseudo, max_prob)."""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias

ART = Path("scripts/artifacts")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

TAU = 0.99


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    prior = np.bincount(y, minlength=3) / len(y)

    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)

    bias, tuned = tune_log_bias(v1_oof, y, prior)
    print(f"v1 bias={bias.round(4).tolist()} tuned={tuned:.5f}")

    # Apply v1's bias to test probs
    test_logp = safelog(v1_test) + bias
    # Softmax-normalize for max_prob calculation
    z = test_logp - test_logp.max(1, keepdims=True)
    test_p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    test_pred = test_p.argmax(1)
    test_max = test_p.max(1)

    keep = test_max >= TAU
    print(f"Kept {keep.sum()} / {len(test)} ({100*keep.mean():.1f}%) at tau={TAU}")
    cls_dist = np.bincount(test_pred[keep], minlength=3)
    print(f"Class dist: L={cls_dist[0]} M={cls_dist[1]} H={cls_dist[2]}")

    out = pd.DataFrame({
        "id": test["id"].values[keep],
        "Irrigation_Need_pseudo": [IDX2CLS[c] for c in test_pred[keep]],
        "max_prob": test_max[keep].astype(np.float32),
    })
    out_p = Path("data/t4_pseudo_labels.csv")
    out.to_csv(out_p, index=False)
    print(f"wrote {out_p} ({len(out)} rows)")


if __name__ == "__main__":
    main()
