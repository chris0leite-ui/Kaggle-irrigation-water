"""H3 OOF validation: simulate router deployment on OOF disagreements,
verify macro-recall lift vs v1 standalone before LB probing."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
SEED = 42


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

    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    raw_oof = _normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))

    prior = np.bincount(y, minlength=3) / len(y)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    raw_bias, raw_tuned = tune_log_bias(raw_oof, y, prior)

    v1_pred = (safelog(v1_oof) + v1_bias).argmax(1)
    raw_pred = (safelog(raw_oof) + raw_bias).argmax(1)
    disagree = v1_pred != raw_pred

    print(f"v1  OOF tuned={v1_tuned:.5f}")
    print(f"raw OOF tuned={raw_tuned:.5f}")
    pcr_v1 = per_class_recall(y, v1_pred)
    print(f"v1 PCR=[L={pcr_v1[0]:.4f} M={pcr_v1[1]:.4f} H={pcr_v1[2]:.4f}]")

    # Need to redo router OOF for validation. Actually the previous run
    # didn't save router_oof for the full train set. Let me reconstruct.
    # The H3 script saved router_oof on the disagreement-only training set.
    # We need to apply the gate to ALL disagreement rows via 5-fold OOF gate.

    # Simpler approach: load h3_router_test_p.npy (test gate predictions)
    # AND simulate via random sampling on OOF disagreements.
    # Even simpler: load the per-fold router OOFs, but those weren't saved.

    # Re-run quick-and-dirty: use logistic on simple features
    print("\n=== Quick-router OOF simulation ===")
    print("(using v1_max_prob - raw_max_prob and per-class diff as gate)")

    # Heuristic gate features: where v1 confidence is much higher than raw, prefer v1
    v1_conf = v1_oof.max(1)
    raw_conf = raw_oof.max(1)
    gate_score = v1_conf - raw_conf  # >0 favors v1
    print(f"OOF disagreement gate-score: median={np.median(gate_score[disagree]):.3f}")

    # OOF: what if we hard-switch on disagreements where gate < threshold to raw?
    print("\n=== Naive gate sweep (v1_max_prob - raw_max_prob) ===")
    for tau in [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10]:
        # Switch to raw where (v1_conf - raw_conf) < tau
        switch_mask = disagree & (gate_score < tau)
        new_pred = v1_pred.copy()
        new_pred[switch_mask] = raw_pred[switch_mask]
        from sklearn.metrics import balanced_accuracy_score
        bal = balanced_accuracy_score(y, new_pred)
        n_sw = switch_mask.sum()
        right_sw = ((raw_pred[switch_mask] == y[switch_mask])).sum()
        prec = right_sw / max(1, n_sw)
        delta = bal - v1_tuned
        pcr = per_class_recall(y, new_pred)
        print(f"  tau={tau:+.2f}: n_sw={n_sw}  prec={prec:.3f}  bal={bal:.5f} d={delta:+.5f}  PCR=[L={pcr[0]:.4f} M={pcr[1]:.4f} H={pcr[2]:.4f}]")

    # Macro-recall sensitivity calc
    n_L, n_M, n_H = np.bincount(y, minlength=3)
    print(f"\nClass counts: L={n_L} M={n_M} H={n_H}")
    print(f"Per-correct-flip macro-recall delta: L={1/(3*n_L):+.2e} M={1/(3*n_M):+.2e} H={1/(3*n_H):+.2e}")


if __name__ == "__main__":
    main()
