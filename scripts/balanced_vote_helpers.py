"""Helpers for balanced-accuracy-aware vote experiments on a CLEANED submission bank.

Three vote variants:
  V1 — rare-class-favoring hard-vote: argmax of per-class vote count, with
       tie-break High > Medium > Low (rare-class wins ties under macro-recall).
  V2 — confidence-weighted hard-vote: each voter's argmax contributes
       max_prob to its argmax class; predict argmax of summed weights.
  V3 — 1/pi_c-weighted soft-vote: arithmetic mean of probs, multiply class
       column by 1/pi_c, renormalize, argmax (then optional log-bias retune).

Each variant produces a (n_rows,) int prediction array. Score it against
the LB-best primary at recipe bias [1.4324, 1.4689, 3.4008].
"""
from __future__ import annotations
import numpy as np

from common import fast_bal_acc

EPS = 1e-12
PI = np.array([0.5872, 0.3795, 0.0333])  # train class prior (Low/Med/High)


def hard_vote_asymmetric(probs_list: list[np.ndarray]) -> np.ndarray:
    """V1: rare-class-favoring hard-vote. argmax with tie-break H > M > L.

    probs_list: list of (n_rows, 3) prob matrices, one per voter.
    """
    args = np.stack([p.argmax(1) for p in probs_list], axis=1)  # (n, V)
    n_class = 3
    counts = np.zeros((args.shape[0], n_class), dtype=np.int32)
    for k in range(n_class):
        counts[:, k] = (args == k).sum(axis=1)
    # Tie-break: High > Medium > Low. Add tiny offsets so ties resolve toward H.
    # Offset must be < 1 vote so it can't override actual majorities.
    offsets = np.array([0.0, 0.1, 0.2])  # H gets largest, L gets 0
    # Re-map: class index 2 = High, 1 = Medium, 0 = Low.
    # offsets[k] for class k; we want H (k=2) largest, M (k=1) middle, L (k=0) lowest.
    offsets = np.array([0.0, 0.1, 0.2])
    return (counts.astype(np.float32) + offsets).argmax(axis=1)


def hard_vote_confidence(probs_list: list[np.ndarray]) -> np.ndarray:
    """V2: confidence-weighted hard-vote.

    Each voter contributes weight = max_prob to its argmax class.
    Predict argmax of summed weights per class.
    """
    n_rows = probs_list[0].shape[0]
    weights = np.zeros((n_rows, 3), dtype=np.float32)
    for p in probs_list:
        a = p.argmax(1)
        m = p.max(1)
        # Scatter-add: for each row, add m to the argmax-class column.
        weights[np.arange(n_rows), a] += m
    return weights.argmax(axis=1)


def soft_vote_class_weighted(probs_list: list[np.ndarray]) -> np.ndarray:
    """V3: 1/pi_c-weighted soft-vote, returns prob matrix (not argmax).

    Mean probs across voters, then divide each class column by pi_c and
    renormalize. Caller can argmax directly or apply log-bias.
    """
    avg = np.mean(np.stack(probs_list, axis=0), axis=0)
    weighted = avg / PI[None, :]
    return weighted / weighted.sum(axis=1, keepdims=True)


def score_predictions(pred: np.ndarray, y: np.ndarray, anchor_pred: np.ndarray,
                      label: str = "") -> dict:
    """4-metric scorecard for a prediction array against an anchor.

    Returns: bal_acc, per-class recall delta vs anchor, error count delta,
    net rare-class flip (positive = added rare-class predictions).
    """
    cc = np.bincount(y, minlength=3)
    bal_p = fast_bal_acc(y, pred, class_counts=cc)
    bal_a = fast_bal_acc(y, anchor_pred, class_counts=cc)
    err_p = int((pred != y).sum())
    err_a = int((anchor_pred != y).sum())
    rec_p = np.array([(pred[y == k] == k).mean() for k in range(3)])
    rec_a = np.array([(anchor_pred[y == k] == k).mean() for k in range(3)])
    # Net rare-class flip: how many High predictions added vs anchor on test-rows
    # we approximate via OOF differences (predicted-class counts).
    add_h = int(((pred == 2) & (anchor_pred != 2)).sum())
    rem_h = int(((pred != 2) & (anchor_pred == 2)).sum())
    net_h = add_h - rem_h
    churn_h = add_h + rem_h
    asym = (net_h / max(churn_h, 1)) if churn_h > 0 else 0.0
    return {
        "label": label,
        "bal_acc": float(bal_p),
        "delta_bal": float(bal_p - bal_a),
        "errs": err_p,
        "delta_errs": err_p - err_a,
        "rec_L": float(rec_p[0]), "rec_M": float(rec_p[1]), "rec_H": float(rec_p[2]),
        "delta_rec_L": float(rec_p[0] - rec_a[0]),
        "delta_rec_M": float(rec_p[1] - rec_a[1]),
        "delta_rec_H": float(rec_p[2] - rec_a[2]),
        "add_H": add_h, "rem_H": rem_h, "net_H": net_h, "asym": float(asym),
    }


def gate_check(score: dict, gate_g1_min: float = 2e-4,
               gate_g2_floor: float = -5e-4,
               gate_g3_max_err_ratio: float = 1.05,
               gate_g4_min_asym: float = 0.5,
               anchor_errs: int = 9415) -> dict:
    """Apply 4-gate filter to a scorecard. Returns pass/fail per gate + overall."""
    g1 = score["delta_bal"] >= gate_g1_min
    g2 = (score["delta_rec_L"] >= gate_g2_floor and
          score["delta_rec_M"] >= gate_g2_floor and
          score["delta_rec_H"] >= gate_g2_floor)
    g3 = score["errs"] <= gate_g3_max_err_ratio * anchor_errs
    g4 = (score["net_H"] >= 0 and score["asym"] >= gate_g4_min_asym)
    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4,
            "all_pass": g1 and g2 and g3 and g4}
