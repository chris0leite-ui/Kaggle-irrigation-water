"""Aggregate-statistics features over a meta-stacker pool of N components.

For each row i, compute 22 scalar features that describe the *uncertainty
geometry* of the pool:
  - per-class {mean, std, max, min, median} across components (3 cls × 5 = 15)
  - entropy of the mean per-row prob vector (1)
  - per-row argmax disagreement count: how many components disagree with the
    modal argmax across the pool (1)
  - per-row std of log(P_M/P_H) margin across components (1)
  - per-row std of log(P_L/P_M) margin across components (1)
  - per-class skew across components (3)

These are bank-aggregate features tier1b's per-component meta-stacker does
NOT see. EDA on 2026-04-26 measured:
  agg ⊥ tier1b meta = AUC 0.6714 on missed-H detection (residual signal).

Inputs:
  stack: ndarray of shape (N_components, N_rows, 3) with normed probs

Returns:
  feats: ndarray (N_rows, 22), dtype float32
  names: list[str] of feature names in column order
"""
from __future__ import annotations

import numpy as np

CLASSES = ["L", "M", "H"]
N_FEATS = 22


def compute_aggregates(stack: np.ndarray) -> tuple[np.ndarray, list[str]]:
    if stack.ndim != 3 or stack.shape[2] != 3:
        raise ValueError(f"expected (N_comp, N_rows, 3), got {stack.shape}")
    N, R, _ = stack.shape
    feats = []
    names = []

    # per-class mean / std / max / min / median (15 cols)
    for k, cls in enumerate(CLASSES):
        pk = stack[:, :, k]  # (N, R)
        feats.append(pk.mean(axis=0))
        names.append(f"agg_mean_{cls}")
        feats.append(pk.std(axis=0))
        names.append(f"agg_std_{cls}")
        feats.append(pk.max(axis=0))
        names.append(f"agg_max_{cls}")
        feats.append(pk.min(axis=0))
        names.append(f"agg_min_{cls}")
        feats.append(np.median(pk, axis=0))
        names.append(f"agg_med_{cls}")

    # entropy of mean per-row prob (1)
    mp = stack.mean(axis=0)  # (R, 3)
    mp_safe = np.clip(mp, 1e-12, 1.0)
    ent = -(mp_safe * np.log(mp_safe)).sum(axis=1)
    feats.append(ent)
    names.append("agg_ent_mean")

    # argmax disagreement count (1)
    argmaxes = stack.argmax(axis=2)  # (N, R)
    # modal argmax per row via bincount along axis=0
    # vectorized: for each row, count argmaxes by class then take argmax
    # using one-hot trick to avoid Python loop
    onehot = np.eye(3, dtype=np.int32)[argmaxes]  # (N, R, 3)
    counts = onehot.sum(axis=0)  # (R, 3)
    modal = counts.argmax(axis=1)  # (R,)
    disagree = (argmaxes != modal[None, :]).sum(axis=0).astype(np.float32)
    feats.append(disagree)
    names.append("agg_disagree_n")

    # per-row std of log(P_M/P_H) and log(P_L/P_M) margins (2)
    eps = 1e-12
    log_mh = np.log(np.clip(stack[:, :, 1], eps, 1)) - np.log(np.clip(stack[:, :, 2], eps, 1))
    log_lm = np.log(np.clip(stack[:, :, 0], eps, 1)) - np.log(np.clip(stack[:, :, 1], eps, 1))
    feats.append(log_mh.std(axis=0))
    names.append("agg_std_logit_MH")
    feats.append(log_lm.std(axis=0))
    names.append("agg_std_logit_LM")

    # per-class skew across components (3) — sample skew via standardized 3rd moment
    for k, cls in enumerate(CLASSES):
        pk = stack[:, :, k]  # (N, R)
        m = pk.mean(axis=0)
        s = pk.std(axis=0) + 1e-12
        skew = ((pk - m[None, :]) / s[None, :]) ** 3
        feats.append(skew.mean(axis=0).astype(np.float32))
        names.append(f"agg_skew_{cls}")

    out = np.stack(feats, axis=1).astype(np.float32)
    if out.shape[1] != N_FEATS:
        raise AssertionError(f"expected {N_FEATS} cols, got {out.shape[1]}")
    return out, names


if __name__ == "__main__":
    # Smoke: a 4-component, 6-row, 3-class fake stack
    rng = np.random.default_rng(0)
    s = rng.dirichlet([1, 1, 1], size=(4, 6))  # (4, 6, 3)
    f, n = compute_aggregates(s.astype(np.float32))
    assert f.shape == (6, N_FEATS)
    print("aggregate names:", n)
    print("shape:", f.shape, "dtype:", f.dtype)
    print("sample row 0:", f[0])
