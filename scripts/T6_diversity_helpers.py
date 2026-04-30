"""T6 helpers — diversity-penalized forward selection over saved OOFs.

Caruana 2004 ensemble selection with diversity term:
  add_step:  argmax_m  [val_macro_recall(ens + m) - beta * argmax_jaccard(m, ens)]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
DATA = Path("data")


def load_y_train() -> np.ndarray:
    return pd.read_csv(DATA / "train.csv")["Irrigation_Need"].map(
        {"Low": 0, "Medium": 1, "High": 2}
    ).to_numpy(dtype=np.int8)


def list_oof_names(min_rows: int = 600000) -> list[str]:
    """List candidate OOF names where shape is (n_train, 3)."""
    out = []
    for p in sorted(ART.glob("oof_*.npy")):
        try:
            arr = np.load(p, mmap_mode="r")
            if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= min_rows:
                out.append(p.stem.replace("oof_", "", 1))
        except Exception:
            pass
    return out


def macro_recall(y_true: np.ndarray, y_pred: np.ndarray, n_cls: int = 3) -> float:
    rec = []
    for c in range(n_cls):
        m = y_true == c
        if m.sum() == 0:
            continue
        rec.append(float((y_pred[m] == c).mean()))
    return sum(rec) / len(rec)


def normed(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return p / np.clip(p.sum(axis=1, keepdims=True), eps, None)


def argmax_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between argmax of two prob arrays (n,3)."""
    aa = a.argmax(1)
    bb = b.argmax(1)
    same = (aa == bb).sum()
    return float(same / len(aa))


def tune_log_bias_simple(probs: np.ndarray, y: np.ndarray, n_steps: int = 31) -> tuple[np.ndarray, float]:
    """Coord-ascent log-bias for 3 classes."""
    best = np.zeros(3, dtype=np.float64)
    log_p = np.log(np.clip(probs, 1e-9, None))
    pred = (log_p + best).argmax(1)
    best_score = macro_recall(y, pred)

    for _ in range(3):
        improved = False
        for c in range(3):
            grid = np.linspace(best[c] - 1.0, best[c] + 4.0, n_steps)
            for v in grid:
                trial = best.copy()
                trial[c] = v
                s = macro_recall(y, (log_p + trial).argmax(1))
                if s > best_score:
                    best_score = s
                    best = trial
                    improved = True
        if not improved:
            break
    return best, best_score
