"""FAISS k-NN distance features w.r.t. 10k rule-perfect original.

For each row in train + test, find the k nearest neighbors in the
10k original dataset (rule-perfect labels by construction). Emit:
  - dist_min:  L2 distance to nearest neighbor
  - dist_mean: mean L2 distance over k neighbors
  - frac_low / frac_med / frac_high: fraction of k-NN with each label

That's 5 features. The motivation: distance to the rule-perfect
manifold is a clean "rule-conformity" signal that recipe FE encodes
implicitly only through the rule indicators + dgp_score. Direct
distance gives the model a continuous signal at finer granularity.

Standardize features (z-score on combined train+test+orig) BEFORE
FAISS so distance is a euclidean-on-z-scored space — this matches
how the host's NN learned its label boundary in normalized space
(per the 2026-04-21 DGP-residuals EDA).

Wall: ~2-3 min on 16-core CPU for 630k+270k queries against 10k
neighbors via faiss.IndexFlatL2 (no quantization needed at this
scale). Runs as a one-shot pre-computation; saves npy files that
recipe_full_te.py loads via NN_DIST_PATH env var.
"""
from __future__ import annotations

import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
TARGET = "Irrigation_Need"
ART = Path("scripts/artifacts")
ART.mkdir(exist_ok=True, parents=True)
K = 16  # neighbors

NUMERIC_COLS = [
    "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
    "Humidity", "Soil_pH", "Organic_Carbon", "Electrical_Conductivity",
    "Sunlight_Hours", "Field_Area_hectare", "Previous_Irrigation_mm",
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def standardize_jointly(train, test, orig, cols):
    """z-score on combined train+test+orig (no leakage; standardization
    is unsupervised). Returns numpy arrays (n, len(cols)) float32."""
    combined = pd.concat([train[cols], test[cols], orig[cols]], axis=0)
    mu = combined.mean(axis=0).values
    sd = combined.std(axis=0).values + 1e-9
    Xtr = ((train[cols].values - mu) / sd).astype(np.float32)
    Xte = ((test[cols].values - mu) / sd).astype(np.float32)
    Xor = ((orig[cols].values - mu) / sd).astype(np.float32)
    return Xtr, Xte, Xor


def build_features(Xtr, Xte, Xor, y_orig, k=K):
    """FAISS IndexFlatL2 over Xor (10k rule-perfect). Query Xtr + Xte.

    Returns (feats_train, feats_test) each (n, 5) float32:
      [dist_min, dist_mean, frac_low, frac_med, frac_high]
    """
    d = Xor.shape[1]
    idx = faiss.IndexFlatL2(d)
    idx.add(np.ascontiguousarray(Xor))

    def _query(X):
        # FAISS returns squared L2; sqrt for interpretability.
        D, I = idx.search(np.ascontiguousarray(X), k)
        D = np.sqrt(np.clip(D, 0, None))
        nn_labels = y_orig[I]  # (n, k)
        out = np.zeros((X.shape[0], 5), dtype=np.float32)
        out[:, 0] = D[:, 0]
        out[:, 1] = D.mean(axis=1)
        for c in (0, 1, 2):
            out[:, 2 + c] = (nn_labels == c).mean(axis=1)
        return out

    log(f"  querying {Xtr.shape[0]:,} train rows over {d}-d, k={k}")
    t0 = time.time()
    feats_tr = _query(Xtr)
    log(f"    train done in {time.time()-t0:.1f}s")
    log(f"  querying {Xte.shape[0]:,} test rows")
    t0 = time.time()
    feats_te = _query(Xte)
    log(f"    test done in {time.time()-t0:.1f}s")
    return feats_tr, feats_te


def main():
    log(f"loading train / test / orig (k={K} neighbors)")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/archive.zip")
    y_orig = orig[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    log(f"  train={len(train):,}  test={len(test):,}  orig={len(orig):,}")

    log("z-scoring 11 numeric features jointly across train+test+orig")
    Xtr, Xte, Xor = standardize_jointly(train, test, orig, NUMERIC_COLS)

    feats_tr, feats_te = build_features(Xtr, Xte, Xor, y_orig)

    out_oof = ART / "oof_nn_dist_features.npy"
    out_test = ART / "test_nn_dist_features.npy"
    np.save(out_oof, feats_tr)
    np.save(out_test, feats_te)
    log(f"wrote {out_oof}  shape={feats_tr.shape}")
    log(f"wrote {out_test}  shape={feats_te.shape}")
    # Diagnostic: per-class distance summaries on train.
    y_tr = train[TARGET].map(CLS_MAP).to_numpy()
    for c, name in enumerate(("Low", "Medium", "High")):
        mask = y_tr == c
        log(f"  class {name:<6} n={mask.sum():>6}  "
            f"dist_min mean={feats_tr[mask, 0].mean():.3f}  "
            f"frac_match_class mean={feats_tr[mask, 2 + c].mean():.3f}")


if __name__ == "__main__":
    main()
