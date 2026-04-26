"""Build all 10k-anchored features once (OOD scores + kNN-from-10k features).

Saves to scripts/artifacts/:
  oof_ood3_train.npy   shape (630000, 3)   GMM_neg_logp, IsoForest, kNN_dist
  test_ood3.npy        shape (270000, 3)   same 3 cols on test
  oof_knn10k_train.npy shape (630000, 8)   8 kNN-from-10k geometric features
  test_knn10k.npy      shape (270000, 8)   same 8 on test

Reusable across deployments #1, #2, #3.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from dgp_formula import dgp_predict

NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
LABELS = ("Low", "Medium", "High")
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}
ART = Path("scripts/artifacts"); ART.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("[1] Loading data...")
    orig = pd.read_csv("data/irrigation_prediction.csv")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"    orig {orig.shape}, train {train.shape}, test {test.shape}")

    print("[2] Standardising 11 numerics on 10k stats only (no leak)...")
    scaler = StandardScaler().fit(orig[NUM_COLS].values)
    Xo = scaler.transform(orig[NUM_COLS].values)
    Xs_train = scaler.transform(train[NUM_COLS].values)
    Xs_test = scaler.transform(test[NUM_COLS].values)
    yo = np.array([LABEL_TO_INT[l] for l in orig["Irrigation_Need"].values])

    print("[3] OOD scorer A: GaussianMixture (32 comps, diag) on 10k...")
    gmm = GaussianMixture(n_components=32, covariance_type="diag",
                           random_state=42, max_iter=200, reg_covar=1e-4)
    gmm.fit(Xo)
    gmm_train = -gmm.score_samples(Xs_train)
    gmm_test = -gmm.score_samples(Xs_test)
    print(f"    train mean={gmm_train.mean():.3f} test mean={gmm_test.mean():.3f}")

    print("[4] OOD scorer B: IsolationForest on 10k...")
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=-1).fit(Xo)
    iso_train = -iso.score_samples(Xs_train)
    iso_test = -iso.score_samples(Xs_test)

    print("[5] OOD scorer C: kNN-density (k=10 mean dist to 10k)...")
    nn10 = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(Xo)
    d_train, _ = nn10.kneighbors(Xs_train)
    d_test, _ = nn10.kneighbors(Xs_test)
    knn_train = d_train.mean(axis=1)
    knn_test = d_test.mean(axis=1)

    ood3_train = np.stack([gmm_train, iso_train, knn_train], axis=1).astype(np.float32)
    ood3_test = np.stack([gmm_test, iso_test, knn_test], axis=1).astype(np.float32)
    np.save(ART / "oof_ood3_train.npy", ood3_train)
    np.save(ART / "test_ood3.npy", ood3_test)
    print(f"    -> oof_ood3_train.npy {ood3_train.shape}, test_ood3.npy {ood3_test.shape}")

    print("[6] kNN-from-10k geometric features (k=20)...")
    nn20 = NearestNeighbors(n_neighbors=20, n_jobs=-1).fit(Xo)
    rule_train = dgp_predict(train)
    rule_test = dgp_predict(test)
    rule_train_int = np.array([LABEL_TO_INT[l] for l in rule_train])
    rule_test_int = np.array([LABEL_TO_INT[l] for l in rule_test])

    def build_knn_feats(X: np.ndarray, rule_int: np.ndarray) -> np.ndarray:
        d_all, idx_all = nn20.kneighbors(X)  # d, idx shape (N, 20)
        nbr_y = yo[idx_all]                  # (N, 20) int labels
        N = X.shape[0]
        out = np.zeros((N, 8), dtype=np.float32)
        for c in range(3):
            out[:, c] = (nbr_y == c).mean(axis=1)  # p_low, p_med, p_high
        out[:, 3] = nbr_y[np.arange(N), 0]         # majority of nearest = nbr-0 label
        # mean-dist to nearest 10k rows of EACH class (mask + mean; nan->fallback)
        for c in range(3):
            mask = (nbr_y == c)
            ds = np.where(mask, d_all, np.nan)
            with np.errstate(invalid="ignore"):
                col = np.nanmean(ds, axis=1)
            col = np.where(np.isfinite(col), col, d_all.max())
            out[:, 4 + c] = col
        # margin: d_to_rule_class - min(d_to_other_class)
        rule_d = out[np.arange(N), 4 + rule_int]
        d_classes = out[:, 4:7]
        # mask out rule class then take min
        d_other = d_classes.copy()
        d_other[np.arange(N), rule_int] = np.inf
        out[:, 7] = rule_d - d_other.min(axis=1)
        return out

    knn_train = build_knn_feats(Xs_train, rule_train_int)
    knn_test = build_knn_feats(Xs_test, rule_test_int)
    np.save(ART / "oof_knn10k_train.npy", knn_train)
    np.save(ART / "test_knn10k.npy", knn_test)
    print(f"    -> oof_knn10k_train.npy {knn_train.shape}, test_knn10k.npy {knn_test.shape}")
    print(f"    knn_train col-means: {knn_train.mean(axis=0).round(4)}")

    print("\nDone. Artifacts saved.")


if __name__ == "__main__":
    main()
