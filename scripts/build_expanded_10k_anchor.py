"""Build 9 expanded 10k-anchor features (N5b family expansion).

Existing 11 features (oof_ood3_train + oof_knn10k_train):
  - 3 OOD scorers (GMM, IsoForest, kNN-density on 10k)
  - 8 kNN-from-10k geometric (class fractions, per-class distances, margin)

This script builds 9 MORE 10k-anchor features:
  - 3 per-class GMM density:  log p(x | class=c, 10k_GMM_c) for c in {L, M, H}
  - 3 per-class kNN distance: mean dist to nearest k=10 of 10k of class c
  - 3 Mahalanobis to per-class centroid: ((x - mu_c)' Sigma_c^-1 (x - mu_c))^0.5

Total 10k-anchor family: 11 + 9 = 20 features.

Save to scripts/artifacts/oof_ood9_train.npy + test_ood9.npy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

NUM_COLS = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]
LABELS = ("Low", "Medium", "High")
LABEL_TO_INT = {l: i for i, l in enumerate(LABELS)}
ART = Path("scripts/artifacts")


def main() -> None:
    print("[1] Loading data + standardizing on 10k stats...")
    orig = pd.read_csv("data/irrigation_prediction.csv")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    scaler = StandardScaler().fit(orig[NUM_COLS].values)
    Xo = scaler.transform(orig[NUM_COLS].values)
    Xs_train = scaler.transform(train[NUM_COLS].values)
    Xs_test = scaler.transform(test[NUM_COLS].values)
    yo = np.array([LABEL_TO_INT[l] for l in orig["Irrigation_Need"].values])
    print(f"    orig {Xo.shape}, train {Xs_train.shape}, test {Xs_test.shape}")
    print(f"    orig class counts: L={np.sum(yo==0)} M={np.sum(yo==1)} H={np.sum(yo==2)}")

    # 9 features = 3 per-class GMM + 3 per-class kNN + 3 Mahalanobis
    feats_train = np.zeros((len(Xs_train), 9), dtype=np.float32)
    feats_test = np.zeros((len(Xs_test), 9), dtype=np.float32)

    print("\n[2] Per-class GMM densities (8 components per class, diag cov)...")
    for c, name in enumerate(LABELS):
        Xc = Xo[yo == c]
        n_comp = min(8, len(Xc) // 30) if len(Xc) >= 60 else max(2, len(Xc) // 20)
        n_comp = max(2, n_comp)
        gmm_c = GaussianMixture(n_components=n_comp, covariance_type="diag",
                                 random_state=42, max_iter=200, reg_covar=1e-3)
        gmm_c.fit(Xc)
        feats_train[:, c] = -gmm_c.score_samples(Xs_train)
        feats_test[:, c] = -gmm_c.score_samples(Xs_test)
        print(f"    class {name} (n={len(Xc)}, {n_comp} comps): "
              f"train neg-logp mean={feats_train[:, c].mean():.3f}")

    print("\n[3] Per-class kNN distances (k=10, mean dist to nearest of class c)...")
    for c, name in enumerate(LABELS):
        Xc = Xo[yo == c]
        k = min(10, len(Xc))
        nn_c = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(Xc)
        d_tr, _ = nn_c.kneighbors(Xs_train)
        d_te, _ = nn_c.kneighbors(Xs_test)
        feats_train[:, 3 + c] = d_tr.mean(axis=1)
        feats_test[:, 3 + c] = d_te.mean(axis=1)
        print(f"    class {name}: train mean dist={feats_train[:, 3+c].mean():.3f}")

    print("\n[4] Mahalanobis distance to per-class centroid (diag cov approx)...")
    for c, name in enumerate(LABELS):
        Xc = Xo[yo == c]
        mu_c = Xc.mean(axis=0)
        var_c = Xc.var(axis=0) + 1e-3  # ridge
        # Mahalanobis = sqrt((x - mu)' diag(1/var) (x - mu)) = sqrt(sum (x_i - mu_i)^2 / var_i)
        d_tr = np.sqrt(((Xs_train - mu_c) ** 2 / var_c).sum(axis=1))
        d_te = np.sqrt(((Xs_test - mu_c) ** 2 / var_c).sum(axis=1))
        feats_train[:, 6 + c] = d_tr
        feats_test[:, 6 + c] = d_te
        print(f"    class {name}: train mean Mahalanobis={d_tr.mean():.3f}")

    print("\n[5] Sanity diagnostic — correlate each feature with PRIMARY error indicator...")
    from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed
    BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
    y = load_y()
    s3_o, _ = build_lbbest_stack(y)
    ms_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    ms_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    ms_o_iso, _ = iso_cal(ms_o, ms_t, y)

    def log_blend(probs_list, weights):
        s = np.zeros_like(probs_list[0])
        for p, w in zip(probs_list, weights):
            s = s + w * np.log(np.clip(p, 1e-12, 1))
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    p_primary = log_blend([s3_o, ms_o_iso], np.array([0.70, 0.30]))
    pred_primary = (np.log(np.clip(p_primary, 1e-12, 1)) + BIAS).argmax(1)
    err = (y != pred_primary).astype(int)
    feature_names = [
        "gmm_L", "gmm_M", "gmm_H",
        "knn_d_L", "knn_d_M", "knn_d_H",
        "maha_L", "maha_M", "maha_H",
    ]
    for i, name in enumerate(feature_names):
        from scipy.stats import spearmanr
        sr = spearmanr(feats_train[:, i], err)[0]
        print(f"    {name:8s}  Spearman vs (y!=primary_argmax) = {sr:+.4f}")

    np.save(ART / "oof_ood9_train.npy", feats_train)
    np.save(ART / "test_ood9.npy", feats_test)
    print(f"\nSaved -> oof_ood9_train.npy {feats_train.shape}, test_ood9.npy {feats_test.shape}")


if __name__ == "__main__":
    main()
