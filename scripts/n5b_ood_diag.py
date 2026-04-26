"""N5b OOD diagnostic: does 10k-anchored novelty correlate with NN flips?

Senior-DS reframe: 10k original is rule-perfect by construction. NN-flipped
synth rows (~10,304) are the only rows where y != rule(x). Hypothesis: the
NN's smooth boundary perturbs labels for rows whose feature joint lives
off-manifold relative to 10k's training distribution. Test by fitting OOD
scorers on 10k numerics only, score each synth row, correlate with the
known flip indicator |y - rule(x)|.

Decision gate (printed at end):
  PROCEED if any scorer hits |Spearman corr| >= 0.05 OR Cohen's d >= 0.10
  KILL if all scorers below both thresholds.

If PROCEED -> scaffold Option A (kNN-from-10k features). If KILL -> the
10k-as-anchor family is closed; lock the 2 finals and stop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
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
ART = Path("scripts/artifacts"); ART.mkdir(parents=True, exist_ok=True)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt((va + vb) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def report(name: str, scores: np.ndarray, flip: np.ndarray) -> dict:
    s_flip = scores[flip == 1]; s_clean = scores[flip == 0]
    pr = pearsonr(scores, flip)[0]; sr = spearmanr(scores, flip)[0]
    d = cohens_d(s_flip, s_clean)
    out = {
        "scorer": name,
        "pearson_r": round(float(pr), 5),
        "spearman_r": round(float(sr), 5),
        "cohens_d_flip_minus_clean": round(d, 5),
        "mean_score_flipped": round(float(s_flip.mean()), 5),
        "mean_score_clean": round(float(s_clean.mean()), 5),
        "p99_flipped": round(float(np.quantile(s_flip, 0.99)), 5),
        "p99_clean": round(float(np.quantile(s_clean, 0.99)), 5),
    }
    print(f"  {name:18s}  pearson={pr:+.4f}  spearman={sr:+.4f}  d={d:+.4f}")
    return out


def main() -> None:
    print("[1] Loading data...")
    orig = pd.read_csv("data/irrigation_prediction.csv")
    train = pd.read_csv("data/train.csv")
    print(f"    orig {orig.shape}, train {train.shape}")

    print("[2] Computing rule + flip indicator on synth train...")
    rule_train = dgp_predict(train)
    y_train = train["Irrigation_Need"].values
    flip = (y_train != rule_train).astype(int)
    n_flip = int(flip.sum())
    print(f"    flip rate {n_flip}/{len(flip)} = {n_flip/len(flip)*100:.3f}%")

    print("[3] Standardising 11 numerics on 10k stats (no leak)...")
    scaler = StandardScaler().fit(orig[NUM_COLS].values)
    Xo = scaler.transform(orig[NUM_COLS].values)
    Xs = scaler.transform(train[NUM_COLS].values)

    results = []

    print("[4] Scorer A: GMM (n_components=32, diag cov) on 10k...")
    gmm = GaussianMixture(n_components=32, covariance_type="diag",
                           random_state=42, max_iter=200, reg_covar=1e-4)
    gmm.fit(Xo)
    neg_logp = -gmm.score_samples(Xs)  # high = OOD
    results.append(report("GMM_neg_logp", neg_logp, flip))

    print("[5] Scorer B: IsolationForest on 10k...")
    iso = IsolationForest(n_estimators=200, contamination=0.05,
                          random_state=42, n_jobs=-1).fit(Xo)
    iso_score = -iso.score_samples(Xs)  # high = OOD
    results.append(report("IsolationForest", iso_score, flip))

    print("[6] Scorer C: kNN-density (mean dist to k=10 in 10k)...")
    nn = NearestNeighbors(n_neighbors=10, n_jobs=-1).fit(Xo)
    d, _ = nn.kneighbors(Xs)
    knn_dist = d.mean(axis=1)  # high = OOD
    results.append(report("kNN_meandist_k10", knn_dist, flip))

    print("[7] Diagnostic correlation with rule-distance proxy...")
    # Sanity check: a known-good signal (signed dist to nearest threshold)
    # should correlate with flips. Validates pipeline correctness.
    sm_dist = np.abs(train["Soil_Moisture"].values - 25.0)
    rf_dist = np.abs(train["Rainfall_mm"].values - 300.0)
    min_thresh_dist = np.minimum(sm_dist / 5, rf_dist / 50)  # crude scale
    results.append(report("min_thresh_dist*", -min_thresh_dist, flip))

    print("\n[8] VERDICT")
    best_sr = max(abs(r["spearman_r"]) for r in results[:3])  # exclude sanity
    best_d = max(abs(r["cohens_d_flip_minus_clean"]) for r in results[:3])
    proceed = best_sr >= 0.05 or best_d >= 0.10
    verdict = "PROCEED" if proceed else "KILL"
    print(f"  best |spearman| over 3 OOD scorers: {best_sr:.4f}")
    print(f"  best |cohen's d|  over 3 OOD scorers: {best_d:.4f}")
    print(f"  Gates: spearman>=0.05 OR d>=0.10  ->  {verdict}")
    if proceed:
        print("  Recommendation: scaffold Option A (kNN-from-10k features)")
    else:
        print("  Recommendation: 10k-as-anchor family is closed; lock the 2 finals")

    out_path = ART / "n5b_ood_diag_results.json"
    with open(out_path, "w") as f:
        json.dump({"verdict": verdict, "best_abs_spearman": best_sr,
                   "best_abs_cohens_d": best_d, "scorers": results,
                   "n_train": int(len(flip)), "n_flipped": n_flip,
                   "n_orig": int(len(orig))}, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
