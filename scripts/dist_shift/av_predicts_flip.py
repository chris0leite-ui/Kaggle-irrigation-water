"""Probe — does AV-classifier P(synth | row) correlate with rule-flip
presence on train OOF?

If AUC(P_synth, flip) >> 0.55 on the synth half of the AV oof, the
drift signature carries flip-detection information that NO existing
recipe feature uses. This is the gating diagnostic for the option
family that builds an FE leg around the AV score.

Compares against:
- N5b GMM/IsoForest/kNN density on orig features (Cohen's d ~ 0.20,
  Spearman 0.024 — barely above noise).
- 2026-04-21 within-flip Cohen's d on non-rule features (Humidity
  +0.076 at score=3, Prev_Irrig +0.107 at score=3).

If our AV-based score has AUC > 0.55, it's the strongest single
diagnostic on this problem.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats

from scripts.dist_shift.loader import ARTI, load


def _bits(df):
    return {
        "dry": (df["Soil_Moisture"] < 25).astype(int),
        "norain": (df["Rainfall_mm"] < 300).astype(int),
        "hot": (df["Temperature_C"] > 30).astype(int),
        "windy": (df["Wind_Speed_kmh"] > 10).astype(int),
        "nomulch": (df["Mulching_Used"] == "No").astype(int),
        "kc": df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2,
    }


def _score(b):
    return 2 * (b["dry"] + b["norain"]) + b["hot"] + b["windy"] + b["nomulch"] + b["kc"]


def _rule(score):
    out = np.full(len(score), "Medium", dtype=object)
    out[score <= 3] = "Low"
    out[score >= 7] = "High"
    return out


def main() -> None:
    av_oof = np.load(ARTI / "av_oof.npy")
    train, _, orig = load()
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(train), size=10000, replace=False)
    synth_sub = train.iloc[sub_idx].reset_index(drop=True)

    # AV oof layout: first 10000 rows = orig, last 10000 rows = synth_sub
    p_synth_on_synth = av_oof[10000:]
    p_synth_on_orig = av_oof[:10000]

    print(f"AV P(synth) percentiles on synth rows:    {np.percentile(p_synth_on_synth, [1, 25, 50, 75, 99]).round(3)}")
    print(f"AV P(synth) percentiles on orig rows:     {np.percentile(p_synth_on_orig, [1, 25, 50, 75, 99]).round(3)}")

    # rule-flip on the synth subsample
    b = _bits(synth_sub)
    s = _score(b).to_numpy()
    rule = _rule(s)
    flip = (rule != synth_sub["Irrigation_Need"].values)
    print(f"\nsynth subsample n={len(synth_sub)} flip count = {flip.sum()} ({flip.mean()*100:.3f}%)")

    # AUC of P(synth) for predicting flip
    auc = roc_auc_score(flip.astype(int), p_synth_on_synth)
    print(f"\n=== HEADLINE: AUC(P(synth), flip) on synth subsample = {auc:.4f} ===")

    # Cohen's d: P(synth) on flip vs clean
    a = p_synth_on_synth[~flip]
    bb = p_synth_on_synth[flip]
    pooled = np.sqrt(0.5 * (a.var(ddof=1) + bb.var(ddof=1))) if len(bb) > 1 else 0.0
    d = (bb.mean() - a.mean()) / pooled if pooled > 0 else 0.0
    ks = stats.ks_2samp(a, bb)
    print(f"\nP(synth) flip vs clean:")
    print(f"  mean clean = {a.mean():.4f}  std = {a.std(ddof=1):.4f}  n = {len(a)}")
    print(f"  mean flip  = {bb.mean():.4f}  std = {bb.std(ddof=1):.4f}  n = {len(bb)}")
    print(f"  Cohen's d  = {d:.3f}     KS = {ks.statistic:.3f}  p = {ks.pvalue:.2e}")

    # Per-score breakdown — is the signal strong specifically at flip-prone scores?
    print("\nPer-score AUC of P(synth) for flip:")
    rows = []
    for sc in sorted(set(s)):
        mask = (s == sc)
        if mask.sum() < 20:
            continue
        n_flip_sc = int(flip[mask].sum())
        if n_flip_sc < 5 or n_flip_sc == mask.sum():
            rows.append({"score": int(sc), "n": int(mask.sum()), "n_flip": n_flip_sc, "auc": "n/a"})
            continue
        a_sc = roc_auc_score(flip[mask].astype(int), p_synth_on_synth[mask])
        rows.append({"score": int(sc), "n": int(mask.sum()), "n_flip": n_flip_sc, "auc": round(float(a_sc), 4)})
    print(pd.DataFrame(rows).to_string(index=False))

    out = {
        "auc_av_p_synth_predicts_flip": float(auc),
        "cohen_d_av_p_synth_flip_vs_clean": float(d),
        "n_synth_subsample": int(len(synth_sub)),
        "n_flips_in_subsample": int(flip.sum()),
        "p_synth_mean_clean": float(a.mean()),
        "p_synth_mean_flip": float(bb.mean()),
        "per_score_auc": rows,
    }
    (ARTI / "av_predicts_flip_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {ARTI/'av_predicts_flip_results.json'}")


if __name__ == "__main__":
    main()
