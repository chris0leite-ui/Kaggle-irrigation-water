"""Preflight — full-train + full-test P(synth) from AV classifier.

Strategy for leak-free OOF on full 630k train:
- For the 10k AV-training subsample (sub_idx, seed=42): use the AV
  5-fold OOF predictions already on disk (av_oof.npy holds all 20k;
  rows 10000-19999 are the synth subsample's OOFs).
- For the remaining 620k train rows: predict from a full-fit AV
  classifier (orig 10k + train_sub 10k) which never saw any of them.
- For the 270k test: same full-fit AV classifier.

Output:
  oof_av_p_synth_train.npy  (630000,)   — leak-free P(synth | train_row)
  test_av_p_synth.npy       (270000,)   — full-fit P(synth | test_row)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.dist_shift.av import _build_features  # reuse exact FE
from scripts.dist_shift.loader import ARTI, load


def main() -> None:
    train, test, orig = load()
    rng = np.random.RandomState(42)
    sub_idx = rng.choice(len(train), size=10000, replace=False)
    n_orig = len(orig)
    n_synth_sub = len(sub_idx)

    print(f"orig {n_orig} | train {len(train)} | test {len(test)} | sub {n_synth_sub}")

    # Reuse the same FE pipeline as av.py, with shared cat-factorize maps.
    combined_train_av = pd.concat([orig, train.iloc[sub_idx]], ignore_index=True)
    X_av, fact_maps = _build_features(combined_train_av)
    y_av = np.concatenate([np.zeros(n_orig), np.ones(n_synth_sub)]).astype("int32")

    # Fit final AV classifier on the full 20k (no holdout).
    clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
        eval_metric="auc", tree_method="hist", n_jobs=8, random_state=42,
    )
    clf.fit(X_av, y_av, verbose=False)
    print("Full-fit AV classifier ready.")

    # Predict on test (270k) and on the FULL train (630k).
    X_test, _ = _build_features(test, fact_maps=fact_maps)
    X_train, _ = _build_features(train, fact_maps=fact_maps)
    p_test = clf.predict_proba(X_test)[:, 1].astype("float32")
    p_train_fullfit = clf.predict_proba(X_train)[:, 1].astype("float32")

    # For the 10k subsample, use the leak-free OOF from av.py.
    av_oof = np.load(ARTI / "av_oof.npy")
    p_synth_oof_subsample = av_oof[n_orig:]  # 10k
    assert len(p_synth_oof_subsample) == n_synth_sub

    # Build the 630k OOF: full-fit predictions overwritten by OOF on sub_idx.
    p_train_oof = p_train_fullfit.copy()
    p_train_oof[sub_idx] = p_synth_oof_subsample.astype("float32")

    print(f"\nP(synth | train) percentiles (full 630k, leak-free):")
    print(f"  {np.percentile(p_train_oof, [1, 25, 50, 75, 99]).round(4)}")
    print(f"P(synth | test)  percentiles (full 270k):")
    print(f"  {np.percentile(p_test, [1, 25, 50, 75, 99]).round(4)}")

    np.save(ARTI / "oof_av_p_synth_train.npy", p_train_oof)
    np.save(ARTI / "test_av_p_synth.npy", p_test)

    # Sanity check vs the diagnostic on synth subsample
    oof_sub = p_train_oof[sub_idx]
    print(f"\nSanity: OOF-on-sub mean = {oof_sub.mean():.4f} (expect ~0.57 from diagnostic)")

    out = {
        "n_orig": int(n_orig),
        "n_synth_sub": int(n_synth_sub),
        "n_train_full": int(len(train)),
        "n_test": int(len(test)),
        "p_train_oof_percentiles_1_25_50_75_99": [float(x) for x in np.percentile(p_train_oof, [1, 25, 50, 75, 99])],
        "p_test_percentiles_1_25_50_75_99": [float(x) for x in np.percentile(p_test, [1, 25, 50, 75, 99])],
        "p_synth_mean_test": float(p_test.mean()),
    }
    (ARTI / "av_full_predict_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {ARTI/'oof_av_p_synth_train.npy'} and test_av_p_synth.npy")


if __name__ == "__main__":
    main()
