"""A1 — 10k-only host-NN-inversion.

Train LGBM purely on the 10k irrigation_prediction.csv (clean labels, host's
NN training source). Apply to synthetic train to identify rule-violator
prediction agreement (does the 10k-trained model agree with synthetic NN
on which rows are flipped?). Apply to test as a potential new override
component.

Key questions:
  1. Does 10k-trained model achieve > rule's 98.4% accuracy on 10k itself?
  2. Does 10k-trained model agree with synthetic train labels at ~98.4%
     (i.e., does it pick up the same NN-flip pattern)?
  3. On synthetic train rule-violators (10,304 rows), how often does the
     10k-trained model predict the true (flipped) class vs the rule class?
  4. On test, where does it disagree with 4b? Is precision good?
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


ART = Path("scripts/artifacts")
SUB = Path("submissions")
LMH = ["L", "M", "H"]
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}
LMH_NAMES = {0: "Low", 1: "Medium", 2: "High"}


def compute_dgp_score(df: pd.DataFrame) -> np.ndarray:
    dry = (df["Soil_Moisture"] < 25).astype(int)
    norain = (df["Rainfall_mm"] < 300).astype(int)
    hot = (df["Temperature_C"] > 30).astype(int)
    windy = (df["Wind_Speed_kmh"] > 10).astype(int)
    nomulch = (df["Mulching_Used"] == "No").astype(int)
    kc = df["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int) * 2
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).to_numpy()


def dgp_rule(score):
    pred = np.full_like(score, 1, dtype=np.int8)
    pred[score <= 3] = 0
    pred[score >= 7] = 2
    return pred


def csv_to_argmax(path: Path) -> np.ndarray:
    s = pd.read_csv(path)["Irrigation_Need"]
    return s.map(LMH_REV).to_numpy(dtype=np.int8)


def main():
    t0 = time.time()
    print("=== A1: 10k-only host-NN-inversion ===\n")

    orig = pd.read_csv("data/irrigation_prediction.csv")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    test_ids = test["id"].to_numpy()

    print(f"Original 10k: {orig.shape}, dist: {orig['Irrigation_Need'].value_counts().to_dict()}")

    feature_cols = [c for c in orig.columns if c not in ["id", "Irrigation_Need"]]
    print(f"feature_cols: {feature_cols}")

    y_orig = orig["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    X_orig = orig[feature_cols].copy()

    y_synth = train["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    X_synth = train[feature_cols].copy()

    X_test = test[feature_cols].copy()

    # Encode categoricals
    cat_cols = X_orig.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X_orig[c] = X_orig[c].astype("category")
        X_synth[c] = X_synth[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    # 5-fold OOF on 10k
    print(f"\n5-fold CV on 10k...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_orig = np.zeros((len(X_orig), 3), dtype=np.float32)
    test_folds = np.zeros((5, len(X_test), 3), dtype=np.float32)
    synth_folds = np.zeros((5, len(X_synth), 3), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(X_orig, y_orig)):
        Xtr, Xva = X_orig.iloc[tr], X_orig.iloc[va]
        ytr, yva = y_orig[tr], y_orig[va]
        model = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03, num_leaves=15,
            min_child_samples=20, subsample=0.85, colsample_bytree=0.85,
            random_state=42 + fold, verbose=-1, n_jobs=-1,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(40, verbose=False)])
        oof_orig[va] = model.predict_proba(Xva)
        test_folds[fold] = model.predict_proba(X_test)
        synth_folds[fold] = model.predict_proba(X_synth)
        print(f"  fold {fold+1}: best_iter={model.best_iteration_}, "
              f"acc={float((oof_orig[va].argmax(1) == yva).mean()):.4f}")

    p_test = test_folds.mean(axis=0)
    p_synth = synth_folds.mean(axis=0)

    # Q1: 10k OOF accuracy vs rule baseline
    am_orig = oof_orig.argmax(1).astype(np.int8)
    score_orig = compute_dgp_score(orig)
    rule_orig = dgp_rule(score_orig)
    print(f"\n=== 10k OOF: ===")
    print(f"  Model accuracy: {float((am_orig == y_orig).mean()):.4f}")
    print(f"  Rule accuracy:  {float((rule_orig == y_orig).mean()):.4f}")
    print(f"  Model agrees with rule: {float((am_orig == rule_orig).mean()):.4f}")

    # Q2: 10k-trained model on synthetic train — agreement with synthetic labels
    am_synth = p_synth.argmax(1).astype(np.int8)
    score_synth = compute_dgp_score(train)
    rule_synth = dgp_rule(score_synth)
    print(f"\n=== 10k-trained model on synthetic train: ===")
    print(f"  Model accuracy on synthetic labels: {float((am_synth == y_synth).mean()):.4f}")
    print(f"  Rule accuracy on synthetic labels:  {float((rule_synth == y_synth).mean()):.4f}")
    print(f"  Model agrees with rule on synthetic: {float((am_synth == rule_synth).mean()):.4f}")

    # Q3: on synthetic train rule-violators, does 10k-model predict TRUE class or RULE class?
    viol_mask = rule_synth != y_synth
    n_viol = int(viol_mask.sum())
    print(f"\n=== On {n_viol} synthetic violators: ===")
    print(f"  10k-model accuracy on violators: {float((am_synth[viol_mask] == y_synth[viol_mask]).mean()):.4f}")
    print(f"  10k-model agrees with rule on violators: {float((am_synth[viol_mask] == rule_synth[viol_mask]).mean()):.4f}")

    # Q4: On test, where does 10k-model disagree with 4b?
    fb = csv_to_argmax(SUB / "submission_idea4b_selective_override.csv")
    am_test = p_test.argmax(1).astype(np.int8)
    conf_test = p_test.max(axis=1)
    disagree_4b = am_test != fb
    n_dis = int(disagree_4b.sum())
    print(f"\n=== Test: 10k-model disagrees with 4b ===")
    print(f"  Total disagreements: {n_dis}")

    # Sweep confidence thresholds
    print(f"\nFlip candidates by confidence threshold:")
    for thresh in [0.50, 0.70, 0.85, 0.90, 0.95, 0.99]:
        flip_mask = disagree_4b & (conf_test >= thresh)
        n = int(flip_mask.sum())
        if n == 0: continue
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to: continue
                d = int(((fb == fr) & (am_test == to) & flip_mask).sum())
                if d > 0:
                    dirs[f"{LMH[fr]}->{LMH[to]}"] = d
        print(f"  conf >= {thresh:.2f}: n={n:>5}  dirs={dirs}")

    # Save
    np.save(ART / "A1_oof_10k.npy", oof_orig)
    np.save(ART / "A1_synth_pred.npy", p_synth)
    np.save(ART / "A1_test_pred.npy", p_test)
    print(f"\nSaved A1 prediction artifacts")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
