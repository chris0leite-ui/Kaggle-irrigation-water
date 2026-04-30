"""B3 — Flip-specialist: train 3-class LGBM SOLELY on rule-violator rows.

Rule-violators = train rows where DGP rule prediction != ground truth label.
~1.6% of 630k = ~10k rows. These are the host NN's deliberate flips. Train
a 3-class classifier on this subset to learn the host NN's flip behavior
directly.

Validation: 5-fold CV WITHIN the rule-violator subset, report per-class
recall and overall macro-recall. If macro-recall on held-out flips is well
above random (0.333), the model is learning structure.

Distinct from W13 (which was binary "is row wrong"): this is a 3-class
output telling us WHICH class to flip to, not just whether to flip.
Distinct from "score=6 specialist" (which was per-score-cell, not per-rule-
violation).

Inference at test time: for each test row, compute rule prediction; ask
specialist what label to use; if specialist's class has confidence ≥ τ,
use specialist's class, else use anchor (4b). τ tuned on TRAIN-OOF
specialist held-out precision per direction.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score


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


def main():
    t0 = time.time()
    print("=== B3: Flip-specialist on rule-violator rows ===\n")

    train = pd.read_csv("data/train.csv")
    y_full = train["Irrigation_Need"].map(LMH_REV).to_numpy(dtype=np.int8)
    score_full = compute_dgp_score(train)
    rule_full = dgp_rule(score_full)

    n_full = len(train)
    rule_acc = float((rule_full == y_full).mean())
    n_violators = int((rule_full != y_full).sum())
    print(f"TRAIN total: {n_full}, rule accuracy: {rule_acc:.5f}")
    print(f"Rule violators (rule != label): {n_violators} ({n_violators/n_full*100:.2f}%)")

    # Rule-violator subset
    viol_mask = rule_full != y_full
    train_v = train[viol_mask].copy()
    y_v = y_full[viol_mask]
    score_v = score_full[viol_mask]
    rule_v = rule_full[viol_mask]

    # Distribution of (rule_pred, true_label) directions
    print(f"\nRule-violator direction breakdown:")
    print(f"{'direction (rule->true)':<20} {'n':>6}")
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            n = int(((rule_v == fr) & (y_v == to)).sum())
            if n > 0:
                print(f"  {LMH[fr]}->{LMH[to]:<4}              {n:>6}")

    # Score distribution
    print(f"\nScore distribution within rule-violators:")
    for sc in sorted(set(score_v)):
        n = int((score_v == sc).sum())
        print(f"  score={sc}: n={n}")

    # Build training dataset (only rule-violators, target = true label)
    feature_cols = [c for c in train.columns if c not in ["id", "Irrigation_Need"]]
    X_v = train_v[feature_cols].copy()
    cat_cols = X_v.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        X_v[c] = X_v[c].astype("category")

    # 5-fold CV stratified by true label
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_pred = np.zeros((len(X_v), 3), dtype=np.float32)
    oof_argmax = np.full(len(X_v), -1, dtype=np.int8)

    for fold, (tr, va) in enumerate(skf.split(X_v, y_v)):
        Xtr, Xva = X_v.iloc[tr], X_v.iloc[va]
        ytr, yva = y_v[tr], y_v[va]
        model = lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            min_child_samples=15, subsample=0.85, colsample_bytree=0.85,
            random_state=42 + fold, verbose=-1, n_jobs=-1,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        p = model.predict_proba(Xva)
        oof_pred[va] = p
        oof_argmax[va] = p.argmax(1)
        print(f"fold {fold+1}: best_iter={model.best_iteration_}, "
              f"val_acc={float((p.argmax(1) == yva).mean()):.4f}")

    # Overall metrics
    acc = float((oof_argmax == y_v).mean())
    macro_recall = recall_score(y_v, oof_argmax, average="macro")
    per_class_recall = recall_score(y_v, oof_argmax, average=None)
    print(f"\nOOF overall accuracy on rule-violators: {acc:.4f}  (random=0.333)")
    print(f"OOF macro-recall: {macro_recall:.4f}")
    print(f"Per-class recall: L={per_class_recall[0]:.3f} M={per_class_recall[1]:.3f} H={per_class_recall[2]:.3f}")

    # Per (rule_pred, true) direction precision
    print(f"\n=== Per-direction (rule_pred -> true) ===")
    print(f"{'direction':<10} {'n':>6} {'OOF P(specialist correct)':>25}")
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            mask = (rule_v == fr) & (y_v == to)
            n = int(mask.sum())
            if n == 0: continue
            p_correct = float((oof_argmax[mask] == y_v[mask]).mean())
            print(f"{LMH[fr]+'->'+LMH[to]:<10} {n:>6}  {p_correct:>25.3f}")

    # Confidence calibration: where specialist is HIGHLY CONFIDENT, what is precision?
    print(f"\n=== Specialist confidence vs correctness ===")
    max_p = oof_pred.max(axis=1)
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
        mask = max_p >= thresh
        n = int(mask.sum())
        if n == 0: continue
        acc_t = float((oof_argmax[mask] == y_v[mask]).mean())
        print(f"  conf >= {thresh:.2f}: n={n:>5}  acc={acc_t:.4f}")

    # Save artifacts
    np.save("scripts/artifacts/B3_oof_specialist.npy", oof_pred)
    print(f"\nElapsed: {time.time()-t0:.1f}s")
    print(f"Saved OOF: scripts/artifacts/B3_oof_specialist.npy")


if __name__ == "__main__":
    main()
