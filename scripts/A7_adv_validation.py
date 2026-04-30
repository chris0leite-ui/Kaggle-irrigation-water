"""A7 — Adversarial validation: train vs test distribution shift sanity check.

5-min experiment. If AUC ≈ 0.5, train and test are i.i.d. and the entire
"distribution shift mechanism" family of ideas (B1 mix-10k, A7 reweighting,
test-similarity sample weighting) is dead. If AUC ≥ 0.55, real shift exists
and these ideas have a Bayesian floor.

Train binary classifier with target = 1[is_test], features = the original
20 columns. 5-fold CV, report mean AUC + per-feature importance.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def main():
    t0 = time.time()
    print("=== A7: Adversarial validation (train vs test) ===\n")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print(f"train={train.shape}, test={test.shape}")

    # Drop id + label
    feature_cols = [c for c in train.columns if c not in ["id", "Irrigation_Need"]]
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    # Build adversarial dataset
    X = pd.concat([X_train, X_test], ignore_index=True)
    y_adv = np.concatenate([np.zeros(len(X_train), dtype=np.int8),
                            np.ones(len(X_test), dtype=np.int8)])
    print(f"adversarial dataset: {X.shape}, n_test={int(y_adv.sum())}")

    # Encode categoricals as category type for lightgbm
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    print(f"categorical cols: {cat_cols}")
    for c in cat_cols:
        X[c] = X[c].astype("category")

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    fold_imps = []
    for fold, (tr, va) in enumerate(skf.split(X, y_adv)):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y_adv[tr], y_adv[va]
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            min_child_samples=200, subsample=0.8, colsample_bytree=0.8,
            random_state=42 + fold, verbose=-1,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(20, verbose=False)])
        p = model.predict_proba(Xva)[:, 1]
        auc = roc_auc_score(yva, p)
        aucs.append(auc)
        fold_imps.append(model.feature_importances_)
        print(f"fold {fold+1}: AUC={auc:.4f}  best_iter={model.best_iteration_}")

    mean_auc = np.mean(aucs)
    print(f"\nMean AUC: {mean_auc:.4f}  (std={np.std(aucs):.4f})")

    # Feature importance
    imp = np.mean(fold_imps, axis=0)
    imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp})
    imp_df = imp_df.sort_values("importance", ascending=False).head(10)
    print(f"\nTop 10 features in train-vs-test classifier:")
    print(imp_df.to_string(index=False))

    print(f"\nElapsed: {time.time()-t0:.1f}s")

    # Verdict
    print(f"\n=== Verdict ===")
    if mean_auc < 0.52:
        print("AUC ~0.5 → train and test are i.i.d. — distribution-shift mechanisms (B1 10k mix, sample reweighting by test-similarity) have NO Bayesian floor.")
    elif mean_auc < 0.60:
        print("AUC 0.52-0.60 → MILD distribution shift. Distribution-shift mechanisms have a small floor; expected EV is moderate.")
    else:
        print("AUC ≥ 0.60 → SIGNIFICANT distribution shift. Distribution-shift mechanisms become high-EV — pursue B1 (10k mix), test-similarity reweighting, etc.")


if __name__ == "__main__":
    main()
