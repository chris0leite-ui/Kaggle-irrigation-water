"""Rule distillation: train a flexible model on 10k original with ALL
features (not just the 6 rule cols) and use its predictions as input
signal to the main hybrid.

Idea: the 10k original is a NOISE-FREE view of the host's NN decision
surface (rule is 100% there). A flexible model (LGBM with shallow trees,
or a calibrated head) trained on the 10k — using all 19 features — can
capture what the NN used BEYOND the 6-feature rule we reverse-engineered.
If the NN weighted Humidity / Previous_Irrigation / EC / Field_Area at
decision time, those weights show up as feature importance on the
10k-trained model.

Output is an OOF-equivalent prediction for the 630k synthetic (trained
on all 10k with no CV since 10k is a separate dataset; synthetic is
out-of-sample by construction). These probs become 3 new feature cols
for LGBM-dist.

Baseline: LGBM-dist OOF tuned 0.97266.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
SUB.mkdir(exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_dist(df: pd.DataFrame) -> pd.DataFrame:
    """Same 43-col dist feature set as benchmark_xgb_dist."""
    from xgb_specialist_678 import add_distance_features
    return add_distance_features(df)


def encode_cats(tr: pd.DataFrame, te: pd.DataFrame,
                orig: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Shared categorical encoding across all three DataFrames."""
    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        vals = sorted(set(tr[c].unique()) | set(te[c].unique())
                      | set(orig[c].unique()))
        mapping = {v: i for i, v in enumerate(vals)}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")
        orig[c] = orig[c].map(mapping).astype("int32")
    return num_cols, cat_cols


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/irrigation_prediction.csv")
    orig[ID] = np.arange(-len(orig), 0)

    log("adding dist features")
    tr = add_dist(tr); te = add_dist(te); orig = add_dist(orig)
    num_cols, cat_cols = encode_cats(tr, te, orig)
    feat_cols = num_cols + cat_cols

    X_tr = tr[feat_cols].copy()
    X_te = te[feat_cols].copy()
    X_orig = orig[feat_cols].copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_te[c] = X_te[c].astype("category")
        X_orig[c] = X_orig[c].astype("category")

    y_tr = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    y_orig = orig[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y_tr) / len(y_tr)

    # --------------------- train distillation model on 10k original (CV) --
    # 5-fold CV on original to get honest OOF-equivalent feature importance.
    # For synthetic train/test, fit on ALL 10k (they're out-of-sample).
    log("training distillation model on 10k original (5-fold CV)")
    lgb_params = dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        learning_rate=0.03, num_leaves=31, min_data_in_leaf=20,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        lambda_l2=0.1, verbose=-1, seed=SEED,
    )
    skf_orig = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_orig = np.zeros((len(orig), 3), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(skf_orig.split(X_orig, y_orig)):
        dtr = lgb.Dataset(X_orig.iloc[tr_idx], label=y_orig[tr_idx],
                          categorical_feature=cat_cols)
        dva = lgb.Dataset(X_orig.iloc[va_idx], label=y_orig[va_idx],
                          categorical_feature=cat_cols, reference=dtr)
        m = lgb.train(lgb_params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(0)])
        oof_orig[va_idx] = m.predict(X_orig.iloc[va_idx],
                                     num_iteration=m.best_iteration)

    orig_bal = balanced_accuracy_score(y_orig, oof_orig.argmax(axis=1))
    log(f"distillation OOF on 10k original: argmax={orig_bal:.5f}")

    # fit on all 10k for downstream predictions
    log("fitting on all 10k original for synthetic predictions")
    dtr_all = lgb.Dataset(X_orig, label=y_orig, categorical_feature=cat_cols)
    m_all = lgb.train(lgb_params, dtr_all, num_boost_round=2000,
                      callbacks=[lgb.log_evaluation(0)])
    tr_dist_probs = m_all.predict(X_tr)   # (630k, 3)
    te_dist_probs = m_all.predict(X_te)   # (270k, 3)

    tr_dist_bal = balanced_accuracy_score(y_tr, tr_dist_probs.argmax(axis=1))
    log(f"distillation model argmax bal_acc on 630k synthetic: {tr_dist_bal:.5f}")

    # feature importance
    imp = pd.DataFrame({
        "feature": feat_cols,
        "gain": m_all.feature_importance(importance_type="gain"),
        "split": m_all.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    log("distillation feature importance (top 10):")
    print(imp.head(10).to_string(index=False))

    # -- attach distillation probs to LGBM-dist inputs and retrain on synth -
    log("training LGBM-dist + distillation-prob features (5-fold on 630k)")
    X_tr["dist_P_Low"] = tr_dist_probs[:, 0].astype(np.float32)
    X_tr["dist_P_Med"] = tr_dist_probs[:, 1].astype(np.float32)
    X_tr["dist_P_High"] = tr_dist_probs[:, 2].astype(np.float32)
    X_te["dist_P_Low"] = te_dist_probs[:, 0].astype(np.float32)
    X_te["dist_P_Med"] = te_dist_probs[:, 1].astype(np.float32)
    X_te["dist_P_High"] = te_dist_probs[:, 2].astype(np.float32)

    main_params = dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
        verbose=-1, seed=SEED,
    )
    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test_probs = np.zeros((len(te), 3), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, y_tr)):
        t0 = time.time()
        dtr = lgb.Dataset(X_tr.iloc[tr_idx], label=y_tr[tr_idx],
                          categorical_feature=cat_cols)
        dva = lgb.Dataset(X_tr.iloc[va_idx], label=y_tr[va_idx],
                          categorical_feature=cat_cols, reference=dtr)
        m = lgb.train(main_params, dtr, num_boost_round=4000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False),
                                 lgb.log_evaluation(0)])
        best_iters.append(m.best_iteration)
        oof[va_idx] = m.predict(X_tr.iloc[va_idx], num_iteration=m.best_iteration)
        test_probs += m.predict(X_te, num_iteration=m.best_iteration) / N_FOLDS
        bal = balanced_accuracy_score(y_tr[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={m.best_iteration}  "
            f"argmax_bal={bal:.5f}  ({time.time()-t0:.1f}s)")

    def tune(p, y, prior):
        lp = np.log(np.clip(p, 1e-9, 1.0))
        b = -np.log(prior)
        best = balanced_accuracy_score(y, (lp + b).argmax(axis=1))
        grid = np.linspace(-3, 3, 61)
        for _ in range(25):
            imp = False
            for k in range(3):
                base = b.copy()
                sc = []
                for g in grid:
                    base[k] = b[k] + g
                    sc.append(balanced_accuracy_score(y, (lp + base).argmax(axis=1)))
                j = int(np.argmax(sc))
                if sc[j] > best + 1e-6:
                    b[k] = b[k] + grid[j]
                    best = sc[j]
                    imp = True
            if not imp:
                break
        return b, best

    argmax_bal = balanced_accuracy_score(y_tr, oof.argmax(axis=1))
    bias, tuned = tune(oof, y_tr, prior)
    cm = confusion_matrix(y_tr, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))

    print(f"\n=== LGBM-dist + rule-distillation probs (OOF bal_acc) ===")
    print(f"  argmax             : {argmax_bal:.5f}")
    print(f"  tuned log-bias     : {tuned:.5f}")
    print(f"  baseline LGBM-dist : 0.97266")
    print(f"  Δ                  : {tuned - 0.97266:+.5f}")
    print(f"  bias               : {bias.round(3).tolist()}")
    print(f"  OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_lgbm_rule_distill.npy", oof)
    np.save(ART / "test_lgbm_rule_distill.npy", test_probs)
    with open(ART / "rule_distillation_results.json", "w") as f:
        json.dump({
            "orig_oof_bal": float(orig_bal),
            "dist_on_synth_bal": float(tr_dist_bal),
            "argmax_bal": float(argmax_bal),
            "tuned_bal": float(tuned),
            "delta_vs_lgbm_dist": float(tuned - 0.97266),
            "log_bias": bias.tolist(),
            "best_iters": [int(x) for x in best_iters],
            "top_features_dist": imp.head(15).to_dict(orient="records"),
        }, f, indent=2)

    # submission if lift
    if tuned > 0.97266:
        tuned_idx = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(axis=1)
        pd.DataFrame({ID: te[ID],
                      TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
            SUB / "submission_lgbm_rule_distill_tuned.csv", index=False)
        log("submission written (OOF > baseline)")


if __name__ == "__main__":
    main()
