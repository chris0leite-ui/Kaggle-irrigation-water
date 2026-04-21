"""XGBoost-dist with rule-routing for rows at dgp_score in {0, 1}.

At train time: drop rows with dgp_score in {0, 1} from both train and
early-stopping eval folds. At predict time: route those rows to the
rule (rule predicts Low for score <= 3), use XGB for everything else.

Hypothesis: scores 1 and 2 are deep in the Low side of the score
distribution, far from the class boundary at score=3. The rule is
near-100% accurate there, so XGB gains no signal from training on
them. Removing them frees XGB to spend all its boosting depth on
the ambiguous rows where the flip signal lives.

Pipeline mirrors benchmark_xgb_dist.py (same 43-feature dist set,
same XGB params, same log-bias tuning).

Artefacts:
  scripts/artifacts/oof_xgb_dist_routed_v4.npy        (full-row OOF)
  scripts/artifacts/test_xgb_dist_routed_v4.npy       (full-row test)
  scripts/artifacts/xgb_dist_routed_v4_results.json
  submissions/submission_xgb_dist_routed_v4_tuned.csv
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
ROUTED_SCORES = (0, 1)
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values

    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)

    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)

    out["dry"] = dry
    out["norain"] = norain
    out["hot"] = hot
    out["windy"] = windy
    out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)

    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)

    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)

    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)

    return out


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building distance features")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    tr_scores = tr["dgp_score"].values
    te_scores = te["dgp_score"].values
    tr_routed_mask = np.isin(tr_scores, ROUTED_SCORES)
    te_routed_mask = np.isin(te_scores, ROUTED_SCORES)
    log(f"train rows routed to rule (score in {ROUTED_SCORES}): "
        f"{tr_routed_mask.sum()} / {len(tr)} ({tr_routed_mask.mean()*100:.2f}%)")
    log(f"test  rows routed to rule (score in {ROUTED_SCORES}): "
        f"{te_routed_mask.sum()} / {len(te)} ({te_routed_mask.mean()*100:.2f}%)")

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features: {len(feat_cols)} ({len(num_cols)} numeric + {len(cat_cols)} categorical)")

    # Rule onehot for routed rows: the rule says "Low" (score <= 3 -> Low).
    # We use soft-clipped onehot so log-bias doesn't receive -inf gradient.
    rule_prob_low = np.array([1.0 - 2e-9, 1e-9, 1e-9], dtype=np.float64)

    log("running 5-fold stratified XGBoost with score-{0,1} routing")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred_xgb = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    xgb_params = dict(
        objective="multi:softprob",
        num_class=len(CLASSES),
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
        seed=SEED,
    )

    dte = xgb.DMatrix(X_test, enable_categorical=True)
    best_iters = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        tr_filtered = tr_idx[~np.isin(tr_scores[tr_idx], ROUTED_SCORES)]
        va_filtered = va_idx[~np.isin(tr_scores[va_idx], ROUTED_SCORES)]
        n_dropped_tr = len(tr_idx) - len(tr_filtered)
        n_dropped_va = len(va_idx) - len(va_filtered)

        dtr = xgb.DMatrix(X.iloc[tr_filtered], label=y[tr_filtered], enable_categorical=True)
        dva = xgb.DMatrix(X.iloc[va_filtered], label=y[va_filtered], enable_categorical=True)
        booster = xgb.train(
            xgb_params, dtr, num_boost_round=4000,
            evals=[(dva, "val")],
            early_stopping_rounds=100,
            verbose_eval=0,
        )
        best_iter = booster.best_iteration
        best_iters.append(best_iter)

        # predict on ALL va rows (not just filtered) so we can route
        dva_full = xgb.DMatrix(X.iloc[va_idx], enable_categorical=True)
        val_pred_all = booster.predict(dva_full, iteration_range=(0, best_iter + 1))

        # route: score in {1,2} -> rule_prob_low; else -> XGB
        va_mask = tr_routed_mask[va_idx]
        oof[va_idx[~va_mask]] = val_pred_all[~va_mask]
        oof[va_idx[va_mask]] = rule_prob_low

        # test: average XGB probs across folds; route at test time after loop
        test_pred_xgb += booster.predict(dte, iteration_range=(0, best_iter + 1)) / N_FOLDS

        fold_bal_all = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        fold_bal_nonrouted = balanced_accuracy_score(
            y[va_idx[~va_mask]], val_pred_all[~va_mask].argmax(axis=1)
        )
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={best_iter}  "
            f"dropped_tr={n_dropped_tr} dropped_va={n_dropped_va}  "
            f"bal_acc(routed argmax, all val)={fold_bal_all:.5f}  "
            f"bal_acc(non-routed only, XGB argmax)={fold_bal_nonrouted:.5f}  "
            f"({time.time()-t0:.1f}s)")

    # assemble routed test predictions
    test_pred = test_pred_xgb.copy()
    test_pred[te_routed_mask] = rule_prob_low

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))

    log("coord-ascent over per-class log-bias (on routed OOF)")
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"  bias = {dict(zip(CLASSES, bias.round(4)))}  oof_bal_acc={tuned_bal:.5f}")

    cm = confusion_matrix(
        y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"OOF confusion matrix (routed):\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # diagnostic: rule accuracy on routed rows, XGB accuracy on non-routed rows
    routed_rows_y = y[tr_routed_mask]
    rule_pred_on_routed = np.zeros(tr_routed_mask.sum(), dtype=np.int32)  # rule -> Low = 0
    rule_acc_on_routed = balanced_accuracy_score(routed_rows_y, rule_pred_on_routed)
    rule_raw_acc = (routed_rows_y == 0).mean()
    log(f"  rule raw acc on routed rows (Low fraction): {rule_raw_acc:.5f}")
    log(f"  rule bal_acc on routed rows: {rule_acc_on_routed:.5f}")

    non_routed_mask = ~tr_routed_mask
    xgb_only_argmax_bal = balanced_accuracy_score(
        y[non_routed_mask], oof[non_routed_mask].argmax(axis=1)
    )
    log(f"  XGB bal_acc on non-routed rows (argmax): {xgb_only_argmax_bal:.5f}")

    print("\n=== XGBoost-dist + score-{0,1} routing (OOF bal_acc) ===")
    print(f"  argmax (routed OOF)          : {argmax_bal:.5f}")
    print(f"  prior-reweight (routed OOF)  : {reweight_bal:.5f}")
    print(f"  tuned log-bias               : {tuned_bal:.5f}")
    print(f"  baseline XGB-dist (no FE ref): 0.97304")
    print(f"  Δ vs baseline                : {tuned_bal - 0.97304:+.5f}")

    np.save(ART_DIR / "oof_xgb_dist_routed_v4.npy", oof)
    np.save(ART_DIR / "test_xgb_dist_routed_v4.npy", test_pred)
    with open(ART_DIR / "xgb_dist_routed_v4_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "routed_scores": list(ROUTED_SCORES),
            "train_rows_routed": int(tr_routed_mask.sum()),
            "test_rows_routed": int(te_routed_mask.sum()),
            "rule_raw_acc_on_routed": float(rule_raw_acc),
            "best_iters_per_fold": [int(x) for x in best_iters],
            "n_features": len(feat_cols),
            "class_priors": prior.tolist(),
            "log_bias": bias.tolist(),
            "argmax_bal_acc": float(argmax_bal),
            "reweight_bal_acc": float(reweight_bal),
            "tuned_bal_acc": float(tuned_bal),
            "xgb_only_argmax_bal_acc_on_nonrouted": float(xgb_only_argmax_bal),
        }, f, indent=2)

    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT_DIR / "submission_xgb_dist_routed_v4_tuned.csv", index=False
    )
    log(f"OOF + test probs saved to {ART_DIR}/; submission to {OUT_DIR}/")


if __name__ == "__main__":
    main()
