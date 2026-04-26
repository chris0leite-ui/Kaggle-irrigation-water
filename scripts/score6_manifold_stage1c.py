"""Stage 1c: regularized-specialist competition on teacher-Medium domain.

v2 (depth=6, 3000 rounds, 40 feats) hits AUC 0.79 here while a plain L2-LR
on the same features hits AUC 0.85. v2 is overfitting. This stage tests
whether stronger regularization or a different inductive bias can push
AUC + override capacity higher than v2.

Candidates:
  (a) L2-LR (C=1.0, balanced-aware)
  (b) L2-LR + teacher_PH × {non-rule features} interactions
  (c) shallow XGB (depth=3, n_round=200, lr=0.05)
  (d) kNN(k=50) on standardized features

For each: 5-fold OOF AUC + top-N precision (n ∈ {5,10,25,50,100}) +
best macro-delta + override count at break-even precision.

The winner is the candidate that maximizes macro-delta at its own
optimal n. If no candidate clears v2's +0.000086, score=6 lever is
information-bounded; lock LB and stop.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import CLS2IDX, add_distance_features, log_blend  # noqa: E402

ART = Path("scripts/artifacts")
OUT = ART / "score6_manifold_stage1c_results.json"

N_H_TOTAL = 21009
N_M_TOTAL = 239074


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def macro_delta(y_true_3cls: np.ndarray, override_local_idx: np.ndarray) -> float:
    c = int((y_true_3cls[override_local_idx] == 2).sum())
    w_m = int((y_true_3cls[override_local_idx] == 1).sum())
    return (c / N_H_TOTAL - w_m / N_M_TOTAL) / 3.0


def evaluate(name: str, oof_score: np.ndarray, y_3cls: np.ndarray,
             y_bin: np.ndarray, ns: list[int]) -> dict:
    auc = float(roc_auc_score(y_bin, oof_score))
    order = np.argsort(-oof_score)
    topn_eval = {}
    for n in ns:
        if n > len(y_bin):
            continue
        idx = order[:n]
        c = int((y_3cls[idx] == 2).sum())
        w = int((y_3cls[idx] == 1).sum())
        d = macro_delta(y_3cls, idx)
        topn_eval[f"n_{n}"] = {"correct": c, "wrong_m": w,
                                "prec": c / n, "macro_delta": d}
    best_kv = max(topn_eval.items(), key=lambda kv: kv[1]["macro_delta"])
    log(f"  {name}: AUC={auc:.4f}  best_n={best_kv[0]}  "
        f"correct={best_kv[1]['correct']}  prec={best_kv[1]['prec']:.3f}  "
        f"macro_Δ={best_kv[1]['macro_delta']:+.6f}")
    return {"auc": auc, "topn": topn_eval, "best_n": best_kv[0],
            "best_macro_delta": best_kv[1]["macro_delta"]}


def cv_oof(model_factory, X: np.ndarray, y_bin: np.ndarray,
           seed: int = 42, n_splits: int = 5,
           preproc: str = "standard") -> np.ndarray:
    """Generic 5-fold OOF predict_proba for a single binary classifier."""
    oof = np.zeros(len(y_bin), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(X, y_bin):
        if preproc == "standard":
            sc = StandardScaler().fit(X[tr_idx])
            X_tr = sc.transform(X[tr_idx])
            X_va = sc.transform(X[va_idx])
        else:
            X_tr, X_va = X[tr_idx], X[va_idx]
        m = model_factory()
        m.fit(X_tr, y_bin[tr_idx])
        oof[va_idx] = m.predict_proba(X_va)[:, 1]
    return oof


def main() -> None:
    log("loading train")
    tr = pd.read_csv("data/train.csv")
    tr = add_distance_features(tr)

    y_full = tr["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int8)
    score = tr["dgp_score"].to_numpy().astype(np.int8)

    log("building LB-best 3-way teacher")
    oof_r = np.load(ART / "oof_recipe_full_te.npy")
    oof_s1 = np.load(ART / "oof_recipe_pseudolabel.npy")
    oof_s7 = np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")
    teacher = log_blend([oof_r, oof_s1, oof_s7], np.array([0.25, 0.35, 0.40]))
    bias = np.array([1.4324, 1.4689, 3.4008])
    teacher_pred = np.argmax(np.log(np.clip(teacher, 1e-9, 1.0)) + bias, axis=1).astype(np.int8)

    domain = (score == 6) & (teacher_pred == 1)
    n_dom = int(domain.sum())
    log(f"domain: {n_dom:,} rows  H={int(((y_full==2)&domain).sum())}")

    y_3 = y_full[domain]
    y_bin = (y_3 == 2).astype(np.int8)

    # Feature set: the 21 raw + 5 teacher-meta + 8 cat-encoded = 34 cols
    raw_feats = [
        "Humidity", "Previous_Irrigation_mm", "Electrical_Conductivity",
        "Soil_pH", "Organic_Carbon", "Sunlight_Hours", "Field_Area_hectare",
        "Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh",
        "sm_dist", "rf_dist", "tc_dist", "ws_dist",
        "sm_abs", "rf_abs", "tc_abs", "ws_abs",
        "sm_x_rf", "tc_x_ws",
    ]
    cat_feats_raw = ["Crop_Type", "Soil_Type", "Region", "Season",
                     "Mulching_Used", "Irrigation_Type", "Water_Source",
                     "Crop_Growth_Stage"]
    df = tr.loc[domain, raw_feats + cat_feats_raw].reset_index(drop=True)
    for c in cat_feats_raw:
        mapping = {v: i for i, v in enumerate(sorted(df[c].dropna().unique()))}
        df[c] = df[c].map(mapping).fillna(-1).astype(np.int32)

    teacher_dom = teacher[domain]
    df["teacher_PL"] = teacher_dom[:, 0].astype(np.float32)
    df["teacher_PM"] = teacher_dom[:, 1].astype(np.float32)
    df["teacher_PH"] = teacher_dom[:, 2].astype(np.float32)
    df["teacher_mh_margin"] = (df["teacher_PM"] - df["teacher_PH"]).astype(np.float32)
    df["teacher_mh_ratio"] = (np.log(np.clip(df["teacher_PH"], 1e-9, 1)) -
                                np.log(np.clip(df["teacher_PM"], 1e-9, 1))).astype(np.float32)
    teacher_feats = ["teacher_PL", "teacher_PM", "teacher_PH",
                     "teacher_mh_margin", "teacher_mh_ratio"]

    base_cols = raw_feats + cat_feats_raw + teacher_feats
    log(f"base feature count: {len(base_cols)}")

    # Add teacher_PH × top-3 raw feature interactions
    interaction_feats = []
    for raw in ["Soil_pH", "Humidity", "Previous_Irrigation_mm"]:
        col = f"PH_x_{raw}"
        df[col] = (df["teacher_PH"] * df[raw]).astype(np.float32)
        interaction_feats.append(col)
    log(f"with PH×raw interactions: +{len(interaction_feats)}")

    ns_eval = [5, 10, 25, 50, 100, 200, 500, 1000]
    results = {}

    # Baseline: v2 already on disk
    log("=== baseline: v2 specialist ===")
    oof_v2 = np.load(ART / "oof_spec6_mh_v2.npy")[domain]
    results["v2"] = evaluate("v2_baseline", oof_v2, y_3, y_bin, ns_eval)

    # Univariate teacher_PH
    log("=== A: univariate teacher_PH ===")
    results["univariate_PH"] = evaluate(
        "teacher_PH", df["teacher_PH"].to_numpy(), y_3, y_bin, ns_eval
    )

    # B: L2-LR on base features
    log("=== B: L2-LR (C=1, base 34 feats) ===")
    X = df[base_cols].to_numpy().astype(np.float32)
    oof = cv_oof(lambda: LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
                 X, y_bin)
    results["lr_base"] = evaluate("lr_base", oof, y_3, y_bin, ns_eval)

    # C: L2-LR + interactions
    log("=== C: L2-LR + PH×raw interactions ===")
    X = df[base_cols + interaction_feats].to_numpy().astype(np.float32)
    oof = cv_oof(lambda: LogisticRegression(C=1.0, max_iter=500, solver="lbfgs"),
                 X, y_bin)
    results["lr_interactions"] = evaluate("lr_interactions", oof, y_3, y_bin, ns_eval)

    # D: L2-LR with class_weight=balanced
    log("=== D: L2-LR balanced ===")
    X = df[base_cols + interaction_feats].to_numpy().astype(np.float32)
    oof = cv_oof(lambda: LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                              class_weight="balanced"),
                 X, y_bin)
    results["lr_balanced"] = evaluate("lr_balanced", oof, y_3, y_bin, ns_eval)

    # E: L2-LR very strong reg
    log("=== E: L2-LR (C=0.01, heavy reg) ===")
    X = df[base_cols + interaction_feats].to_numpy().astype(np.float32)
    oof = cv_oof(lambda: LogisticRegression(C=0.01, max_iter=500, solver="lbfgs"),
                 X, y_bin)
    results["lr_heavy"] = evaluate("lr_heavy", oof, y_3, y_bin, ns_eval)

    # F: shallow XGB
    log("=== F: shallow XGB (depth=3, 200 rounds, lr=0.05) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_xgb = np.zeros(len(y_bin), dtype=np.float32)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(df.index, y_bin)):
        dtr = xgb.DMatrix(df.iloc[tr_idx], label=y_bin[tr_idx])
        dva = xgb.DMatrix(df.iloc[va_idx], label=y_bin[va_idx])
        params = dict(objective="binary:logistic", eval_metric="auc",
                       max_depth=3, learning_rate=0.05, min_child_weight=20,
                       subsample=0.9, colsample_bytree=0.9,
                       reg_alpha=2.0, reg_lambda=2.0,
                       tree_method="hist", verbosity=0, seed=42)
        booster = xgb.train(params, dtr, num_boost_round=200,
                              evals=[(dva, "v")], early_stopping_rounds=30,
                              verbose_eval=0)
        oof_xgb[va_idx] = booster.predict(dva, iteration_range=(0, booster.best_iteration + 1))
    results["xgb_shallow"] = evaluate("xgb_shallow", oof_xgb, y_3, y_bin, ns_eval)

    # G: kNN on standardized features
    log("=== G: kNN(k=50) on standardized base+interactions ===")
    X = df[base_cols + interaction_feats].to_numpy().astype(np.float32)
    oof = cv_oof(lambda: KNeighborsClassifier(n_neighbors=50, n_jobs=-1),
                 X, y_bin)
    results["knn50"] = evaluate("knn50", oof, y_3, y_bin, ns_eval)

    # Summary
    log("\n=== SUMMARY (sorted by best macro_delta) ===")
    log(f"{'cand':18s} {'AUC':>7s} {'best_n':>8s} {'correct':>8s} {'prec':>7s} {'macro_Δ':>11s}")
    sorted_res = sorted(results.items(), key=lambda kv: -kv[1]["best_macro_delta"])
    for name, r in sorted_res:
        bn = r["best_n"]
        c = r["topn"][bn]["correct"]
        p = r["topn"][bn]["prec"]
        log(f"{name:18s} {r['auc']:7.4f} {bn:>8s} {c:>8d} {p:7.3f} {r['best_macro_delta']:+11.6f}")

    with open(OUT, "w") as f:
        json.dump({"domain_n": n_dom, "h_count": int(y_bin.sum()),
                   "results": results}, f, indent=2)
    log(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
