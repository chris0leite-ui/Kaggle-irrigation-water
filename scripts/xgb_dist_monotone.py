"""XGBoost-dist with monotone constraints, one-vs-rest formulation.

Rationale: XGB's `multi:softprob` monotone_constraints applies per-class
identically — useless because it would push all 3 class logits the same
direction. One-vs-rest binary heads let us apply per-class signs.

Monotone assumption (from the DGP rule):
  - higher `dgp_score` => more likely High, less likely Low
  - higher `sm_dist`, `rf_dist` (wetter / rainier) => more likely Low
  - higher `tc_dist`, `ws_dist` (hotter / windier) => more likely High
  - binary stress flags (dry, norain, hot, windy, nomulch, kc_active)
    => more likely High, less likely Low
  - Medium is NON-monotonic in score (peaks at 4-6), so Medium head is
    left unconstrained.

Pipeline: same 5-fold StratifiedKFold(seed=42) / same 43-feature dist
set as benchmark_xgb_dist.py, but three binary XGBs per fold with
`objective=binary:logistic` + monotone_constraints. Final prob is the
row-wise normalized [P(Low), P(Medium), P(High)].

Compared to vanilla XGB-dist (0.97304) and LB-best. Fold-1 error
Jaccard reported so we know whether to run routed + spec too.

Artefacts:
  scripts/artifacts/oof_xgb_dist_monotone.npy
  scripts/artifacts/test_xgb_dist_monotone.npy
  scripts/artifacts/xgb_dist_monotone_results.json
  submissions/submission_xgb_dist_monotone_tuned.csv
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


# Per-feature signed constraint table.
#   +1 = monotone increasing => higher feat value pushes P(this class) up
#   -1 = monotone decreasing => higher feat value pushes P(this class) down
#    0 = unconstrained
#
# Low head: wetter / cooler / calmer => more likely Low
LOW_CONSTRAINTS = {
    "Soil_Moisture": +1,    # wetter -> Low
    "Rainfall_mm": +1,      # rainier -> Low
    "Temperature_C": -1,    # hotter -> not Low
    "Wind_Speed_kmh": -1,   # windier -> not Low
    "sm_dist": +1,
    "rf_dist": +1,
    "tc_dist": -1,
    "ws_dist": -1,
    "dry": -1,
    "norain": -1,
    "hot": -1,
    "windy": -1,
    "nomulch": -1,
    "kc_active": -1,
    "dgp_score": -1,
    "rule_pred": -1,
    "score_dist_low_mid": -1,   # positive side means past Low boundary
    "score_dist_mid_high": -1,
}
# High head: drier / hotter / windier => more likely High
HIGH_CONSTRAINTS = {
    "Soil_Moisture": -1,
    "Rainfall_mm": -1,
    "Temperature_C": +1,
    "Wind_Speed_kmh": +1,
    "sm_dist": -1,
    "rf_dist": -1,
    "tc_dist": +1,
    "ws_dist": +1,
    "dry": +1,
    "norain": +1,
    "hot": +1,
    "windy": +1,
    "nomulch": +1,
    "kc_active": +1,
    "dgp_score": +1,
    "rule_pred": +1,
    "score_dist_low_mid": +1,
    "score_dist_mid_high": +1,
}
# Medium head: non-monotonic, leave unconstrained.
MED_CONSTRAINTS: dict[str, int] = {}


def constraint_tuple(feat_cols: list[str], mapping: dict[str, int]) -> str:
    """XGB expects a `(sign, sign, ...)` string aligned with feat_cols."""
    signs = [mapping.get(c, 0) for c in feat_cols]
    return "(" + ",".join(str(s) for s in signs) + ")"


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

    # Sanity-log the monotone vectors
    low_str = constraint_tuple(feat_cols, LOW_CONSTRAINTS)
    high_str = constraint_tuple(feat_cols, HIGH_CONSTRAINTS)
    med_str = constraint_tuple(feat_cols, MED_CONSTRAINTS)
    n_low_constrained = sum(1 for c in feat_cols if LOW_CONSTRAINTS.get(c, 0) != 0)
    n_high_constrained = sum(1 for c in feat_cols if HIGH_CONSTRAINTS.get(c, 0) != 0)
    log(f"  monotone: Low head has {n_low_constrained} constrained features")
    log(f"  monotone: High head has {n_high_constrained} constrained features")
    log(f"  monotone: Medium head fully unconstrained")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    # OOF: raw per-class binary probs (before normalization)
    oof_raw = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_raw = np.zeros((len(te), len(CLASSES)), dtype=np.float64)
    best_iters_by_class = {c: [] for c in CLASSES}

    dte = xgb.DMatrix(X_test, enable_categorical=True)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr_X = X.iloc[tr_idx]
        dva_X = X.iloc[va_idx]
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        for cls_name, cls_idx, ctable in [
            ("Low", 0, low_str),
            ("Medium", 1, med_str),
            ("High", 2, high_str),
        ]:
            y_tr_bin = (y_tr == cls_idx).astype(np.int32)
            y_va_bin = (y_va == cls_idx).astype(np.int32)
            dtr = xgb.DMatrix(dtr_X, label=y_tr_bin, enable_categorical=True)
            dva = xgb.DMatrix(dva_X, label=y_va_bin, enable_categorical=True)
            params = dict(
                objective="binary:logistic",
                eval_metric="logloss",
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=5,
                subsample=0.9,
                colsample_bytree=0.9,
                tree_method="hist",
                enable_categorical=True,
                monotone_constraints=ctable,
                verbosity=0,
                seed=SEED,
            )
            booster = xgb.train(
                params, dtr, num_boost_round=4000,
                evals=[(dva, "val")],
                early_stopping_rounds=100,
                verbose_eval=0,
            )
            best_iter = booster.best_iteration
            best_iters_by_class[cls_name].append(best_iter)
            oof_raw[va_idx, cls_idx] = booster.predict(
                dva, iteration_range=(0, best_iter + 1)
            )
            test_raw[:, cls_idx] += booster.predict(
                dte, iteration_range=(0, best_iter + 1)
            ) / N_FOLDS

        log(f"  fold {fold+1}/{N_FOLDS}  best_iters "
            f"Low={best_iters_by_class['Low'][-1]} "
            f"Med={best_iters_by_class['Medium'][-1]} "
            f"High={best_iters_by_class['High'][-1]}  "
            f"({time.time()-t0:.1f}s)")

    # Row-wise normalize OvR binary probs -> 3-class probs
    oof = oof_raw / np.clip(oof_raw.sum(axis=1, keepdims=True), 1e-9, None)
    test_pred = test_raw / np.clip(test_raw.sum(axis=1, keepdims=True), 1e-9, None)

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    reweight_bal = balanced_accuracy_score(y, (oof / prior).argmax(axis=1))

    log("coord-ascent over per-class log-bias")
    bias, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"  bias = {dict(zip(CLASSES, bias.round(4)))}  oof_bal_acc={tuned_bal:.5f}")

    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log(f"OOF confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # Jaccard vs current LB-best for blend-prospect diagnostic
    oof_greedy = ART_DIR / "oof_greedy_blend.npy"
    oof_lb_best_path = ART_DIR / "oof_hybrid_lgbmxgb_blend.npy"
    jacs = {}
    err_monotone = set(np.where(oof.argmax(axis=1) != y)[0].tolist())
    for label, p in [("greedy_blend", oof_greedy),
                     ("hybrid_lgbmxgb_blend", oof_lb_best_path)]:
        if p.exists():
            ref = np.load(p)
            err_ref = set(np.where(ref.argmax(axis=1) != y)[0].tolist())
            if err_ref:
                jac = len(err_monotone & err_ref) / len(err_monotone | err_ref)
                jacs[label] = {"jaccard": float(jac),
                               "monotone_errs": len(err_monotone),
                               "ref_errs": len(err_ref)}
    if jacs:
        for k, v in jacs.items():
            log(f"  Error Jaccard vs {k}: {v['jaccard']:.4f}  "
                f"(monotone errs={v['monotone_errs']}, ref errs={v['ref_errs']})")

    print("\n=== XGBoost-dist + monotone (OOF bal_acc, OvR) ===")
    print(f"  argmax            : {argmax_bal:.5f}")
    print(f"  prior-reweight    : {reweight_bal:.5f}")
    print(f"  tuned log-bias    : {tuned_bal:.5f}")
    print(f"  ref XGB-dist      : 0.97304")
    print(f"  Δ vs ref          : {tuned_bal - 0.97304:+.5f}")

    np.save(ART_DIR / "oof_xgb_dist_monotone.npy", oof)
    np.save(ART_DIR / "test_xgb_dist_monotone.npy", test_pred)
    with open(ART_DIR / "xgb_dist_monotone_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "feat_cols": feat_cols,
            "n_low_constrained": n_low_constrained,
            "n_high_constrained": n_high_constrained,
            "low_constraint_tuple": low_str,
            "high_constraint_tuple": high_str,
            "medium_constraint_tuple": med_str,
            "best_iters_by_class": best_iters_by_class,
            "class_priors": prior.tolist(),
            "log_bias": bias.tolist(),
            "argmax_bal_acc": float(argmax_bal),
            "reweight_bal_acc": float(reweight_bal),
            "tuned_bal_acc": float(tuned_bal),
            "jaccard_vs_refs": jacs,
        }, f, indent=2)

    tuned_test_idx = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT_DIR / "submission_xgb_dist_monotone_tuned.csv", index=False
    )
    log(f"OOF + test probs saved to {ART_DIR}/; submission to {OUT_DIR}/")


if __name__ == "__main__":
    main()
