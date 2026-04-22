"""Per-rule-class XGB specialists: Low (scores 0-3), Medium (4-6), High (7-9).

Each specialist trains on only the rows where the rule predicts a given
class, so each model's gradient focuses entirely on discriminating
rule-correct vs rule-flipped within its class domain.

Domain sizes (train):
  Low    (scores 0-3)   373,601 rows  -- class dist  Low 368k / Med 5k / High 0
  Medium (scores 4-6)   235,456 rows  -- class dist  Low 1.6k / Med 232k / High 1.8k
  High   (scores 7-9)    20,943 rows  -- class dist  Low 0    / Med 1.7k / High 19.3k

Per-specialist rule-error rates (what each specialist must recover):
  Low     1.4% of its 373k rows are actually Medium
  Medium  1.4% of its 235k rows are Low or High
  High   14.7% of its 21k rows  are actually Medium

Same 5-fold stratified-on-y split as main XGB, so the per-specialist
OOFs align row-wise and can be stacked into a single OOF matrix via
per-row routing by dgp_score:
    score in {0,1,2,3} -> Low-spec probs
    score in {4,5,6}   -> Medium-spec probs
    score in {7,8,9}   -> High-spec probs

Saves:
    scripts/artifacts/oof_xgb_per_class_spec.npy   (630k × 3; fused)
    scripts/artifacts/test_xgb_per_class_spec.npy  (270k × 3; fused)
    scripts/artifacts/oof_xgb_spec_low.npy
    scripts/artifacts/oof_xgb_spec_med.npy
    scripts/artifacts/oof_xgb_spec_high.npy
    (same for _test_, _spec_ artifacts)
    scripts/artifacts/xgb_per_class_spec_results.json
    submissions/submission_xgb_per_class_spec_tuned.csv
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

SPECIALISTS = {
    "low":  {"scores": (0, 1, 2, 3), "tag": "Low"},
    "med":  {"scores": (4, 5, 6),    "tag": "Medium"},
    "high": {"scores": (7, 8, 9),    "tag": "High"},
}

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
    log(f"features: {len(feat_cols)}  prior: {dict(zip(CLASSES, prior.round(4)))}")

    xgb_params = dict(
        objective="multi:softprob",
        num_class=3,
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

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_splits = list(skf.split(X, y))

    # One fused OOF/test via per-row routing across specialists
    oof_fused = np.zeros((len(tr), 3), dtype=np.float64)
    test_fused = np.zeros((len(te), 3), dtype=np.float64)

    per_spec = {}

    for name, meta in SPECIALISTS.items():
        sc = meta["scores"]
        tag = meta["tag"]
        log(f"=== Specialist '{name}' (scores {sc}, rule predicts {tag}) ===")
        tr_mask = np.isin(tr_scores, sc)
        te_mask = np.isin(te_scores, sc)
        n_tr = tr_mask.sum()
        n_te = te_mask.sum()
        dist = np.bincount(y[tr_mask], minlength=3)
        log(f"  n_train={n_tr}  n_test={n_te}  class dist: "
            f"{dict(zip(CLASSES, dist.tolist()))}")

        oof_spec = np.zeros((len(tr), 3), dtype=np.float64)
        test_spec = np.zeros((len(te), 3), dtype=np.float64)

        # pre-build the test DMatrix restricted to this specialist's domain
        te_dom_idx = np.where(te_mask)[0]
        dte = xgb.DMatrix(X_test.iloc[te_dom_idx], enable_categorical=True)
        best_iters = []

        for fold, (tr_idx, va_idx) in enumerate(fold_splits):
            t0 = time.time()
            tr_dom = tr_idx[np.isin(tr_scores[tr_idx], sc)]
            va_dom = va_idx[np.isin(tr_scores[va_idx], sc)]

            if len(tr_dom) == 0 or len(va_dom) == 0:
                log(f"  fold {fold+1}  empty; skipping")
                continue

            dtr = xgb.DMatrix(X.iloc[tr_dom], label=y[tr_dom], enable_categorical=True)
            dva = xgb.DMatrix(X.iloc[va_dom], label=y[va_dom], enable_categorical=True)
            booster = xgb.train(
                xgb_params, dtr, num_boost_round=4000,
                evals=[(dva, "val")],
                early_stopping_rounds=100,
                verbose_eval=0,
            )
            best_iter = booster.best_iteration
            best_iters.append(best_iter)

            val_pred = booster.predict(dva, iteration_range=(0, best_iter + 1))
            oof_spec[va_dom] = val_pred

            test_pred = booster.predict(dte, iteration_range=(0, best_iter + 1))
            for i, pos in enumerate(te_dom_idx):
                test_spec[pos] += test_pred[i] / N_FOLDS

            fold_bal = balanced_accuracy_score(y[va_dom], val_pred.argmax(axis=1))
            raw_acc = (val_pred.argmax(axis=1) == y[va_dom]).mean()
            log(f"  fold {fold+1}/{N_FOLDS}  n_tr={len(tr_dom)}  n_va={len(va_dom)}  "
                f"best_iter={best_iter}  "
                f"bal_acc={fold_bal:.5f}  raw_acc={raw_acc:.5f}  "
                f"({time.time()-t0:.1f}s)")

        # per-specialist summary (on its domain only)
        dom_y = y[tr_mask]
        dom_oof = oof_spec[tr_mask]
        argmax_bal = balanced_accuracy_score(dom_y, dom_oof.argmax(axis=1))
        raw_acc = (dom_oof.argmax(axis=1) == dom_y).mean()
        rule_target = CLS2IDX[tag]
        rule_pred = np.full(len(dom_y), rule_target, dtype=np.int32)
        rule_raw = (rule_pred == dom_y).mean()
        rule_bal = balanced_accuracy_score(dom_y, rule_pred)
        cm = confusion_matrix(dom_y, dom_oof.argmax(axis=1), labels=[0, 1, 2])
        log(f"  domain-only: rule_raw={rule_raw:.5f} rule_bal={rule_bal:.5f}  "
            f"spec_raw={raw_acc:.5f}  spec_bal={argmax_bal:.5f}  "
            f"(Δ vs rule: {argmax_bal - rule_bal:+.5f})")
        print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

        # stash and fuse
        np.save(ART_DIR / f"oof_xgb_spec_{name}.npy", oof_spec)
        np.save(ART_DIR / f"test_xgb_spec_{name}.npy", test_spec)
        oof_fused[tr_mask] = oof_spec[tr_mask]
        test_fused[te_mask] = test_spec[te_mask]

        per_spec[name] = {
            "scores": list(sc),
            "rule_tag": tag,
            "n_train": int(n_tr),
            "n_test": int(n_te),
            "class_dist": dist.tolist(),
            "best_iters": [int(x) for x in best_iters],
            "rule_raw_acc": float(rule_raw),
            "rule_bal_acc": float(rule_bal),
            "spec_raw_acc": float(raw_acc),
            "spec_bal_acc": float(argmax_bal),
        }

    # evaluate fused OOF globally
    argmax_fused = balanced_accuracy_score(y, oof_fused.argmax(axis=1))
    reweight_fused = balanced_accuracy_score(y, (oof_fused / prior).argmax(axis=1))
    bias, tuned_fused = tune_log_bias(oof_fused, y, prior)
    log(f"\nFused OOF (per-row routed to the matching specialist):")
    log(f"  argmax        : {argmax_fused:.5f}")
    log(f"  prior-reweight: {reweight_fused:.5f}")
    log(f"  tuned log-bias: {tuned_fused:.5f}")
    log(f"  bias = {dict(zip(CLASSES, bias.round(4)))}")

    cm = confusion_matrix(
        y, (np.log(np.clip(oof_fused, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"Fused OOF confusion matrix:\n"
        f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    print("\n=== per-class specialists (fused OOF bal_acc) ===")
    print(f"  argmax         : {argmax_fused:.5f}")
    print(f"  prior-reweight : {reweight_fused:.5f}")
    print(f"  tuned          : {tuned_fused:.5f}")
    print(f"  baseline XGB   : 0.97304")
    print(f"  routed-{{1,2}}   : 0.97333")
    print(f"  hybrid routed+spec-678 : 0.97352 (previous best)")
    print(f"  Δ vs hybrid              : {tuned_fused - 0.97352:+.5f}")

    np.save(ART_DIR / "oof_xgb_per_class_spec.npy", oof_fused)
    np.save(ART_DIR / "test_xgb_per_class_spec.npy", test_fused)
    with open(ART_DIR / "xgb_per_class_spec_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "specialists": per_spec,
            "fused_argmax_bal_acc": float(argmax_fused),
            "fused_reweight_bal_acc": float(reweight_fused),
            "fused_tuned_bal_acc": float(tuned_fused),
            "log_bias": bias.tolist(),
        }, f, indent=2)

    tuned_test_idx = (np.log(np.clip(test_fused, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT_DIR / "submission_xgb_per_class_spec_tuned.csv", index=False
    )
    log(f"fused OOF + test probs saved to {ART_DIR}/; submission to {OUT_DIR}/")


if __name__ == "__main__":
    main()
