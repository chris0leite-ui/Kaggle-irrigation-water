"""LGBM-dist + leak-free target encoding from the 10k original dataset.

The 10k original has 100% rule accuracy, so target-encoding categorical
features against ITS labels gives a rule-deterministic view of each
category's class distribution. Critically, this encoding never touches
the synthetic training labels — it's leak-free by construction, no
inner CV needed on the encoder itself. The synthetic folds only see
lookup values from a separate, cleaner dataset.

Encodings added:
  Single-cat:  for each of 8 categoricals, 3 cols = P(y=Low,Med,High | cat)
  Pairwise:    for 6 interaction pairs, 3 cols each

Laplace smoothing (alpha=1 for single, alpha=5 for pairs); unseen combos
fall back to the global original prior.

Baseline comparison: LGBM-dist OOF tuned 0.97266.
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

MAIN_CATS = [
    "Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
    "Irrigation_Type", "Water_Source", "Mulching_Used", "Region",
]
# Interactions chosen by domain priors (agronomy + climate):
PAIR_CATS = [
    ("Soil_Type", "Crop_Type"),
    ("Crop_Type", "Crop_Growth_Stage"),
    ("Season", "Region"),
    ("Soil_Type", "Season"),
    ("Crop_Type", "Season"),
    ("Crop_Type", "Irrigation_Type"),
]

ART = Path("scripts/artifacts")
SUB = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
SUB.mkdir(exist_ok=True)


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same dist features as benchmark_xgb_dist.py (43-col set)."""
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


def compute_te_from_source(
    df_src: pd.DataFrame, y_src: np.ndarray, key_cols: list[str],
    alpha: float,
) -> dict:
    """Return dict: key_tuple -> per-class prob array (3,)."""
    k = len(CLASSES)
    prior = np.bincount(y_src, minlength=k) / len(y_src)
    smooth = alpha * prior  # Laplace: each class gets alpha*prior prior count

    # Group by key, accumulate counts
    key_vals = pd.MultiIndex.from_arrays([df_src[c].values for c in key_cols])
    counts = pd.DataFrame({"y": y_src, "k": key_vals}).groupby("k")["y"]
    out = {}
    for key, idx in df_src.groupby(key_cols, observed=True).groups.items():
        if not isinstance(key, tuple):
            key = (key,)
        y_here = y_src[idx.values if hasattr(idx, "values") else np.array(idx)]
        c = np.bincount(y_here, minlength=k).astype(np.float64)
        c += smooth
        p = c / c.sum()
        out[key] = p
    return out, prior


def apply_te(
    df: pd.DataFrame, key_cols: list[str],
    te_lookup: dict, fallback_prior: np.ndarray,
    col_prefix: str,
) -> np.ndarray:
    """Return (n_rows, 3) array of per-class TE values."""
    vals = list(zip(*[df[c].values for c in key_cols]))
    out = np.tile(fallback_prior[None, :], (len(df), 1))
    for i, key in enumerate(vals):
        if key in te_lookup:
            out[i] = te_lookup[key]
    return out


def main() -> None:
    t_all = time.time()
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/original/irrigation_prediction.csv")
    log(f"train: {len(tr)}   test: {len(te)}   original: {len(orig)}")

    log("building dist features (tr, te)")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    y_orig = orig[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # ------------------------------------------------------- TE: single cat --
    log("computing single-cat TE from original")
    te_cols = []
    te_feat_names = []
    for c in MAIN_CATS:
        lookup, cat_prior = compute_te_from_source(orig, y_orig, [c], alpha=1.0)
        vals_tr = apply_te(tr, [c], lookup, cat_prior, f"te_{c}")
        vals_te = apply_te(te, [c], lookup, cat_prior, f"te_{c}")
        vals_or = apply_te(orig, [c], lookup, cat_prior, f"te_{c}")  # not used
        for k, cls in enumerate(CLASSES):
            name = f"teO_{c}__{cls}"
            tr[name] = vals_tr[:, k].astype(np.float32)
            te[name] = vals_te[:, k].astype(np.float32)
            te_feat_names.append(name)
        log(f"  TE[{c:22s}]  keys={len(lookup)}  fallback_prior={cat_prior.round(3).tolist()}")

    # -------------------------------------------------------- TE: pairwise ---
    log("computing pairwise-cat TE from original")
    for (a, b) in PAIR_CATS:
        lookup, pair_prior = compute_te_from_source(orig, y_orig, [a, b], alpha=5.0)
        vals_tr = apply_te(tr, [a, b], lookup, pair_prior, f"te_{a}_{b}")
        vals_te = apply_te(te, [a, b], lookup, pair_prior, f"te_{a}_{b}")
        for k, cls in enumerate(CLASSES):
            name = f"teO_{a}_{b}__{cls}"
            tr[name] = vals_tr[:, k].astype(np.float32)
            te[name] = vals_te[:, k].astype(np.float32)
            te_feat_names.append(name)
        log(f"  TE[{a}×{b:20s}]  keys={len(lookup)}")

    log(f"total TE cols added: {len(te_feat_names)}")

    # ---------------------------------------------- label-encode the raw cats --
    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    log(f"feature count: {len(feat_cols)} ({len(num_cols)} num + {len(cat_cols)} cat)")

    X = tr[feat_cols].copy()
    X_test = te[feat_cols].copy()
    for c in cat_cols:
        X[c] = X[c].astype("category")
        X_test[c] = X_test[c].astype("category")

    # ----------------------------------------------------------- LGBM fit --
    log("running 5-fold stratified LGBM-dist + TE")
    oof = np.zeros((len(tr), 3), dtype=np.float64)
    test = np.zeros((len(te), 3), dtype=np.float64)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    lgb_params = dict(
        objective="multiclass", num_class=3, metric="multi_logloss",
        learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
        verbose=-1, seed=SEED,
    )
    best_iters = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        dtr = lgb.Dataset(X.iloc[tr_idx], label=y[tr_idx],
                          categorical_feature=cat_cols)
        dva = lgb.Dataset(X.iloc[va_idx], label=y[va_idx],
                          categorical_feature=cat_cols, reference=dtr)
        model = lgb.train(
            lgb_params, dtr, num_boost_round=4000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)],
        )
        best_iters.append(model.best_iteration)
        oof[va_idx] = model.predict(X.iloc[va_idx], num_iteration=model.best_iteration)
        test += model.predict(X_test, num_iteration=model.best_iteration) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  best_iter={model.best_iteration}  "
            f"argmax_bal={bal:.5f}  ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------ log-bias tune --
    def tune_log_bias(p, y, prior):
        log_p = np.log(np.clip(p, 1e-9, 1.0))
        bias = -np.log(prior)
        best = balanced_accuracy_score(y, (log_p + bias).argmax(axis=1))
        grid = np.linspace(-3, 3, 61)
        for _ in range(25):
            improved = False
            for k in range(3):
                base = bias.copy()
                sc = []
                for g in grid:
                    base[k] = bias[k] + g
                    sc.append(balanced_accuracy_score(y, (log_p + base).argmax(axis=1)))
                j = int(np.argmax(sc))
                if sc[j] > best + 1e-6:
                    bias[k] = bias[k] + grid[j]
                    best = sc[j]
                    improved = True
            if not improved:
                break
        return bias, best

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias, tuned = tune_log_bias(oof, y, prior)
    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))

    print(f"\n=== LGBM-dist + TE-from-original (OOF bal_acc) ===")
    print(f"  argmax               : {argmax_bal:.5f}")
    print(f"  tuned log-bias       : {tuned:.5f}")
    print(f"  baseline LGBM-dist   : 0.97266")
    print(f"  Δ                    : {tuned - 0.97266:+.5f}")
    print(f"  bias                 : {bias.round(3).tolist()}")
    print(f"  OOF confusion matrix:\n"
          f"{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_lgbm_te_orig.npy", oof)
    np.save(ART / "test_lgbm_te_orig.npy", test)
    with open(ART / "benchmark_te_orig_results.json", "w") as f:
        json.dump({
            "seed": SEED, "n_folds": N_FOLDS,
            "n_features": len(feat_cols),
            "n_te_features": len(te_feat_names),
            "best_iters": [int(x) for x in best_iters],
            "argmax_bal": float(argmax_bal),
            "tuned_bal": float(tuned),
            "delta_vs_lgbm_dist": float(tuned - 0.97266),
            "log_bias": bias.tolist(),
        }, f, indent=2)

    tuned_idx = (np.log(np.clip(test, 1e-9, 1.0)) + bias).argmax(axis=1)
    pd.DataFrame({ID: pd.read_csv("data/test.csv")[ID],
                  TARGET: [IDX2CLS[i] for i in tuned_idx]}).to_csv(
        SUB / "submission_lgbm_te_orig_tuned.csv", index=False)
    log(f"done in {time.time()-t_all:.1f}s; artifacts + submission written")


if __name__ == "__main__":
    main()
