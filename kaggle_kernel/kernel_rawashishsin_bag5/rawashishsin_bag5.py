#!/usr/bin/env python3
"""
Multi-seed bag of rawashishsin v3 (N1 from CLAUDE.md 2026-04-29).
================================================================
Same pipeline as rawashishsin v3 (LB 0.98109) but the TargetEncoder
random_state is varied across {7, 123, 2024, 9999}. SEED=42 is
already on disk as oof_rawashishsin_2600.npy.

Why: sklearn's TargetEncoder(multiclass, cv=5, smooth='auto') is
seed-sensitive (CLAUDE.md 2026-04-28: 93bp CV drift). Geomean-bagging
5 TE seeds reduces variance at the dominant input slot of v1's
natural-cal bank without perturbing v1's load-bearing 7-component
composition.

Outputs PER TE_SEED:
  oof_rawashishsin_te{TE_SEED}.npy
  test_rawashishsin_te{TE_SEED}.npy
  rawashishsin_te{TE_SEED}_results.json
  per-seed-per-fold checkpoints

Wall budget: 4 seeds * ~36min = ~2.4h on Kaggle GPU. Inside 9h cap.
SMOKE: 1 seed x 2-fold x 20k subsample x 300 rounds (~5 min).
"""
from __future__ import annotations
import os, sys, gc, json, time
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Top-level flags for Kaggle (override via sed before push)
SMOKE_OVERRIDE = False
PROBE_OVERRIDE = False
IS_SMOKE = SMOKE_OVERRIDE or os.environ.get("SMOKE", "0") == "1"
IS_PROBE = PROBE_OVERRIDE or os.environ.get("PROBE", "0") == "1"
SEED = 42  # fold split + XGB seed (deterministic)

# Multi-seed bag — TargetEncoder.random_state varies across these
# (4 NEW seeds; SEED=42 is already on disk as oof_rawashishsin_2600.npy)
TE_SEEDS_DEFAULT = "7,123,2024,9999"
TE_SEEDS = [int(s) for s in os.environ.get("TE_SEEDS", TE_SEEDS_DEFAULT).split(",")]
if IS_SMOKE:
    TE_SEEDS = TE_SEEDS[:1]  # smoke only first seed

# Outputs
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("scripts/artifacts")
OUT.mkdir(parents=True, exist_ok=True)
MODE = "smoke" if IS_SMOKE else ("probe" if IS_PROBE else "prod")

print(f"[boot] SMOKE={IS_SMOKE} PROBE={IS_PROBE}", flush=True)
print(f"[boot] TE_SEEDS={TE_SEEDS}", flush=True)
print(f"[boot] Out: {OUT}", flush=True)


def find_data():
    candidates = {
        "train": [
            "/kaggle/input/playground-series-s6e4/train.csv",
            "/kaggle/input/competitions/playground-series-s6e4/train.csv",
            "data/train.csv",
        ],
        "test": [
            "/kaggle/input/playground-series-s6e4/test.csv",
            "/kaggle/input/competitions/playground-series-s6e4/test.csv",
            "data/test.csv",
        ],
        "orig": [
            "/kaggle/input/datasets/miadul/irrigation-water-requirement-prediction-dataset/irrigation_prediction.csv",
            "/kaggle/input/irrigation-water-requirement-prediction-dataset/irrigation_prediction.csv",
            "/kaggle/input/irrigation-prediction-original/irrigation_prediction.csv",
            "data/archive_orig.csv",
            "/tmp/orig_data/irrigation_prediction.csv",
        ],
    }
    paths = {}
    for k, opts in candidates.items():
        for p in opts:
            if Path(p).exists():
                paths[k] = p
                break
        if k not in paths:
            # Try recursive search under /kaggle/input
            root = Path("/kaggle/input")
            if root.exists():
                hits = list(root.rglob(Path(opts[0]).name))
                if hits:
                    paths[k] = str(hits[0])
        if k not in paths:
            raise FileNotFoundError(f"Could not find {k}; tried {opts}")
    print(f"[boot] data paths: {paths}", flush=True)
    return paths


def load_data(paths):
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    orig = pd.read_csv(paths["orig"])
    print(f"[load] train={train.shape} test={test.shape} orig={orig.shape}", flush=True)
    return train, test, orig


# ---------------------------------------------------------------------------
# FE FROM RAWASHISHSIN
# ---------------------------------------------------------------------------
TARGET = "Irrigation_Need"
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}

CAT_COLS = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season',
            'Irrigation_Type', 'Water_Source', 'Mulching_Used', 'Region']
NUM_COLS = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity',
            'Temperature_C', 'Humidity', 'Rainfall_mm', 'Sunlight_Hours',
            'Wind_Speed_kmh', 'Field_Area_hectare', 'Previous_Irrigation_mm']
TOP_CAT_COLS = ['Crop_Growth_Stage', 'Mulching_Used', 'Crop_Type']
TOP_NUM_COLS = ['Soil_Moisture', 'Temperature_C', 'Wind_Speed_kmh', 'Rainfall_mm']
TOP_COLS = ['Soil_Moisture', 'Crop_Growth_Stage', 'Temperature_C',
            'Mulching_Used', 'Wind_Speed_kmh', 'Rainfall_mm']


def add_threshold_distances(df):
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(int)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(int)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(int)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(int)
    df["moist_rain"] = df["Soil_Moisture"] / (df["Rainfall_mm"] + 1)
    df["moist_temp"] = df["Soil_Moisture"] / (df["Temperature_C"] + 1)
    df["moist_wind"] = df["Soil_Moisture"] / (df["Wind_Speed_kmh"] + 1)
    df["ET_proxy"] = (df["Temperature_C"] * df["Wind_Speed_kmh"] * df["Sunlight_Hours"]) / (df["Humidity"] + 1)
    df["heat_stress"] = df["Temperature_C"] * df["Sunlight_Hours"]
    df["drying_force"] = df["Wind_Speed_kmh"] * df["Temperature_C"] / (df["Humidity"] + 1)
    df["water_supply"] = df["Rainfall_mm"] + df["Previous_Irrigation_mm"]
    df["water_deficit"] = df["Soil_Moisture"] - df["water_supply"] * 0.1
    df["soil_quality"] = df["Organic_Carbon"] / (df["Electrical_Conductivity"] + 0.1)
    df["moist_x_temp"] = df["Soil_Moisture"] * df["Temperature_C"]
    df["wind_x_temp"] = df["Wind_Speed_kmh"] * df["Temperature_C"]
    return df


def add_formula_features(df):
    df['high_score'] = (
        (df['Soil_Moisture'] < 25) * 2 +
        (df['Rainfall_mm'] < 300) * 2 +
        (df['Temperature_C'] > 30) * 1 +
        (df['Wind_Speed_kmh'] > 10) * 1
    )
    df['low_score'] = (
        (df['Crop_Growth_Stage'].isin(['Harvest', 'Sowing'])) * 2 +
        (df['Mulching_Used'] == 'Yes') * 1
    )
    df['formula_score'] = df['high_score'] - df['low_score']
    df['formula_pred'] = 0
    df.loc[(df['formula_score'] > 0) & (df['formula_score'] <= 3), 'formula_pred'] = 1
    df.loc[df['formula_score'] > 3, 'formula_pred'] = 2
    return df


def add_ngrams(train_fe, test_fe, orig_fe):
    BIGRAM_COLS = []
    TRIGRAM_COLS = []
    for c1, c2 in combinations(TOP_CAT_COLS, 2):
        col_name = f"BG_{c1}_{c2}"
        for df in [train_fe, test_fe, orig_fe]:
            df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str)
        BIGRAM_COLS.append(col_name)
    for c1, c2, c3 in combinations(TOP_CAT_COLS, 3):
        col_name = f"TG_{c1}_{c2}_{c3}"
        for df in [train_fe, test_fe, orig_fe]:
            df[col_name] = df[c1].astype(str) + "_" + df[c2].astype(str) + "_" + df[c3].astype(str)
        TRIGRAM_COLS.append(col_name)
    NGRAM = BIGRAM_COLS + TRIGRAM_COLS
    for col in NGRAM:
        combined = pd.concat([train_fe[col], test_fe[col], orig_fe[col]], axis=0).astype(str)
        labels, _ = pd.factorize(combined)
        n_train = len(train_fe)
        n_test = len(test_fe)
        train_fe[col] = labels[:n_train]
        test_fe[col] = labels[n_train: n_train + n_test]
        orig_fe[col] = labels[n_train + n_test:]
    return NGRAM


def add_bin_cat_int(train_fe, test_fe, orig_fe):
    BIN_CAT_INT_COLS = []
    for num_col in TOP_NUM_COLS:
        bin_col = f"{num_col}_bin"
        train_fe[bin_col], bins = pd.qcut(train_fe[num_col], q=5, labels=False,
                                          retbins=True, duplicates='drop')
        for df in [test_fe, orig_fe]:
            df[bin_col] = pd.cut(df[num_col], bins=bins, labels=False,
                                 include_lowest=True).fillna(0).astype(int)
        for cat_col in TOP_CAT_COLS:
            int_name = f"{num_col}_bin_{cat_col}_int"
            for df in [train_fe, test_fe, orig_fe]:
                df[int_name] = df[bin_col].astype(str) + "_" + df[cat_col].astype(str)
            BIN_CAT_INT_COLS.append(int_name)
    for df in [train_fe, test_fe, orig_fe]:
        df.drop(columns=[f"{num_col}_bin" for num_col in TOP_NUM_COLS], inplace=True)
    for col in BIN_CAT_INT_COLS:
        codes, uniques = pd.factorize(train_fe[col])
        train_fe[col] = codes.astype('int')
        mapping = {val: i for i, val in enumerate(uniques)}
        test_fe[col] = test_fe[col].map(mapping).fillna(-1).astype(int)
        orig_fe[col] = orig_fe[col].map(mapping).fillna(-1).astype(int)
    return BIN_CAT_INT_COLS


def add_num_cat_agg(train_fe, test_fe, orig_fe):
    NUM_CAT_AGG_COLS = []
    for cat_col in TOP_CAT_COLS:
        for num_col in TOP_NUM_COLS:
            group_means = train_fe.groupby(cat_col)[num_col].mean()
            for df in [train_fe, test_fe, orig_fe]:
                avg_name = f"{num_col}_avg_by_{cat_col}"
                df[avg_name] = df[cat_col].map(group_means).astype('float')
                diff_name = f"{num_col}_diff_{cat_col}"
                df[diff_name] = (df[num_col] - df[avg_name]).astype('float')
                ratio_name = f"{num_col}_ratio_{cat_col}"
                df[ratio_name] = (df[num_col] / (df[avg_name] + 1e-6)).astype('float')
                NUM_CAT_AGG_COLS.extend([avg_name, diff_name, ratio_name])
    return list(set(NUM_CAT_AGG_COLS))


def add_round_digit_decimal_bins(train_fe, test_fe, orig_fe):
    round_config = {'Soil_Moisture': [0, -1], 'Temperature_C': [-1],
                    'Rainfall_mm': [0, -1, -2, -3], 'Wind_Speed_kmh': [0, -1]}
    digit_config = {'Soil_Moisture': [-1, 0, 1, 2], 'Temperature_C': [-1, 0, 1, 2],
                    'Rainfall_mm': [-3, -2, -1, 0, 1, 2], 'Wind_Speed_kmh': [-1, 0, 1, 2]}
    ROUND, DIGITS, DECIMALS, BINS = [], [], [], []
    for col, r_values in round_config.items():
        for r in r_values:
            feat = f"{col}_r{r}"
            for df in [train_fe, test_fe, orig_fe]:
                df[feat] = df[col].round(r)
            ROUND.append(feat)
    for col, k_values in digit_config.items():
        for k in k_values:
            feat = f"{col}_d{k}"
            for df in [train_fe, test_fe, orig_fe]:
                df[feat] = ((df[col] * 10 ** k) % 10).astype(int)
            DIGITS.append(feat)
    for col in TOP_NUM_COLS:
        feat = f"{col}_decimal"
        for df in [train_fe, test_fe, orig_fe]:
            df[feat] = (df[col] % 1).round(2)
        DECIMALS.append(feat)
    for col in TOP_NUM_COLS:
        feat = f"{col}_bin"
        train_fe[feat], bin_edges = pd.qcut(train_fe[col], q=10, labels=False, duplicates='drop', retbins=True)
        test_fe[col] = test_fe[col].clip(bin_edges[0], bin_edges[-1])
        test_fe[feat] = pd.cut(test_fe[col], bins=bin_edges, labels=False, include_lowest=True).astype(int)
        orig_fe[col] = orig_fe[col].clip(bin_edges[0], bin_edges[-1])
        orig_fe[feat] = pd.cut(orig_fe[col], bins=bin_edges, labels=False, include_lowest=True).astype(int)
        BINS.append(feat)
    return ROUND, DIGITS, DECIMALS, BINS


def add_pairs(train_fe, test_fe, orig_fe):
    PAIRS = []
    train_len = len(train_fe)
    test_len = len(test_fe)
    combined_len = train_len + test_len + len(orig_fe)
    for col1, col2 in combinations(TOP_COLS, 2):
        name = f"{col1}__{col2}"
        combined_str = pd.concat([
            train_fe[col1].astype(str) + '_' + train_fe[col2].astype(str),
            test_fe[col1].astype(str) + '_' + test_fe[col2].astype(str),
            orig_fe[col1].astype(str) + '_' + orig_fe[col2].astype(str),
        ], ignore_index=True)
        combined_codes, _ = pd.factorize(combined_str)
        del combined_str; gc.collect()
        n_unique = len(np.unique(combined_codes))
        if n_unique > combined_len // 2 or n_unique <= 1:
            del combined_codes
            continue
        train_fe[name] = combined_codes[:train_len]
        test_fe[name] = combined_codes[train_len:train_len + test_len]
        orig_fe[name] = combined_codes[train_len + test_len:]
        PAIRS.append(name)
        del combined_codes; gc.collect()
    return PAIRS


# ---------------------------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------------------------
def tune_log_bias(probs, y_true, eps=1e-15):
    from sklearn.metrics import balanced_accuracy_score
    bias = np.zeros(3, dtype=np.float64)
    log_p = np.log(np.clip(probs, eps, 1.0))
    best = balanced_accuracy_score(y_true, np.argmax(log_p + bias, axis=1))
    for step in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
        improved = True
        while improved:
            improved = False
            for ci in range(3):
                for d in (-1.0, 1.0):
                    c = bias.copy()
                    c[ci] += d * step
                    s = balanced_accuracy_score(y_true, np.argmax(log_p + c, axis=1))
                    if s > best + 1e-9:
                        bias = c; best = s; improved = True
    return bias, float(best)


def main():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import TargetEncoder
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    paths = find_data()
    train, test, orig = load_data(paths)

    # Map target
    train[TARGET] = train[TARGET].map(TARGET_MAP)
    orig[TARGET] = orig[TARGET].map(TARGET_MAP)

    # SMOKE: subsample
    if IS_SMOKE:
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.head(5_000).reset_index(drop=True)
        print(f"[smoke] subsampled train={train.shape} test={test.shape}", flush=True)

    # FE pipeline
    print("[fe] building features...", flush=True)
    train_fe = train.copy()
    test_fe = test.copy()
    orig_fe = orig.copy()
    train_fe = add_threshold_distances(train_fe)
    test_fe = add_threshold_distances(test_fe)
    orig_fe = add_threshold_distances(orig_fe)
    train_fe = add_formula_features(train_fe)
    test_fe = add_formula_features(test_fe)
    orig_fe = add_formula_features(orig_fe)
    NGRAM = add_ngrams(train_fe, test_fe, orig_fe)
    BIN_CAT_INT = add_bin_cat_int(train_fe, test_fe, orig_fe)
    NUM_CAT_AGG = add_num_cat_agg(train_fe, test_fe, orig_fe)
    ROUND, DIGITS, DECIMALS, BINS = add_round_digit_decimal_bins(train_fe, test_fe, orig_fe)
    PAIRS = add_pairs(train_fe, test_fe, orig_fe)
    print(f"[fe] NGRAM={len(NGRAM)} BIN_CAT_INT={len(BIN_CAT_INT)} NUM_CAT_AGG={len(NUM_CAT_AGG)} "
          f"ROUND={len(ROUND)} DIGITS={len(DIGITS)} DECIMALS={len(DECIMALS)} BINS={len(BINS)} PAIRS={len(PAIRS)}",
          flush=True)

    # cat dtype
    for df in [train_fe, test_fe, orig_fe]:
        for col in CAT_COLS:
            df[col] = df[col].astype('category')

    X_train = train_fe.drop([TARGET, "id"], axis=1)
    y_train = train_fe[TARGET].values
    X_orig = orig_fe.drop(TARGET, axis=1)
    y_orig = orig_fe[TARGET].values
    X_test = test_fe.drop("id", axis=1)
    print(f"[shape] X_train={X_train.shape} X_orig={X_orig.shape} X_test={X_test.shape}", flush=True)

    # XGB params (rawashishsin's exact)
    USE_GPU = os.environ.get("USE_GPU", "1") == "1"
    # Reduced from rawashishsin's 2600 -> 1500 to fit Kaggle 60min GPU cap
    # at depth=3 + lr=0.05 the val loss curve from SMOKE shows convergence by ~800 rounds
    N_EST_PROD = int(os.environ.get("N_EST", "2600"))
    n_est = 300 if IS_SMOKE else N_EST_PROD
    params_xgb = {
        'objective': 'multi:softprob', 'num_class': 3,
        'n_estimators': n_est, 'learning_rate': 0.05,
        'max_depth': 3, 'subsample': 0.9, 'colsample_bytree': 0.8,
        'max_bin': 1100, 'eval_metric': 'mlogloss', 'n_jobs': -1,
        'random_state': SEED, 'enable_categorical': True,
    }
    if USE_GPU:
        params_xgb['device'] = 'cuda'
        params_xgb['tree_method'] = 'hist'
    print(f"[xgb] params: {params_xgb}", flush=True)

    # 5-fold StratifiedKFold(seed=42) -> aligned with all our other OOFs
    # NB: split is INDEPENDENT of TE_SEED (we vary only the encoder seed).
    N_FOLDS = 1 if IS_PROBE else 5
    if IS_SMOKE:
        N_FOLDS = 2
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_splits_all = list(skf.split(X_train, y_train))
    if IS_PROBE:
        fold_splits_all = fold_splits_all[:1]
    if IS_SMOKE:
        fold_splits_all = fold_splits_all[:2]

    ORIG_ROW_WEIGHT = 0.5
    DROP_COLS = PAIRS + NUM_CAT_AGG + BIN_CAT_INT
    te_features = X_train.columns.tolist()

    # Hard wall-time cap per CLAUDE.md GPU rule (covers ALL TE_SEEDS)
    WALL_KILL_SEC = int(os.environ.get("WALL_KILL_SEC", str(60 * 60 * 8)))  # 8h default
    t_start = time.time()
    print(f"[wall] hard kill at t+{WALL_KILL_SEC}s ({WALL_KILL_SEC//60}min) for ALL seeds", flush=True)

    all_seed_results = {}

    for seed_idx, te_seed in enumerate(TE_SEEDS):
        elapsed = time.time() - t_start
        if elapsed >= WALL_KILL_SEC:
            print(f"[wall-kill] before TE_SEED={te_seed} (elapsed {elapsed:.0f}s)", flush=True)
            break
        SEED_TAG = f"_te{te_seed}" if not IS_SMOKE else f"_smoke_te{te_seed}"
        oof_path = OUT / f"oof_rawashishsin{SEED_TAG}.npy"
        test_path = OUT / f"test_rawashishsin{SEED_TAG}.npy"
        results_path = OUT / f"rawashishsin{SEED_TAG}_results.json"
        print(f"\n=== TE_SEED={te_seed}  ({seed_idx+1}/{len(TE_SEEDS)})  outputs: {SEED_TAG} ===",
              flush=True)
        oof_preds = np.zeros((len(X_train), 3))
        test_preds = np.zeros((len(X_test), 3))
        fold_scores = []

        for fold_idx, (tr_idx, va_idx) in enumerate(fold_splits_all):
            elapsed = time.time() - t_start
            if elapsed >= WALL_KILL_SEC:
                print(f"[wall-kill] inside seed {te_seed} fold {fold_idx} "
                      f"(elapsed {elapsed:.0f}s)", flush=True)
                break
            ckpt_oof = OUT / f"oof_rawashishsin{SEED_TAG}_fold{fold_idx}.npy"
            ckpt_test = OUT / f"test_rawashishsin{SEED_TAG}_fold{fold_idx}.npy"
            ckpt_va = OUT / f"va_idx_rawashishsin{SEED_TAG}_fold{fold_idx}.npy"
            if ckpt_oof.exists() and ckpt_test.exists() and ckpt_va.exists():
                print(f"[seed {te_seed}][fold {fold_idx}] checkpoint found, loading", flush=True)
                saved_va = np.load(ckpt_va)
                saved_oof = np.load(ckpt_oof)
                saved_test = np.load(ckpt_test)
                oof_preds[saved_va] = saved_oof
                test_preds += saved_test / len(fold_splits_all)
                ba = balanced_accuracy_score(y_train[saved_va], np.argmax(saved_oof, axis=1))
                fold_scores.append(ba)
                print(f"  loaded BA={ba:.6f}", flush=True)
                continue

            print(f"\n[seed {te_seed}][fold {fold_idx+1}/{len(fold_splits_all)}]", flush=True)
            t0 = time.time()
            X_tr_real = X_train.iloc[tr_idx].copy()
            y_tr_real = y_train[tr_idx]
            X_val = X_train.iloc[va_idx].copy()
            y_val = y_train[va_idx]
            X_tr_combined = pd.concat([X_tr_real, X_orig], axis=0).reset_index(drop=True)
            y_tr_combined = np.concatenate([y_tr_real, y_orig])

            # *** N1 KEY DIFFERENCE: TE encoder uses te_seed instead of fixed SEED ***
            encoder = TargetEncoder(target_type="multiclass", smooth="auto", cv=5,
                                    random_state=te_seed)
            X_tr_te = encoder.fit_transform(X_tr_combined[te_features], y_tr_combined)
            X_val_te = encoder.transform(X_val[te_features])
            X_test_te = encoder.transform(X_test[te_features])
            te_cols = []
            for col in te_features:
                for class_id in range(3):
                    te_cols.append(f"TE_{col}_class{class_id}")
            X_tr_combined[te_cols] = X_tr_te
            X_val[te_cols] = X_val_te
            X_test_copy = X_test.copy()
            X_test_copy[te_cols] = X_test_te

            X_tr_final = X_tr_combined.drop(columns=DROP_COLS)
            X_val_final = X_val.drop(columns=DROP_COLS)
            X_test_final = X_test_copy.drop(columns=DROP_COLS)

            sample_weights = compute_sample_weight('balanced', y_tr_combined).astype(np.float32)
            sample_weights[len(tr_idx):] *= ORIG_ROW_WEIGHT

            model = XGBClassifier(**params_xgb)
            model.fit(X_tr_final, y_tr_combined, eval_set=[(X_val_final, y_val)],
                      sample_weight=sample_weights, verbose=400)
            val_proba = model.predict_proba(X_val_final)
            test_proba_fold = model.predict_proba(X_test_final)

            oof_preds[va_idx] = val_proba
            test_preds += test_proba_fold / len(fold_splits_all)
            fold_ba = balanced_accuracy_score(y_val, np.argmax(val_proba, axis=1))
            fold_scores.append(fold_ba)
            wall = time.time() - t0
            print(f"  BA={fold_ba:.6f}  wall={wall:.1f}s", flush=True)

            np.save(ckpt_oof, val_proba)
            np.save(ckpt_test, test_proba_fold)
            np.save(ckpt_va, va_idx)
            del model, X_tr_combined, X_val, X_test_copy
            del X_tr_final, X_val_final, X_test_final
            del encoder, X_tr_te, X_val_te, X_test_te
            gc.collect()

        n_completed = sum(1 for i in range(len(fold_splits_all))
                          if (OUT / f"oof_rawashishsin{SEED_TAG}_fold{i}.npy").exists())
        if n_completed == len(fold_splits_all):
            oof_argmax = balanced_accuracy_score(y_train, np.argmax(oof_preds, axis=1))
            bias, oof_tuned = tune_log_bias(oof_preds, y_train)
            print(f"\n[seed {te_seed}] argmax OOF = {oof_argmax:.6f}  "
                  f"tuned = {oof_tuned:.6f}  bias={bias.round(4).tolist()}", flush=True)
            np.save(oof_path, oof_preds)
            np.save(test_path, test_preds)
            results = {
                "te_seed": te_seed,
                "oof_argmax": float(oof_argmax),
                "oof_tuned": float(oof_tuned),
                "tuned_bias": [float(x) for x in bias],
                "fold_scores": [float(s) for s in fold_scores],
                "n_folds": len(fold_splits_all),
                "ORIG_ROW_WEIGHT": ORIG_ROW_WEIGHT,
            }
            results_path.write_text(json.dumps(results, indent=2))
            all_seed_results[te_seed] = results
            print(f"[done] saved {oof_path} {test_path}", flush=True)
        else:
            print(f"[partial] seed {te_seed}: {n_completed}/{len(fold_splits_all)} folds done"
                  f" — checkpoints kept, will resume on next run", flush=True)
            # Save partial for inspection
            np.save(oof_path, oof_preds)
            np.save(test_path, test_preds)
            results = {
                "te_seed": te_seed,
                "n_completed_folds": int(n_completed),
                "n_total_folds": int(len(fold_splits_all)),
                "fold_scores_partial": [float(s) for s in fold_scores],
            }
            results_path.write_text(json.dumps(results, indent=2))
            all_seed_results[te_seed] = results

    # Top-level summary
    summary_path = OUT / f"rawashishsin_bag_{MODE}_summary.json"
    summary_path.write_text(json.dumps(
        {"te_seeds": TE_SEEDS, "results": all_seed_results,
         "wall_total_s": time.time() - t_start},
        indent=2,
    ))
    print(f"\n[summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
