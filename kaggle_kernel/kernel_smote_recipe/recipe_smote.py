"""SMOTE-NC + recipe XGB on Kaggle GPU.

Hypothesis: 3.3% High-class prior bottlenecks per-class recall on the
LB-best 4-stack at 0.9774. Synthesizing extra High rows via SMOTE-NC
should lift training-side High signal density and push High recall
above the Pareto frontier, unlocking real blend orthogonality.

Why this is the only own-pipeline lever left:
  - Tier 1c (2026-04-25) confirmed the 63-component meta-stacker bank is
    saturated against greedy-add, meta-on-meta, and seed-bagging.
  - Cross-pollination v3 (2026-04-25) added 2 components → LB regression
    -0.00034 from current LB-best 0.98094.
  - Every architectural NN/FE/blend/calibration lever has been tested.
  - SMOTE-NC is training-data-level — fundamentally new signal source.

CRITICAL: SMOTE-NC on the full 443-feature recipe matrix OOMs at 45 GiB
(one-hot expansion of high-card combo features). Fix: run SMOTE-NC on
RAW 19 cols (8 cats + 11 nums) only, then re-derive combos / digits /
num_as_cat / freq / orig_stats on the augmented rows using saved
factorize / freq / orig_stats maps. Memory peak drops to ~100 MB.

Wall-time budget: 1h hard cap (CLAUDE.md GPU rule). Promise gate after
fold 1 — if fold-1 argmax < 0.97500 OR fold-1 High recall < 0.965, abort
and save partial outputs. If gate passes, continue 5 folds with hard
total kill at t+55min.

Outputs (to /kaggle/working/):
  oof_recipe_smote.npy
  test_recipe_smote.npy
  recipe_smote_results.json (per-fold metrics + gate decision)
  submission_recipe_smote_tuned.csv (only if all 5 folds completed)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path


# ============================== boot =========================================
def _ensure(pkg: str, install: str | None = None) -> None:
    try:
        __import__(pkg.split("[")[0])
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", install or pkg]
        )


_ensure("imblearn", "imbalanced-learn")
_ensure("xgboost")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from imblearn.over_sampling import SMOTENC  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

print(f"[boot] xgboost {xgb.__version__}", flush=True)
print(f"[boot] imblearn {__import__('imblearn').__version__}", flush=True)

try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"], text=True, timeout=10).strip()
    print(f"[boot] GPU: {out}", flush=True)
    USE_GPU = True
except Exception as e:
    print(f"[boot] no GPU: {e}", flush=True)
    USE_GPU = False


KAGGLE_INPUT = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def _find_one(pattern: str) -> Path:
    for p in KAGGLE_INPUT.rglob(pattern):
        return p
    raise FileNotFoundError(f"no match for {pattern} under {KAGGLE_INPUT}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ============================== config =======================================
SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

SMOTE_TARGET = int(os.environ.get("SMOTE_TARGET", "42000"))
SMOTE_K = int(os.environ.get("SMOTE_K", "5"))

# Promise-gate (after fold 1, before continuing folds 2-5)
GATE_ARGMAX_FLOOR = 0.97500
GATE_HIGH_FLOOR = 0.965
GATE_HIGH_LIFT = 0.005   # +0.5pp High over recipe baseline = "lever working"
GATE_ERROR_CEIL = 1.05   # max relative error count vs recipe baseline

# Recipe-fold-1 reference numbers (from prior runs — used for gate diagnostics)
RECIPE_FOLD1_HIGH_RECALL = 0.977

# Wall-time hard kills
TOTAL_KILL_SEC = 55 * 60
FOLD1_KILL_SEC = 25 * 60   # if fold 1 stalls > 25 min, abort

XGB_PARAMS = dict(
    n_estimators=300 if SMOKE else 3000,
    max_depth=4,
    max_leaves=30,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    reg_alpha=5,
    reg_lambda=5,
    max_bin=256 if SMOKE else 1024,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=SEED,
    early_stopping_rounds=50 if SMOKE else 200,
    verbosity=0,
)
if USE_GPU:
    XGB_PARAMS["device"] = "cuda"

RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])


# ============================== FE blocks ====================================
_LOGIT_COEFS = {
    "Low":    dict(bias=16.3173,
                   soil_lt_25=-11.0237, temp_gt_30=-5.8559,
                   rain_lt_300=-10.8500, wind_gt_10=-5.8284,
                   stage=dict(Flowering=-5.4155, Harvest=5.5073,
                              Sowing=5.2299, Vegetative=-5.4617),
                   mulch=dict(No=-3.0014, Yes=2.8613)),
    "Medium": dict(bias=4.6524,
                   soil_lt_25=0.3290, temp_gt_30=-0.0204,
                   rain_lt_300=0.1542, wind_gt_10=0.0841,
                   stage=dict(Flowering=0.3586, Harvest=-0.1348,
                              Sowing=-0.3547, Vegetative=0.3334),
                   mulch=dict(No=0.1883, Yes=0.0142)),
    "High":   dict(bias=-20.9697,
                   soil_lt_25=10.6947, temp_gt_30=5.8763,
                   rain_lt_300=10.6958, wind_gt_10=5.7444,
                   stage=dict(Flowering=5.0569, Harvest=-5.3725,
                              Sowing=-4.8752, Vegetative=5.1283),
                   mulch=dict(No=2.8131, Yes=-2.8755)),
}


def add_threshold_flags(df):
    df["soil_lt_25"] = (df["Soil_Moisture"] < 25).astype(np.int8)
    df["temp_gt_30"] = (df["Temperature_C"] > 30).astype(np.int8)
    df["rain_lt_300"] = (df["Rainfall_mm"] < 300).astype(np.int8)
    df["wind_gt_10"] = (df["Wind_Speed_kmh"] > 10).astype(np.int8)
    return ["soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"]


def add_lr_formula_logits(df):
    stage = df["Crop_Growth_Stage"].astype(str).values
    mulch = df["Mulching_Used"].astype(str).values
    soil = df["soil_lt_25"].values
    temp = df["temp_gt_30"].values
    rain = df["rain_lt_300"].values
    wind = df["wind_gt_10"].values
    cols = []
    for cls, coefs in _LOGIT_COEFS.items():
        logit = (coefs["bias"]
                 + coefs["soil_lt_25"] * soil
                 + coefs["temp_gt_30"] * temp
                 + coefs["rain_lt_300"] * rain
                 + coefs["wind_gt_10"] * wind)
        stage_vals = np.array([coefs["stage"].get(s, 0.0) for s in stage])
        mulch_vals = np.array([coefs["mulch"].get(m, 0.0) for m in mulch])
        name = f"logit_P_{cls}"
        df[name] = (logit + stage_vals + mulch_vals).astype(np.float32)
        cols.append(name)
    return cols


def add_cat_pair_combos_with_map(train, test, orig, cats):
    """Returns (new_cols, combo_pairs, combo_maps)."""
    new_cols, combo_pairs, combo_maps = [], {}, {}
    for c1, c2 in combinations(cats, 2):
        col = f"COMBO_{c1}_{c2}"
        for df in (train, test, orig):
            df[col] = df[c1].astype(str) + "_" + df[c2].astype(str)
        combined = pd.concat([train[col], test[col], orig[col]])
        codes, uniques = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[col] = codes[:s]
        test[col] = codes[s:t]
        orig[col] = codes[t:]
        combo_pairs[col] = (c1, c2)
        combo_maps[col] = {v: i for i, v in enumerate(uniques)}
        new_cols.append(col)
    return new_cols, combo_pairs, combo_maps


def add_digit_features(train, test, orig, nums, digit_range=range(-4, 4)):
    cols = []
    for c in nums:
        for k in digit_range:
            name = f"{c}_digit{k}"
            for df in (train, test, orig):
                df[name] = (df[c] // (10.0 ** k) % 10).astype("int8")
            cols.append(name)
    drop = [c for c in cols if test[c].nunique() == 1]
    for c in drop:
        for df in (train, test, orig):
            df.drop(columns=[c], inplace=True)
    return [c for c in cols if c not in drop]


def add_num_as_cat_with_map(train, test, orig, nums):
    new_cols, nac_maps = [], {}
    for c in nums:
        name = f"CAT_{c}"
        for df in (train, test, orig):
            df[name] = df[c].astype(str)
        combined = pd.concat([train[name], test[name], orig[name]])
        codes, uniques = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[name] = codes[:s]
        test[name] = codes[s:t]
        orig[name] = codes[t:]
        nac_maps[name] = {v: i for i, v in enumerate(uniques)}
        new_cols.append(name)
    return new_cols, nac_maps


def add_freq_features_with_map(train, test, orig, cats):
    new_cols, freq_maps = [], {}
    for c in cats:
        freq = pd.concat([train[c], test[c], orig[c]]).value_counts(normalize=True)
        name = f"FREQ_{c}"
        for df in (train, test, orig):
            df[name] = df[c].map(freq).fillna(0).astype(np.float32)
        new_cols.append(name)
        freq_maps[c] = freq.to_dict()
    return new_cols, freq_maps


def add_orig_mean_std_with_map(train, test, orig, cols_to_aggregate, target):
    new_cols, orig_stat_maps = [], {}
    for c in cols_to_aggregate:
        stats = orig.groupby(c)[target].agg(["mean", "std"]).reset_index()
        stats.columns = [c, f"ORIG_{c}_mean", f"ORIG_{c}_std"]
        for df_name in ("train", "test"):
            df = {"train": train, "test": test}[df_name]
            merged = df.merge(stats, on=c, how="left")
            df[f"ORIG_{c}_mean"] = merged[f"ORIG_{c}_mean"].fillna(0.5).astype(np.float32).values
            df[f"ORIG_{c}_std"]  = merged[f"ORIG_{c}_std"].fillna(0).astype(np.float32).values
        new_cols += [f"ORIG_{c}_mean", f"ORIG_{c}_std"]
        orig_stat_maps[c] = dict(
            mean=dict(zip(stats[c], stats[f"ORIG_{c}_mean"])),
            std=dict(zip(stats[c], stats[f"ORIG_{c}_std"])),
        )
    return new_cols, orig_stat_maps


def factorize_raw_cats_with_map(train, test, orig, cats):
    cat_maps = {}
    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, uniques = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]
        cat_maps[c] = {v: i for i, v in enumerate(uniques)}
    return cat_maps


# ============================== OrderedTE ====================================
class OrderedTE:
    """Per-row cumulative LOO target encoder. Same as recipe_ote.OrderedTE."""

    def __init__(self, a: float = 1.0) -> None:
        from functools import reduce
        self._reduce = reduce
        self.a = float(a)
        self.classes_ = None
        self.prior_ = None
        self.stats_: dict = {}
        self.cols_: list[str] = []

    def fit(self, df, cat_cols, target):
        from functools import reduce
        y = df[target].to_numpy()
        self.classes_ = np.array(sorted(pd.unique(y)))
        counts = np.array([(y == k).sum() for k in self.classes_], dtype=np.float64)
        self.prior_ = counts / counts.sum()
        self.cols_ = list(cat_cols)
        te_cols_out = {}
        for c in self.cols_:
            stats_list = []
            key = df[c].to_numpy()
            for k, cls in enumerate(self.classes_):
                y_bin = (df[target] == cls).astype(np.int32).to_numpy()
                grp = pd.DataFrame({c: key, "y": y_bin})
                grouped = grp.groupby(c, observed=True, sort=False)["y"]
                cum_cnt = grouped.cumcount().to_numpy()
                cum_sum_incl = grouped.cumsum().to_numpy()
                cum_sum_excl = cum_sum_incl - y_bin
                prior = self.prior_[k]
                te = (cum_sum_excl + self.a * prior) / (cum_cnt + self.a)
                te_cols_out[f"{c}_TE_cls{cls}"] = te.astype(np.float32)
                agg = grouped.agg(["count", "sum"]).reset_index()
                agg.columns = [c, f"{c}_n_{cls}", f"{c}_s_{cls}"]
                stats_list.append(agg)
            self.stats_[c] = reduce(
                lambda a_df, b_df: a_df.merge(b_df, on=c, how="outer"),
                stats_list)
        te_df = pd.DataFrame(te_cols_out, index=df.index)
        return pd.concat([df, te_df], axis=1)

    def transform(self, df):
        te_cols_out = {}
        for c in self.cols_:
            stats = self.stats_[c]
            merged = df[[c]].merge(stats, on=c, how="left")
            for k, cls in enumerate(self.classes_):
                n_col = f"{c}_n_{cls}"
                s_col = f"{c}_s_{cls}"
                prior = self.prior_[k]
                n = merged[n_col].fillna(0).to_numpy()
                s = merged[s_col].fillna(0).to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    te = np.where(n > 0,
                                   (s + self.a * prior) / (n + self.a),
                                   prior)
                te_cols_out[f"{c}_TE_cls{cls}"] = te.astype(np.float32)
        te_df = pd.DataFrame(te_cols_out, index=df.index)
        return pd.concat([df, te_df], axis=1)

    def te_col_names(self):
        return [f"{c}_TE_cls{cls}" for c in self.cols_ for cls in self.classes_]


# ============================== SMOTE + redrive ==============================
def run_smote_nc_on_raw(raw_train_df, y_tr, target_n_high, *, cats,
                        k=5, random_state=42):
    """SMOTE-NC on the raw 19-col DF (cats as strings, nums as floats).

    `cats`: explicit list of categorical column names (pandas may report
    cats as 'category' instead of 'object' at production scale, breaking
    dtype-based inference).

    Memory ≈ n_synthesize × k × n_cols_after_onehot × 8 bytes
            ≈ 25k × 5 × 91 × 8 = 91 MB.
    """
    df = raw_train_df.copy()
    for c in cats:
        if c in df.columns and df[c].dtype.name != "object":
            df[c] = df[c].astype(str)
    cat_idx = [df.columns.get_loc(c) for c in cats if c in df.columns]
    if not cat_idx:
        raise RuntimeError(
            f"no cat cols in raw_train_df; have={list(df.columns)} cats={cats}")
    smote = SMOTENC(
        categorical_features=cat_idx,
        sampling_strategy={2: target_n_high},
        k_neighbors=k,
        random_state=random_state,
    )
    return smote.fit_resample(df, y_tr)


def re_derive_fe_aug(raw_aug, *, cats, nums, combo_pairs, combo_maps,
                     nac_maps, freq_maps, orig_stat_maps, cat_maps,
                     digit_cols_keep,
                     digit_range=range(-4, 4)):
    """Re-derive the full FE matrix on SMOTE-augmented raw rows.

    Order matches load_and_engineer:
      threshold_flags + lr_logits → combos → digits → num_as_cat →
      freq → orig_stats → factorize raw cats.
    """
    df = raw_aug.copy().reset_index(drop=True)

    # 1. Threshold flags + LR logits
    add_threshold_flags(df)
    add_lr_formula_logits(df)

    # 2. Combos: str-concat → lookup
    for combo_name, (c1, c2) in combo_pairs.items():
        keys = df[c1].astype(str) + "_" + df[c2].astype(str)
        df[combo_name] = keys.map(combo_maps[combo_name]).fillna(-1).astype(np.int32)

    # 3. Digits: row-wise on raw nums; keep only digit_cols_keep (from prod FE)
    for c in nums:
        for k in digit_range:
            name = f"{c}_digit{k}"
            df[name] = (df[c] // (10.0 ** k) % 10).astype("int8")
    all_digit_names = [f"{c}_digit{k}" for c in nums for k in digit_range]
    for name in all_digit_names:
        if name not in digit_cols_keep and name in df.columns:
            df.drop(columns=[name], inplace=True)

    # 4. Num-as-cat: str-cast → lookup
    for c in nums:
        name = f"CAT_{c}"
        df[name] = df[c].astype(str).map(nac_maps[name]).fillna(-1).astype(np.int32)

    # 5. FREQ: raw cat (str-key) + combo (int-key)
    for c in cats:
        df[f"FREQ_{c}"] = df[c].map(freq_maps[c]).fillna(0).astype(np.float32)
    for combo_name in combo_maps.keys():
        df[f"FREQ_{combo_name}"] = df[combo_name].map(
            freq_maps[combo_name]).fillna(0).astype(np.float32)

    # 6. ORIG mean/std: nums (float-key, mostly miss → 0.5) + cats (str-key)
    for c in (nums + cats):
        df[f"ORIG_{c}_mean"] = df[c].map(
            orig_stat_maps[c]["mean"]).fillna(0.5).astype(np.float32)
        df[f"ORIG_{c}_std"] = df[c].map(
            orig_stat_maps[c]["std"]).fillna(0).astype(np.float32)

    # 7. Final raw-cat factorize
    for c in cats:
        df[c] = df[c].astype(str).map(cat_maps[c]).fillna(-1).astype(np.int32)

    return df


# ============================== load + engineer ==============================
def load_and_engineer():
    """Run full FE pipeline on (train, test, orig). Returns:
      train_fe, test_fe, raw_train (cats as strings + nums), info, test_ids,
      maps  (combo_pairs, combo_maps, nac_maps, freq_maps, orig_stat_maps,
             cat_maps, digit_cols)
    """
    log("loading train / test / orig")
    if "/kaggle/" in str(KAGGLE_INPUT) and KAGGLE_INPUT.exists():
        train_path = _find_one("train.csv")
        test_path = _find_one("test.csv")
        orig_path = None
        for pattern in ("irrigation_prediction.csv", "Irrigation_Prediction.csv",
                        "irrigation-prediction.csv"):
            try:
                orig_path = _find_one(pattern)
                break
            except FileNotFoundError:
                continue
        if orig_path is None:
            raise FileNotFoundError(
                f"no orig CSV found under {KAGGLE_INPUT}; "
                f"check dataset_sources in kernel-metadata.json")
    else:
        train_path = Path("data/train.csv")
        test_path = Path("data/test.csv")
        orig_path = Path("data/archive.zip")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = pd.read_csv(orig_path)

    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)

    if SMOKE:
        log("SMOKE=1 → 20k train / 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
        test_ids = test_ids[:len(test)]

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)} cats={len(cats)} train={len(train):,} "
        f"test={len(test):,} orig={len(orig):,}")

    # Snapshot raw train (cats as strings + nums + y) for SMOTE later
    raw_train = train[cats + nums + [TARGET]].copy()

    log("threshold flags + LR logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
        logits = add_lr_formula_logits(df)

    log("cat × cat pair combos")
    combos, combo_pairs, combo_maps = add_cat_pair_combos_with_map(
        train, test, orig, cats)

    log("digit features")
    digits = add_digit_features(train, test, orig, nums)

    log("num-as-cat")
    num_as_cat, nac_maps = add_num_as_cat_with_map(train, test, orig, nums)

    log("FREQ features")
    freq, freq_maps = add_freq_features_with_map(train, test, orig, cats + combos)

    log("ORIG mean/std")
    orig_stats, orig_stat_maps = add_orig_mean_std_with_map(
        train, test, orig, nums + cats, TARGET)

    log("factorize raw cats")
    cat_maps = factorize_raw_cats_with_map(train, test, orig, cats)

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: cats={len(cats)} combos={len(combos)} "
        f"digits={len(digits)} num_as_cat={len(num_as_cat)} tres={len(tres)} "
        f"logits={len(logits)} freq={len(freq)} orig_stats={len(orig_stats)} "
        f"te_cols={len(info['te_cols'])}")
    maps = dict(
        combo_pairs=combo_pairs, combo_maps=combo_maps,
        nac_maps=nac_maps, freq_maps=freq_maps,
        orig_stat_maps=orig_stat_maps, cat_maps=cat_maps,
        digit_cols=digits,
    )
    return train, test, raw_train, info, test_ids, maps


# ============================== promise gate =================================
def evaluate_fold1_gate(oof_fold, y_fold, recipe_fold1_errs):
    """After fold 1, compute argmax bal_acc + per-class recall + error count.

    Returns (decision, metrics_dict). decision ∈ {"PROCEED", "ABORT"}.
    """
    pred = oof_fold.argmax(1)
    argmax_bal = balanced_accuracy_score(y_fold, pred)
    cm = confusion_matrix(y_fold, pred, labels=[0, 1, 2])
    recalls = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    errs = (pred != y_fold).sum()
    err_ratio = errs / max(recipe_fold1_errs, 1)

    pass_argmax = argmax_bal >= GATE_ARGMAX_FLOOR
    pass_high_floor = recalls[2] >= GATE_HIGH_FLOOR
    pass_high_lift = recalls[2] >= (RECIPE_FOLD1_HIGH_RECALL + GATE_HIGH_LIFT)
    pass_errs = err_ratio <= GATE_ERROR_CEIL

    # PROCEED if at least 2 of (argmax, high_floor, errs) pass,
    # OR High recall lifted by at least +0.5pp (lever working unambiguously).
    n_pass = int(pass_argmax) + int(pass_high_floor) + int(pass_errs)
    decision = "PROCEED" if (n_pass >= 2 or pass_high_lift) else "ABORT"

    metrics = dict(
        argmax_bal=float(argmax_bal),
        recall_low=float(recalls[0]),
        recall_med=float(recalls[1]),
        recall_high=float(recalls[2]),
        errors=int(errs),
        err_ratio=float(err_ratio),
        pass_argmax=bool(pass_argmax),
        pass_high_floor=bool(pass_high_floor),
        pass_high_lift=bool(pass_high_lift),
        pass_errs=bool(pass_errs),
        decision=decision,
    )
    return decision, metrics


# ============================== log-bias tune ================================
def tune_log_bias_coord(oof, y, prior, max_iter=20, step=0.01):
    """Coord-ascent on per-class log-bias for max balanced accuracy."""
    eps = 1e-12
    log_oof = np.log(np.clip(oof, eps, 1.0))
    bias = np.log(np.clip(prior, eps, 1.0)) * 0  # start at zero
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(1))

    grid = np.arange(-3.0, 6.0 + step, step)
    for _ in range(max_iter):
        improved = False
        for c in range(3):
            best_b = bias[c]
            best_local = best
            for v in grid:
                trial = bias.copy()
                trial[c] = v
                b = balanced_accuracy_score(y, (log_oof + trial).argmax(1))
                if b > best_local:
                    best_local = b
                    best_b = v
            if best_local > best:
                bias[c] = best_b
                best = best_local
                improved = True
        if not improved:
            break
    return bias, best


# ============================== run_cv =======================================
def run_cv(train, test, raw_train, info, maps):
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    fold1_metrics = None
    gate_decision = None
    folds_completed = 0
    t_start = time.time()

    smote_target = min(SMOTE_TARGET, 5000 if SMOKE else SMOTE_TARGET)
    cats = info["cats"]
    nums = info["nums"]

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        if time.time() - t_start > TOTAL_KILL_SEC:
            log(f"!! TOTAL_KILL ({TOTAL_KILL_SEC}s) reached at fold {fold}, "
                f"saving partial")
            break
        log(f"=== fold {fold}/{N_FOLDS} ===")
        t_fold0 = time.time()

        # 1. raw train fold (cats as str + nums + y)
        raw_tr = raw_train.iloc[tr_idx].copy().reset_index(drop=True)
        y_tr_full = raw_tr[TARGET].to_numpy()
        raw_tr_nolab = raw_tr.drop(columns=[TARGET])
        log(f"  raw fold-tr: {len(raw_tr_nolab):,} rows × "
            f"{raw_tr_nolab.shape[1]} cols (8 cats + 11 nums)")

        # 2. SMOTE-NC on raw 19 cols (explicit cats list)
        t_smote = time.time()
        try:
            raw_aug, y_aug = run_smote_nc_on_raw(
                raw_tr_nolab, y_tr_full, smote_target,
                cats=cats, k=SMOTE_K, random_state=SEED + fold,
            )
        except Exception as e:
            log(f"  SMOTE-NC failed: {e}; skipping aug for this fold")
            raw_aug = raw_tr_nolab.copy()
            y_aug = y_tr_full.copy()
        n_h_pre = int((y_tr_full == 2).sum())
        n_h_post = int((y_aug == 2).sum())
        log(f"  SMOTE-NC: {len(raw_tr_nolab):,} → {len(raw_aug):,} "
            f"(H {n_h_pre:,} → {n_h_post:,}, +{n_h_post - n_h_pre:,}) "
            f"wall={time.time() - t_smote:.1f}s")

        # 3. Re-derive FE on augmented raw rows
        t_fe = time.time()
        train_aug_fe = re_derive_fe_aug(
            raw_aug,
            cats=cats, nums=nums,
            combo_pairs=maps["combo_pairs"],
            combo_maps=maps["combo_maps"],
            nac_maps=maps["nac_maps"],
            freq_maps=maps["freq_maps"],
            orig_stat_maps=maps["orig_stat_maps"],
            cat_maps=maps["cat_maps"],
            digit_cols_keep=maps["digit_cols"],
        )
        train_aug_fe[TARGET] = y_aug
        log(f"  FE re-derive on aug: {len(train_aug_fe):,} rows × "
            f"{train_aug_fe.shape[1]} cols  wall={time.time() - t_fe:.1f}s")

        # 4. OTE on augmented train (per-fold, leak-free w.r.t. val)
        t_ote = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(train_aug_fe))
        X_tr_shuf = train_aug_fe.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr_aug = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(train.iloc[va_idx].reset_index(drop=True))
        X_te = te.transform(test.copy().reset_index(drop=True))
        log(f"  OTE fit+transform wall={time.time() - t_ote:.1f}s")

        # 5. XGB train
        feat_cols = numeric_feats + te.te_col_names()
        # For aug rows, threshold_flags + logits + freq + orig_stats + tres
        # were re-derived but the val/test side already had them from the
        # initial FE pass. So feat_cols matches across train/val/test.
        y_aug_arr = X_tr_aug[TARGET].to_numpy()
        sw = compute_sample_weight("balanced", y_aug_arr)

        log(f"  XGB train: {len(feat_cols)} feat × {len(X_tr_aug):,} rows")
        t_xgb = time.time()
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr_aug[feat_cols], y_aug_arr,
            sample_weight=sw,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500 if not SMOKE else 100,
        )
        log(f"  XGB done best_iter={model.best_iteration} "
            f"wall={time.time() - t_xgb:.1f}s")

        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        folds_completed += 1
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
            f"total_wall={time.time() - t_fold0:.1f}s")

        # Persist partial after every fold so abort/timeout retains progress
        np.save(OUT_DIR / "oof_recipe_smote.npy", oof)
        np.save(OUT_DIR / "test_recipe_smote.npy", test_pred * N_FOLDS / max(folds_completed, 1))

        # === fold-1 promise gate ===
        if fold == 1:
            decision, fold1_metrics = evaluate_fold1_gate(
                oof[va_idx], y[va_idx], len(va_idx) // 5)
            log(f"\n=== fold-1 PROMISE GATE ===")
            for k, v in fold1_metrics.items():
                log(f"  {k:20s} = {v}")
            log(f"  decision: {decision}")
            gate_decision = decision
            (OUT_DIR / "recipe_smote_fold1_gate.json").write_text(
                json.dumps(fold1_metrics, indent=2))
            if decision == "ABORT":
                log("=== ABORTING after fold 1 (gate failed) ===")
                break

    # Final aggregation
    if folds_completed > 0:
        # Re-normalize test_pred since we may have aborted early
        test_pred = test_pred * N_FOLDS / max(folds_completed, 1)
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                folds_completed=folds_completed,
                fold1_metrics=fold1_metrics,
                gate_decision=gate_decision)


# ============================== main =========================================
def main():
    log(f"config: SMOTE_TARGET={SMOTE_TARGET} K={SMOTE_K} "
        f"smoke={SMOKE} N_FOLDS={N_FOLDS} use_gpu={USE_GPU}")
    log(f"gate floors: argmax≥{GATE_ARGMAX_FLOOR} "
        f"high≥{GATE_HIGH_FLOOR} (lift≥{GATE_HIGH_LIFT}) "
        f"errs≤{GATE_ERROR_CEIL}× recipe")

    train, test, raw_train, info, test_ids, maps = load_and_engineer()
    result = run_cv(train, test, raw_train, info, maps)

    y = train[TARGET].to_numpy()
    folds = result["folds_completed"]

    # Save raw OOF + test arrays
    np.save(OUT_DIR / "oof_recipe_smote.npy", result["oof"])
    np.save(OUT_DIR / "test_recipe_smote.npy", result["test"])

    # Compute tuned bal_acc only if all folds ran
    tuned_bal = None
    bias = None
    if folds == N_FOLDS:
        prior = np.bincount(y, minlength=3) / len(y)
        bias, tuned_bal = tune_log_bias_coord(result["oof"], y.astype(np.int32),
                                               prior)
        log(f"tuned bal_acc = {tuned_bal:.5f}  bias={bias.round(4).tolist()}")

        eps = 1e-9
        test_log = np.log(np.clip(result["test"], eps, 1.0))
        pred_idx = (test_log + bias).argmax(1)
        sub = pd.DataFrame({"id": test_ids,
                            TARGET: [IDX2CLS[i] for i in pred_idx]})
        sub.to_csv(OUT_DIR / "submission_recipe_smote_tuned.csv", index=False)
        log(f"wrote submission_recipe_smote_tuned.csv "
            f"dist={dict(sub[TARGET].value_counts())}")
    else:
        log(f"only {folds}/{N_FOLDS} folds completed — skipping tuned + sub")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, smoke=SMOKE, use_gpu=USE_GPU,
        smote_target=SMOTE_TARGET, smote_k=SMOTE_K,
        folds_completed=folds,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=(
            float(balanced_accuracy_score(y, result["oof"].argmax(1)))
            if folds > 0 else None),
        tuned_log_bias_bal_acc=float(tuned_bal) if tuned_bal else None,
        log_bias=bias.tolist() if bias is not None else None,
        fold1_metrics=result["fold1_metrics"],
        gate_decision=result["gate_decision"],
    )
    (OUT_DIR / "recipe_smote_results.json").write_text(json.dumps(summary, indent=2))
    log("wrote recipe_smote_results.json")
    log(f"FINAL: gate={result['gate_decision']}  folds={folds}/{N_FOLDS}  "
        f"tuned={tuned_bal}")


if __name__ == "__main__":
    main()
