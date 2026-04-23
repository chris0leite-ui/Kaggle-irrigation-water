"""GPU XGBoost 171-pair recipe (Ali Afzal's pairwise-TE magic).

Difference from CPU `scripts/recipe_full_te_171pair.py`:
  - `tree_method='hist', device='cuda'` (5-10x faster than CPU)
  - Aggressive per-fold cleanup (del + gc.collect) to fit in Kaggle's 13GB
    RAM. CPU run OOM'd on fold 2 OTE-fit at 21GB; GPU offloads XGB histogram
    to VRAM, freeing ~4-8GB RAM that lets the 813-OTE feature set fit.
  - Bins 11 numerics to 16-bin quantile cats, builds all C(19,2)=171 pair
    combos vs recipe_full_te's C(8,2)=28 cat x cat combos.
  - Architecturally distinct from every prior tree blend leg — new feature
    surface, not just a different model on same features.

Expected runtime: ~40-50 min on Kaggle P100 (vs ~2.5h CPU with OOM).

Outputs (in /kaggle/working/):
  oof_recipe_171pair.npy
  test_recipe_171pair.npy
  recipe_171pair_results.json
  submission_recipe_171pair.csv
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from functools import reduce
from itertools import combinations
from pathlib import Path

# ========================= environment setup =========================
# Kaggle's default xgboost should have GPU support; verify or install fresh.
try:
    import xgboost as _xgb
    print(f"[boot] xgboost {_xgb.__version__}", flush=True)
    if _xgb.__version__ < "2.0":
        raise ImportError("need xgboost >= 2.0 for device='cuda'")
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "xgboost>=2.1"])
    import xgboost as _xgb
    print(f"[boot] reinstalled xgboost {_xgb.__version__}", flush=True)

# Log GPU availability.
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"],
        text=True, timeout=10,
    ).strip()
    print(f"[boot] GPU info: {out}", flush=True)
except Exception as e:
    print(f"[boot] nvidia-smi error: {e}", flush=True)

# Log RAM at start.
try:
    import psutil
    vm = psutil.virtual_memory()
    print(f"[boot] RAM total={vm.total//(1024**3)}GB free={vm.available//(1024**3)}GB",
          flush=True)
except ImportError:
    psutil = None

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
N_BINS = int(os.environ.get("N_BINS", "16"))

KAGGLE_INPUT = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(exist_ok=True, parents=True)

SMOKE = os.environ.get("SMOKE", "0") == "1"  # v2: production (full 5-fold, 504k train)
if SMOKE:
    N_FOLDS = 2


def log(msg: str) -> None:
    mem = ""
    if psutil is not None:
        vm = psutil.virtual_memory()
        mem = f" [mem={vm.used//(1024**3)}GB/{vm.total//(1024**3)}GB]"
    print(f"[{time.strftime('%H:%M:%S')}]{mem} {msg}", flush=True)


def _find_one(pattern: str) -> Path:
    for p in KAGGLE_INPUT.rglob(pattern):
        return p
    raise FileNotFoundError(f"no match for {pattern} under {KAGGLE_INPUT}")


# ========================= inlined FE functions =========================
_LOGIT_COEFS = {
    "Low":    dict(bias=16.3173, soil_lt_25=-11.0237, temp_gt_30=-5.8559,
                   rain_lt_300=-10.8500, wind_gt_10=-5.8284,
                   stage=dict(Flowering=-5.4155, Harvest=5.5073,
                              Sowing=5.2299, Vegetative=-5.4617),
                   mulch=dict(No=-3.0014, Yes=2.8613)),
    "Medium": dict(bias=4.6524, soil_lt_25=0.3290, temp_gt_30=-0.0204,
                   rain_lt_300=0.1542, wind_gt_10=0.0841,
                   stage=dict(Flowering=0.3586, Harvest=-0.1348,
                              Sowing=-0.3547, Vegetative=0.3334),
                   mulch=dict(No=0.1883, Yes=0.0142)),
    "High":   dict(bias=-20.9697, soil_lt_25=10.6947, temp_gt_30=5.8763,
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
    soil, temp, rain, wind = (df[c].values for c in
                              ("soil_lt_25", "temp_gt_30", "rain_lt_300", "wind_gt_10"))
    cols = []
    for cls, coefs in _LOGIT_COEFS.items():
        logit = (coefs["bias"] + coefs["soil_lt_25"] * soil
                 + coefs["temp_gt_30"] * temp + coefs["rain_lt_300"] * rain
                 + coefs["wind_gt_10"] * wind)
        stage_vals = np.array([coefs["stage"].get(s, 0.0) for s in stage])
        mulch_vals = np.array([coefs["mulch"].get(m, 0.0) for m in mulch])
        name = f"logit_P_{cls}"
        df[name] = (logit + stage_vals + mulch_vals).astype(np.float32)
        cols.append(name)
    return cols


def add_quantile_bins(train, test, orig, nums, n_bins=N_BINS):
    """Bin each numeric to n_bins quantile buckets; fit on combined
    train+test+orig so codes are consistent."""
    new_cols = []
    for c in nums:
        combined = pd.concat([train[c], test[c], orig[c]], ignore_index=True)
        binned = pd.qcut(combined, q=n_bins, duplicates="drop", labels=False)
        binned = binned.fillna(0).astype(np.int16).to_numpy()
        s = len(train); t = s + len(test)
        name = f"BIN_{c}"
        train[name] = binned[:s]
        test[name] = binned[s:t]
        orig[name] = binned[t:]
        new_cols.append(name)
    return new_cols


def add_cat_pair_combos(train, test, orig, cats):
    new_cols = []
    for c1, c2 in combinations(cats, 2):
        col = f"COMBO_{c1}_{c2}"
        for df in (train, test, orig):
            df[col] = df[c1].astype(str) + "_" + df[c2].astype(str)
        combined = pd.concat([train[col], test[col], orig[col]])
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[col] = codes[:s]
        test[col] = codes[s:t]
        orig[col] = codes[t:]
        new_cols.append(col)
    return new_cols


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


def add_freq_features(train, test, orig, cats):
    new_cols = []
    for c in cats:
        freq = pd.concat([train[c], test[c], orig[c]]).value_counts(normalize=True)
        name = f"FREQ_{c}"
        for df in (train, test, orig):
            df[name] = df[c].map(freq).fillna(0).astype(np.float32)
        new_cols.append(name)
    return new_cols


def add_orig_mean_std(train, test, orig, cols_to_aggregate, target):
    new_cols = []
    for c in cols_to_aggregate:
        stats = orig.groupby(c)[target].agg(["mean", "std"]).reset_index()
        stats.columns = [c, f"ORIG_{c}_mean", f"ORIG_{c}_std"]
        for df_name in ("train", "test"):
            df = {"train": train, "test": test}[df_name]
            merged = df.merge(stats, on=c, how="left")
            df[f"ORIG_{c}_mean"] = merged[f"ORIG_{c}_mean"].fillna(0.5).astype(np.float32).values
            df[f"ORIG_{c}_std"]  = merged[f"ORIG_{c}_std"].fillna(0).astype(np.float32).values
        new_cols += [f"ORIG_{c}_mean", f"ORIG_{c}_std"]
    return new_cols


def add_num_as_cat(train, test, orig, nums):
    new_cols = []
    for c in nums:
        name = f"CAT_{c}"
        for df in (train, test, orig):
            df[name] = df[c].astype(str)
        combined = pd.concat([train[name], test[name], orig[name]])
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[name] = codes[:s]
        test[name] = codes[s:t]
        orig[name] = codes[t:]
        new_cols.append(name)
    return new_cols


# ========================= inlined OrderedTE =========================
class OrderedTE:
    def __init__(self, a=1.0):
        self.a = float(a)
        self.classes_ = None
        self.prior_ = None
        self.stats_ = {}
        self.cols_ = []

    def fit(self, df, cat_cols, target):
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
                stats_list,
            )
        te_df = pd.DataFrame(te_cols_out, index=df.index)
        return pd.concat([df, te_df], axis=1)

    def transform(self, df):
        te_cols_out = {}
        for c in self.cols_:
            stats = self.stats_[c]
            merged = df[[c]].merge(stats, on=c, how="left")
            for k, cls in enumerate(self.classes_):
                n_col = f"{c}_n_{cls}"; s_col = f"{c}_s_{cls}"
                prior = self.prior_[k]
                n = merged[n_col].fillna(0).to_numpy()
                s = merged[s_col].fillna(0).to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    te = np.where(n > 0, (s + self.a * prior) / (n + self.a), prior)
                te_cols_out[f"{c}_TE_cls{cls}"] = te.astype(np.float32)
        te_df = pd.DataFrame(te_cols_out, index=df.index)
        return pd.concat([df, te_df], axis=1)

    def te_col_names(self):
        return [f"{c}_TE_cls{cls}" for c in self.cols_ for cls in self.classes_]


# ========================= inlined log-bias tuner =========================
def tune_log_bias(oof, y, prior):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid_default = np.linspace(-3.0, 3.0, 61)
    grid_high = np.linspace(-3.0, 6.0, 91)  # High class needs wide grid (~+3.4)
    for _ in range(25):
        improved = False
        for k in range(3):
            grid = grid_high if k == 2 else grid_default
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
    return bias, float(best)


# ========================= data loading =========================
def load_and_engineer():
    log("listing /kaggle/input/ CSVs")
    for p in sorted(KAGGLE_INPUT.rglob("*.csv")):
        log(f"  {p}")
    train_path = _find_one("train.csv")
    test_path = _find_one("test.csv")
    orig_path = None
    for pattern in ("irrigation_prediction.csv", "Irrigation_Prediction.csv",
                    "irrigation-prediction.csv"):
        try:
            orig_path = _find_one(pattern); break
        except FileNotFoundError:
            continue
    if orig_path is None:
        for p in KAGGLE_INPUT.rglob("*.csv"):
            if p.name not in ("train.csv", "test.csv", "sample_submission.csv"):
                orig_path = p; break
    if orig_path is None:
        raise FileNotFoundError("no original-dataset CSV found")
    log(f"  train: {train_path}")
    log(f"  test:  {test_path}")
    log(f"  orig:  {orig_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    orig = pd.read_csv(orig_path)

    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    if SMOKE:
        log("SMOKE=1 — subsampling 20k train, 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test = test.sample(10_000, random_state=SEED).reset_index(drop=True)
        test_ids = test_ids[:10_000]

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)}  cats={len(cats)}  "
        f"train={len(train)}  test={len(test)}  orig={len(orig)}")

    log("adding threshold flags + LR-formula logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)

    log(f"adding quantile-bin numeric cats (N_BINS={N_BINS})")
    bin_cols = add_quantile_bins(train, test, orig, nums, n_bins=N_BINS)
    pair_keys = cats + bin_cols  # C(19,2)=171 pairs
    log(f"adding all-pair combos over (cats + bins) = {len(list(combinations(pair_keys,2)))} pairs")
    combos = add_cat_pair_combos(train, test, orig, pair_keys)
    log(f"  built {len(combos)} pair combos")

    log("adding digit features")
    digits = add_digit_features(train, test, orig, nums)
    log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)
    log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)
    log("adding ORIG mean/std per col")
    orig_stats_cols = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    # Factorize raw cats AFTER all FE that needs string values is done.
    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]

    info = dict(
        nums=nums, cats=cats, bins=bin_cols, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats_cols,
        te_cols=cats + bin_cols + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: cats={len(cats)} bins={len(bin_cols)} "
        f"combos={len(combos)} digits={len(digits)} num_as_cat={len(num_as_cat)} "
        f"tres={len(tres)} logits={len(logits)} freq={len(freq)} "
        f"orig_stats={len(orig_stats_cols)} te_cols={len(info['te_cols'])}")

    # Free orig — not needed after FE. Saves ~10-20MB but also loosens peak.
    del orig; gc.collect()
    log("freed orig DataFrame after FE")
    return train, test, info, test_ids


# ========================= per-fold training with aggressive cleanup =======
def run_cv(train, test, info, a_ote=1.0):
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob",
        tree_method="hist",
        device="cuda",  # ← GPU
        eval_metric="mlogloss",
        enable_categorical=False,
        n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200,
        verbosity=0,
    )
    log(f"xgb_params: {xgb_params}")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log("  fitting OrderedTE")
        t0 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=a_ote)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=info["te_cols"], target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        del X_tr_shuf; gc.collect()
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s "
            f"({len(te.te_col_names())} OTE cols)")

        feat_cols = numeric_feats + te.te_col_names()
        sw = compute_sample_weight("balanced", y[tr_idx]).astype(np.float32)

        log(f"  training XGB on GPU, {len(feat_cols)} features, "
            f"{len(X_tr)} train / {len(X_va)} val")
        t0 = time.time()
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(
            X_tr[feat_cols].to_numpy(dtype=np.float32), y[tr_idx],
            sample_weight=sw,
            eval_set=[(X_va[feat_cols].to_numpy(dtype=np.float32), y[va_idx])],
            verbose=500,
        )
        oof[va_idx] = model.predict_proba(
            X_va[feat_cols].to_numpy(dtype=np.float32)).astype(np.float32)
        test_pred += model.predict_proba(
            X_te[feat_cols].to_numpy(dtype=np.float32)).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
            f"best_iter={model.best_iteration}  wall={time.time()-t0:.1f}s")

        # Critical: release fold-local memory so next fold's OTE-fit doesn't OOM.
        del X_tr, X_va, X_te, te, sw, model, feat_cols
        gc.collect()

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols_count=len(feat_cols) if 'feat_cols' in dir() else 0)


# ========================= main orchestrator =========================
def main():
    t_start = time.time()
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    suffix = "_smoke" if SMOKE else ""
    oof_path = OUT_DIR / f"oof_recipe_171pair{suffix}.npy"
    test_path = OUT_DIR / f"test_recipe_171pair{suffix}.npy"
    np.save(oof_path, result["oof"])
    np.save(test_path, result["test"])
    log(f"wrote {oof_path} + {test_path}")

    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub_path = OUT_DIR / f"submission_recipe_171pair{suffix}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}  shape={sub.shape}")
    log(f"  pred dist: {dict(sub[TARGET].value_counts())}")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, n_bins=N_BINS, smoke=SMOKE,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        feature_group_sizes={k: len(v) if isinstance(v, list) else v
                             for k, v in info.items() if k != "te_cols"},
        te_col_count=len(info["te_cols"]),
        total_wall_seconds=time.time() - t_start,
    )
    with open(OUT_DIR / f"recipe_171pair_results{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote recipe_171pair_results{suffix}.json "
        f"(total wall {(time.time()-t_start)/60:.1f} min)")


if __name__ == "__main__":
    main()
