"""GPU CatBoost on the recipe feature set (~440 cols).

CPU run of this same recipe + HPs:
  standalone tuned OOF 0.97936 (recipe_full_te XGB: 0.97967)
  hit 2000-iter cap every fold with bestIteration=1999 (no early stop)
  mean fold argmax 0.97806 (recipe XGB: 0.97589) — CatBoost +0.00217 at argmax
  Jaccard vs recipe XGB = 0.8060 (low orthogonality, promising on paper)
  but blend vs recipe XGB peaked at Δ=+0.00001 on fixed recipe bias

Hypothesis: the iter-cap was a bottleneck. GPU with 5-10x speedup lets us
push to 16000 iter + proper early stopping. Additional convergence may:
  - tighten prob scale → better bias compatibility with recipe XGB
  - push standalone OOF to 0.9795+
  - unlock the blend lift the 0.806 Jaccard is promising

GPU HPs:
  task_type='GPU', devices='0:1'        # use whichever GPU is allocated
  depth=4                               # same as CPU
  l2_leaf_reg=10                        # same as CPU
  iterations=16000                      # 8x CPU cap
  od_type='Iter', od_wait=500           # generous early stopping
  bootstrap_type='Bayesian'             # GPU default (Bernoulli is CPU-only)
  loss_function='MultiClass'
  lr=0.1                                # match CPU

Outputs (written to /kaggle/working/):
  oof_recipe_catboost_v2_gpu.npy
  test_recipe_catboost_v2_gpu.npy
  recipe_catboost_v2_gpu_results.json
  submission_recipe_catboost_v2_gpu_tuned.csv
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from functools import reduce
from itertools import combinations
from pathlib import Path

# ========================= environment setup =========================
# Ensure CatBoost is available with GPU support.
try:
    import catboost as _cb  # noqa
    print(f"[boot] catboost {_cb.__version__}", flush=True)
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])
    import catboost as _cb  # noqa

# Log GPU availability via nvidia-smi.
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"],
        text=True, timeout=10,
    ).strip()
    print(f"[boot] GPU info: {out}", flush=True)
except Exception as e:
    print(f"[boot] nvidia-smi error: {e}", flush=True)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

# Kaggle kernel paths are discovered via rglob (mount paths differ by
# competition/dataset slug and sometimes change silently across kernels).
KAGGLE_INPUT = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(exist_ok=True, parents=True)


def _find_one(pattern: str) -> Path:
    for p in KAGGLE_INPUT.rglob(pattern):
        return p
    raise FileNotFoundError(f"no match for {pattern} under {KAGGLE_INPUT}")

# Toggle SMOKE manually before push: True for 2-fold smoke (fast), False for prod
SMOKE = False
if SMOKE:
    N_FOLDS = 2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================= inlined recipe features =========================
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


def add_cat_pair_combos(train, test, orig, cats):
    new_cols = []
    for c1, c2 in combinations(cats, 2):
        col = f"COMBO_{c1}_{c2}"
        for df in (train, test, orig):
            df[col] = df[c1].astype(str) + "_" + df[c2].astype(str)
        combined = pd.concat([train[col], test[col], orig[col]])
        codes, _ = pd.factorize(combined)
        split_tr = len(train)
        split_te = split_tr + len(test)
        train[col] = codes[:split_tr]
        test[col] = codes[split_tr:split_te]
        orig[col] = codes[split_te:]
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
                n_col = f"{c}_n_{cls}"
                s_col = f"{c}_s_{cls}"
                prior = self.prior_[k]
                n = merged[n_col].fillna(0).to_numpy()
                s = merged[s_col].fillna(0).to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    te = np.where(
                        n > 0,
                        (s + self.a * prior) / (n + self.a),
                        prior,
                    )
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
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(
                    balanced_accuracy_score(y, (log_oof + base).argmax(axis=1))
                )
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
    log("listing /kaggle/input/")
    for p in sorted(KAGGLE_INPUT.rglob("*.csv")):
        log(f"  {p}")
    log("loading train / test / orig via rglob")
    train_path = _find_one("train.csv")
    test_path = _find_one("test.csv")
    # Orig dataset CSV — try common names.
    orig_path = None
    for pattern in ("irrigation_prediction.csv", "Irrigation_Prediction.csv",
                    "irrigation-prediction.csv"):
        try:
            orig_path = _find_one(pattern)
            break
        except FileNotFoundError:
            continue
    if orig_path is None:
        # Fall back to any non-train/test csv.
        for p in KAGGLE_INPUT.rglob("*.csv"):
            if p.name not in ("train.csv", "test.csv", "sample_submission.csv"):
                orig_path = p
                break
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

    log("adding cat x cat pair combos")
    combos = add_cat_pair_combos(train, test, orig, cats)
    log("adding digit features")
    digits = add_digit_features(train, test, orig, nums)
    log("adding num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)
    log("adding FREQ features")
    freq = add_freq_features(train, test, orig, cats + combos)
    log("adding ORIG mean/std per col")
    orig_stats = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    for c in cats:
        combined = pd.concat([train[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[c] = codes[:s]
        test[c] = codes[s:t]
        orig[c] = codes[t:]

    info = dict(
        nums=nums, cats=cats, combos=combos, digits=digits,
        num_as_cat=num_as_cat, freq=freq, tres=tres, logits=logits,
        orig_stats=orig_stats,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: "
        f"cats={len(cats)} combos={len(combos)} digits={len(digits)} "
        f"num_as_cat={len(num_as_cat)} tres={len(tres)} logits={len(logits)} "
        f"freq={len(freq)} orig_stats={len(orig_stats)} "
        f"te_cols={len(info['te_cols'])}")
    return train, test, info, test_ids


# ========================= CV loop (GPU) =========================
def run_cv(train, test, info, a_ote=1.0):
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"])
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    best_iters = []

    # v2 distinct HPs: depth=5 (existing v1 catboost variants all depth=4),
    # lower lr=0.04 (existing 0.1), heavier l2=15 (existing 10), Bayesian
    # bootstrap (matches catboost_recipe_gpu but distinct from
    # recipe_full_te_catboost which is Bernoulli), random_strength=2 for
    # more split randomness. Goal: ≥0.97 standalone OOF (meta-absorption
    # threshold) AND structurally different errors than existing 3 v1
    # CatBoost variants.
    cb_params = dict(
        iterations=1500 if SMOKE else 8000,
        depth=5,
        learning_rate=0.04,
        l2_leaf_reg=15.0,
        random_strength=2.0,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        random_seed=SEED,
        od_type="Iter",
        od_wait=100 if SMOKE else 300,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
        task_type="GPU",
        devices="0",
        verbose=False,
    )
    log(f"cb_params: iterations={cb_params['iterations']}, "
        f"depth={cb_params['depth']}, l2={cb_params['l2_leaf_reg']}, "
        f"od_wait={cb_params['od_wait']}")

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
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        sw = compute_sample_weight("balanced", y[tr_idx])

        log(f"  training CatBoost GPU on {len(feat_cols)} features")
        t_fit = time.time()
        model = CatBoostClassifier(**cb_params)
        model.fit(
            X_tr[feat_cols], y[tr_idx],
            sample_weight=sw,
            eval_set=(X_va[feat_cols], y[va_idx]),
            use_best_model=True,
            verbose=1000,
        )
        bi = int(model.tree_count_)
        best_iters.append(bi)
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold}  best_iter={bi}  argmax={bal:.5f}  "
            f"wall={time.time()-t_fit:.1f}s")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== overall OOF argmax = {overall:.5f}  "
        f"(fold mean {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall), feat_cols=feat_cols,
                best_iters=best_iters)


def main():
    train, test, info, test_ids = load_and_engineer()
    result = run_cv(train, test, info)

    y = train[TARGET].to_numpy()
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(OUT_DIR / "oof_recipe_catboost_v2_gpu.npy", result["oof"])
    np.save(OUT_DIR / "test_recipe_catboost_v2_gpu.npy", result["test"])

    eps = 1e-9
    test_log = np.log(np.clip(result["test"], eps, 1.0))
    test_pred_idx = (test_log + bias).argmax(1)
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_idx],
    })
    sub.to_csv(OUT_DIR / "submission_recipe_catboost_v2_gpu_tuned.csv", index=False)
    log("wrote submission_recipe_catboost_v2_gpu_tuned.csv")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS,
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        best_iters=result["best_iters"],
        overall_argmax_bal_acc=result["overall_argmax"],
        tuned_log_bias_bal_acc=tuned,
        log_bias=bias.tolist(),
        n_features=len(result["feat_cols"]),
        cb_params_iterations=(1500 if SMOKE else 8000),
        cb_params_depth=5,
        cb_params_l2_leaf_reg=15.0,
        cb_params_learning_rate=0.04,
        cb_params_random_strength=2.0,
        cb_params_od_wait=(100 if SMOKE else 500),
    )
    with open(OUT_DIR / "recipe_catboost_v2_gpu_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    log("wrote recipe_catboost_v2_gpu_results.json")


if __name__ == "__main__":
    main()
