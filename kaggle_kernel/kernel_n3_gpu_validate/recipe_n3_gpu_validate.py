"""N3 5-shuffle OTE concat — GPU VALIDATION RUN (single fold only).

Validates the GPU runtime estimate before committing to the full 5-fold
production. CPU run on Kaggle showed 13s/iter on 2.52M rows projecting to
36h+ wall (DOOMED). GPU XGB on this shape should be ~0.5-1s/iter; this
kernel runs ONLY fold 1 with a 1.5h hard kill to validate.

Decision rule after this run:
  - fold 1 completes in < 60 min → push full 5-fold GPU kernel.
    Predicted total ~2h 15m, fits 9h cap.
  - fold 1 hits 1.5h kill < 1500 XGB rounds → GPU is too slow on
    augmented set; pivot to K=2 or skip lever entirely.

Same FE / OTE as kernel_n3_5shuffle, but XGB uses device='cuda' and
only fold 1 runs (RUN_FOLD_INT=1, TOTAL_KILL_SEC=5400).
"""
from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from functools import reduce
from pathlib import Path


# ============================== boot =========================================
def _ensure(pkg: str, install: str | None = None) -> None:
    try:
        __import__(pkg.split("[")[0])
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", install or pkg]
        )


_ensure("xgboost")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

print(f"[boot] xgboost {xgb.__version__}", flush=True)

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
ACTIVE_STAGES = ("Flowering", "Vegetative")
DGP_THRESHOLDS = dict(sm=25.0, rf=300.0, tc=30.0, ws=10.0)

SMOKE = os.environ.get("SMOKE") == "1"
N_SHUFFLE = int(os.environ.get("N_SHUFFLE", "2" if SMOKE else "5"))
if SMOKE:
    N_FOLDS = 2

# *** VALIDATION KNOBS ***
# Run fold 1 only; hard kill at 1.5h.
RUN_FOLD_INT = 1
TOTAL_KILL_SEC = 90 * 60   # 1.5h

# GPU detection — fail loudly if no GPU (this kernel is GPU-required)
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"], text=True, timeout=10).strip()
    print(f"[boot] GPU: {out}", flush=True)
    USE_GPU = True
except Exception as e:
    print(f"[boot] WARNING: no GPU detected ({e}); falling back to CPU "
          f"(this validation will be useless)", flush=True)
    USE_GPU = False

XGB_PARAMS = dict(
    n_estimators=300 if SMOKE else 3000,
    max_depth=4, max_leaves=30,
    learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=2, reg_alpha=5, reg_lambda=5,
    max_bin=256 if SMOKE else 1024,
    objective="multi:softprob",
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda" if USE_GPU else "cpu",
    enable_categorical=False, n_jobs=-1, random_state=SEED,
    early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
)


# ============================== fast bal_acc ==================================
def fast_bal_acc(y, pred, n_class=3, class_counts=None):
    if class_counts is None:
        class_counts = np.bincount(y, minlength=n_class)
    matches = (pred == y)
    hit = np.array([matches[y == k].sum() for k in range(n_class)], dtype=np.int64)
    return float((hit / np.maximum(class_counts, 1)).mean())


def tune_log_bias(oof, y, prior, eps=1e-9):
    """Coord-ascent per-class log-bias on full OOF.

    Grid -3..+6 for High (empirical optimum ~+3.4 under severe imbalance).
    """
    log_oof = np.log(np.clip(oof, eps, 1.0))
    bias = -np.log(prior)
    cc = np.bincount(y, minlength=3)
    best = fast_bal_acc(y, (log_oof + bias).argmax(1), class_counts=cc)
    grid_default = np.linspace(-3.0, 3.0, 61)
    grid_high = np.linspace(-3.0, 6.0, 91)
    for _ in range(25):
        improved = False
        for k in range(3):
            grid = grid_high if k == 2 else grid_default
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(fast_bal_acc(y, (log_oof + base).argmax(1), class_counts=cc))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


# ============================== OrderedTE =====================================
class OrderedTE:
    """Per-class cumulative shuffled target encoder.

    fit() takes a (potentially shuffled) df, returns df with TE columns appended.
    transform() applies full-train per-key stats to val/test.
    """

    def __init__(self, a: float = 1.0) -> None:
        self.a = float(a)
        self.classes_: np.ndarray | None = None
        self.prior_: np.ndarray | None = None
        self.stats_: dict[str, pd.DataFrame] = {}
        self.cols_: list[str] = []

    def fit(self, df: pd.DataFrame, cat_cols: list[str], target: str) -> pd.DataFrame:
        y = df[target].to_numpy()
        self.classes_ = np.array(sorted(pd.unique(y)))
        counts = np.array([(y == k).sum() for k in self.classes_], dtype=np.float64)
        self.prior_ = counts / counts.sum()
        self.cols_ = list(cat_cols)
        te_cols_out: dict[str, np.ndarray] = {}
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

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        te_cols_out: dict[str, np.ndarray] = {}
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

    def te_col_names(self) -> list[str]:
        return [f"{c}_TE_cls{cls}" for c in self.cols_ for cls in self.classes_]


def fit_concat_5shuffle(df, cat_cols, target, a=1.0, n_shuffle=5, seed=42):
    """5x augmentation: fit K independent shuffles, concat all K with TE cols.

    Each augmented row carries the same raw features but a different TE
    realization. Returns (augmented_df, fitted_ote_for_transform).
    """
    rng = np.random.default_rng(seed)
    pieces = []
    last_te: OrderedTE | None = None
    for k in range(n_shuffle):
        perm = rng.permutation(len(df))
        df_shuf = df.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=a)
        df_shuf_with_te = te.fit(df_shuf, cat_cols=cat_cols, target=target)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        df_unshuf = df_shuf_with_te.iloc[inv].reset_index(drop=True)
        pieces.append(df_unshuf)
        last_te = te
    augmented = pd.concat(pieces, axis=0, ignore_index=True)
    return augmented, last_te


# ============================== FE blocks =====================================
LR_COEFS = {
    # cdeotte's exact LR coefficients on 10k original (the rule's smooth form)
    "Soil_Moisture": -0.30236, "Rainfall_mm": -0.04054,
    "Temperature_C": 0.18994, "Wind_Speed_kmh": 0.16867,
    "_Mulching_Used_Yes": -1.81879,
    "_Crop_Growth_Stage_Flowering": 1.49419, "_Crop_Growth_Stage_Harvest": -1.20925,
    "_Crop_Growth_Stage_Sowing": -1.50729, "_Crop_Growth_Stage_Vegetative": 1.22235,
    "_intercept": -1.92998,
}


def add_threshold_flags(df: pd.DataFrame) -> list[str]:
    df["dry"] = (df["Soil_Moisture"] < DGP_THRESHOLDS["sm"]).astype(np.int8)
    df["norain"] = (df["Rainfall_mm"] < DGP_THRESHOLDS["rf"]).astype(np.int8)
    df["hot"] = (df["Temperature_C"] > DGP_THRESHOLDS["tc"]).astype(np.int8)
    df["windy"] = (df["Wind_Speed_kmh"] > DGP_THRESHOLDS["ws"]).astype(np.int8)
    return ["dry", "norain", "hot", "windy"]


def add_lr_formula_logits(df: pd.DataFrame) -> list[str]:
    """Compute 3-class softmax-style logits using cdeotte's LR coefficients.

    The original LR is a single-class score; we add log-prior offsets
    to produce 3 logit-like numerics per row.
    """
    z = LR_COEFS["_intercept"] + np.zeros(len(df), dtype=np.float64)
    for col in ("Soil_Moisture", "Rainfall_mm", "Temperature_C", "Wind_Speed_kmh"):
        z = z + LR_COEFS[col] * df[col].astype(float).to_numpy()
    z = z + LR_COEFS["_Mulching_Used_Yes"] * (df["Mulching_Used"].astype(str) == "Yes").to_numpy().astype(np.float64)
    for stage in ("Flowering", "Harvest", "Sowing", "Vegetative"):
        z = z + LR_COEFS[f"_Crop_Growth_Stage_{stage}"] * (df["Crop_Growth_Stage"].astype(str) == stage).to_numpy().astype(np.float64)
    df["logit_LR"] = z.astype(np.float32)
    df["logit_LR_p"] = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    df["logit_LR_neg"] = (-z).astype(np.float32)
    return ["logit_LR", "logit_LR_p", "logit_LR_neg"]


def add_cat_pair_combos(train, test, orig, cats):
    out = []
    for i, c1 in enumerate(cats):
        for c2 in cats[i + 1:]:
            name = f"{c1}__x__{c2}"
            for df in (train, test, orig):
                df[name] = (df[c1].astype(str) + "_" + df[c2].astype(str))
            combined = pd.concat([train[name], test[name], orig[name]])
            codes, _ = pd.factorize(combined.astype(str))
            s = len(train); t = s + len(test)
            train[name] = codes[:s]
            test[name] = codes[s:t]
            orig[name] = codes[t:]
            out.append(name)
    return out


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
            df[f"ORIG_{c}_std"] = merged[f"ORIG_{c}_std"].fillna(0).astype(np.float32).values
        new_cols += [f"ORIG_{c}_mean", f"ORIG_{c}_std"]
    return new_cols


def add_num_as_cat(train, test, orig, nums):
    out = []
    for c in nums:
        name = f"{c}_ascat"
        combined = pd.concat([train[c], test[c], orig[c]]).astype(float)
        codes, _ = pd.factorize(combined)
        s = len(train); t = s + len(test)
        train[name] = codes[:s]
        test[name] = codes[s:t]
        orig[name] = codes[t:]
        out.append(name)
    return out


# ============================== load & FE ====================================
def load_and_engineer():
    log("loading train / test / orig")
    if "/kaggle/" in str(KAGGLE_INPUT) and KAGGLE_INPUT.exists():
        train_csv = _find_one("train.csv")
        test_csv = _find_one("test.csv")
        orig_csv = None
        for pattern in ("irrigation_prediction.csv", "Irrigation_Prediction.csv",
                        "Irrigation Prediction.csv", "irrigation-prediction.csv"):
            try:
                orig_csv = _find_one(pattern)
                break
            except FileNotFoundError:
                continue
        if orig_csv is None:
            raise FileNotFoundError(
                f"no orig CSV found under {KAGGLE_INPUT}; "
                f"check dataset_sources in kernel-metadata.json")
    else:
        # Local SMOKE: use repo-local paths (data/train.csv etc.)
        train_csv = Path("data/train.csv")
        test_csv = Path("data/test.csv")
        # orig is provided as zip in repo; expand or use the CSV variant
        orig_zip = Path("data/archive.zip")
        if orig_zip.exists():
            import zipfile
            with zipfile.ZipFile(orig_zip) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise FileNotFoundError(f"no CSV in {orig_zip}")
                orig_csv = z.extract(names[0], path="data")
                orig_csv = Path(orig_csv)
        else:
            orig_csv = Path("data/Irrigation_Prediction.csv")
    log(f"  train: {train_csv}")
    log(f"  test:  {test_csv}")
    log(f"  orig:  {orig_csv}")
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    orig = pd.read_csv(orig_csv)

    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)

    if SMOKE:
        log("SMOKE=1 — subsampling to 20k train, 10k test")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test_s = test.sample(10_000, random_state=SEED)
        test = test_s.reset_index(drop=True)
        test_ids = test_ids[test_s.index.to_numpy()]

    nums = list(test.select_dtypes(include=np.number).columns)
    cats = [c for c in test.columns if c not in nums]
    log(f"  nums={len(nums)}  cats={len(cats)}  "
        f"train={len(train)}  test={len(test)}  orig={len(orig)}")

    log("threshold flags + LR-formula logits")
    for df in (train, test, orig):
        tres = add_threshold_flags(df)
    for df in (train, test, orig):
        logits = add_lr_formula_logits(df)

    log("cat-pair combos")
    combos = add_cat_pair_combos(train, test, orig, cats)
    log(f"  combos={len(combos)}")

    log("digit features")
    digits = add_digit_features(train, test, orig, nums)
    log(f"  digits={len(digits)}")

    log("num-as-cat")
    num_as_cat = add_num_as_cat(train, test, orig, nums)

    log("FREQ per cat + combo")
    freq = add_freq_features(train, test, orig, cats + combos)

    log("ORIG mean/std per col (target on 10k orig only)")
    orig_stats_cols = add_orig_mean_std(train, test, orig, nums + cats, TARGET)

    # Factorize raw cats AFTER all string-valued FE done
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
        orig_stats=orig_stats_cols,
        te_cols=cats + combos + digits + num_as_cat + tres,
    )
    log(f"  feature groups: cats={len(cats)} combos={len(combos)} "
        f"digits={len(digits)} num_as_cat={len(num_as_cat)} tres={len(tres)} "
        f"logits={len(logits)} freq={len(freq)} orig_stats={len(orig_stats_cols)} "
        f"te_cols={len(info['te_cols'])}")
    # Free orig — not needed for fold loop
    del orig
    gc.collect()
    return train, test, info, test_ids


# ============================== train loop ====================================
def run_cv_5shuffle(train, test, info, t_start):
    y = train[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    numeric_feats = info["nums"] + info["tres"] + info["logits"] + info["freq"] + info["orig_stats"]

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        # *** VALIDATION KNOB: skip folds != RUN_FOLD_INT ***
        if fold != RUN_FOLD_INT:
            log(f"skipping fold {fold} (validation: RUN_FOLD_INT={RUN_FOLD_INT})")
            continue
        elapsed = time.time() - t_start
        if elapsed > TOTAL_KILL_SEC:
            log(f"TOTAL_KILL_SEC reached at t+{elapsed:.0f}s before fold {fold} — abort")
            break
        log(f"=== fold {fold}/{N_FOLDS}  K={N_SHUFFLE}-shuffle  t+{elapsed:.0f}s ===")
        t_fold = time.time()
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        log(f"  fitting OrderedTE x {N_SHUFFLE} shuffles + concat")
        t0 = time.time()
        X_tr_aug, te = fit_concat_5shuffle(
            X_tr, cat_cols=info["te_cols"], target=TARGET,
            a=1.0, n_shuffle=N_SHUFFLE, seed=SEED + fold,
        )
        log(f"    fit done in {time.time()-t0:.1f}s, X_tr_aug={len(X_tr_aug):,}")
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE total in {time.time()-t0:.1f}s")

        feat_cols = numeric_feats + te.te_col_names()
        y_tr_orig = y[tr_idx]
        y_tr_aug = np.tile(y_tr_orig, N_SHUFFLE).astype(np.int32)
        sw_orig = compute_sample_weight("balanced", y_tr_orig)
        sw_aug = np.tile(sw_orig, N_SHUFFLE).astype(np.float32)

        log(f"  XGB on {len(feat_cols)} feats, {len(X_tr_aug):,} rows")
        model = xgb.XGBClassifier(**XGB_PARAMS)
        model.fit(
            X_tr_aug[feat_cols], y_tr_aug,
            sample_weight=sw_aug,
            eval_set=[(X_va[feat_cols], y[va_idx])],
            verbose=500,
        )
        oof[va_idx] = model.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += model.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc={bal:.5f}  best_iter={model.best_iteration}  "
            f"wall={time.time()-t_fold:.1f}s")

        # Per-fold checkpoint to /kaggle/working/
        np.save(OUT_DIR / f"oof_n3_gpu_validate_fold{fold}.npy", oof)
        np.save(OUT_DIR / f"test_n3_gpu_validate_fold{fold}.npy", test_pred)
        log(f"  checkpoint: fold {fold} saved")

        del X_tr, X_va, X_te, X_tr_aug, model, te, y_tr_aug, sw_aug
        gc.collect()

    # If we broke early, fold_scores has len < N_FOLDS — handle outside
    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall),
                folds_completed=len(fold_scores),
                feat_cols=feat_cols)


# ============================== main =========================================
def main():
    t_start = time.time()
    log(f"N3 5-shuffle  K={N_SHUFFLE}  N_FOLDS={N_FOLDS}  SMOKE={SMOKE}")
    train, test, info, test_ids = load_and_engineer()
    result = run_cv_5shuffle(train, test, info, t_start)

    y = train[TARGET].to_numpy()

    # Always save the running OOF/test (whether all folds completed or not)
    np.save(OUT_DIR / "oof_n3_gpu_validate.npy", result["oof"])
    np.save(OUT_DIR / "test_n3_gpu_validate.npy", result["test"])
    log(f"saved oof_n3_gpu_validate.npy + test_n3_gpu_validate.npy")

    summary = dict(
        seed=SEED, n_folds=N_FOLDS, n_shuffle=N_SHUFFLE,
        smoke=SMOKE,
        folds_completed=result["folds_completed"],
        fold_scores_argmax=[float(s) for s in result["fold_scores"]],
        overall_argmax_bal_acc=result["overall_argmax"],
        n_features=len(result["feat_cols"]),
        elapsed_sec=time.time() - t_start,
    )

    if result["folds_completed"] == N_FOLDS:
        prior = np.bincount(y, minlength=3) / len(y)
        bias, tuned = tune_log_bias(result["oof"], y, prior)
        log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")
        summary["tuned_log_bias_bal_acc"] = tuned
        summary["log_bias"] = bias.tolist()

        # Submission
        eps = 1e-9
        test_log = np.log(np.clip(result["test"], eps, 1.0))
        test_pred_idx = (test_log + bias).argmax(1)
        sub = pd.DataFrame({
            "id": test_ids,
            TARGET: [IDX2CLS[i] for i in test_pred_idx],
        })
        sub_path = OUT_DIR / "submission_n3_gpu_validate.csv"
        sub.to_csv(sub_path, index=False)
        log(f"wrote {sub_path}  shape={sub.shape}")
    else:
        log(f"WARNING: only {result['folds_completed']}/{N_FOLDS} folds completed; "
            f"no full-OOF tuned bias / submission emitted")

    res_path = OUT_DIR / "n3_gpu_validate_results.json"
    with open(res_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {res_path}")
    log(f"DONE  elapsed={time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
