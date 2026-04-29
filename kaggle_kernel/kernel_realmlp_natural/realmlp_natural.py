"""Natural-cal RealMLP probe — first NN test of the calibration finding.

Mirrors kernel_realmlp/realmlp_pytabkit.py (LB +0.00003 contribution in
3-stack at n_ens=1) but applies the 2026-04-29 RF natural-cal recipe:
  - ORIG augmentation: concat 10k rule-perfect rows into per-fold train
    pool with sample_weight = ORIG_ROW_WEIGHT (default 0.5). Same trick
    rawashishsin uses to anchor calibration.
  - TargetEncoder cv=5 (was cv=2): rawashishsin's structural FE choice.
    sklearn's CV-shuffled smoothing replaces the role L1/L2 reg plays in
    XGB; for NNs, it just produces stabler per-fold encodings.
  - No class-balanced sample weight (already absent in baseline RealMLP;
    explicit here for clarity).
  - n_ens=1, n_epochs=40 retained — proven LB-positive base config.

Hypothesis (per CLAUDE.md 2026-04-29 calibration analysis): every prior
NN null exhibited magnitude trap (errs +5-30% over anchor) AT LEAST
partly because class-balanced training over-pushes High predictions at
depth-limited capacity. A naturally-calibrated NN (no upweight + ORIG
anchor + smoothed TE) MAY produce errs ≤ 1.05× anchor for the first
time — clearing the gate that's blocked every NN. Bayesian prior: ~25-30%.

Diagnostic:
  - Tuned log-bias drift from -log(prior). Prior drift was [0.70, 0.50,
    0.00] for RealMLP n_ens=1 baseline. Natural-cal target: |drift| ≤ 0.3
    each class.
  - Standalone tuned OOF (gate ≥ 0.974 for blend-leg viability).
  - Errors at recipe bias vs 4-stack anchor (gate ≤ 1.05× = 9886).

Outputs in /kaggle/working/:
  oof_realmlp_natural.npy
  test_realmlp_natural.npy
  realmlp_natural_results.json

ETA: 35-55 min on P100 GPU. Same wall-time safety nets as baseline.
SMOKE=1 env var for 2-fold 20k smoke (~5 min).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

# ========================= environment setup =========================
def _gpu_arch():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, timeout=10,
        ).strip().splitlines()
        return [x.strip() for x in out if x.strip()]
    except Exception as e:
        print(f"[boot] nvidia-smi error: {e}", flush=True)
        return []

_arches = _gpu_arch()
print(f"[boot] gpu compute_cap = {_arches}", flush=True)
if any(a in ("6.0", "6.1") for a in _arches):
    print("[boot] sm_60/61 detected - pinning torch+torchvision cu121", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--upgrade", "--force-reinstall", "--no-deps",
        "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "pytabkit", "lightning",
])
import pytabkit as _pt
print(f"[boot] pytabkit {getattr(_pt, '__version__', 'unknown')}", flush=True)

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from pytabkit import RealMLP_TD_Classifier


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

# Natural-cal knobs (the actual experimental change vs baseline)
ORIG_ROW_WEIGHT = 0.5  # rawashishsin parity
TE_CV = 5              # was 2; sklearn's CV-shuffled smoothing

KAGGLE_INPUT = Path("/kaggle/input")
_DEFAULT_OUT = Path("/kaggle/working")
if _DEFAULT_OUT.exists() or _DEFAULT_OUT.parent.exists():
    OUT_DIR = _DEFAULT_OUT
else:
    OUT_DIR = Path(os.environ.get("SMOKE_OUTPUT_DIR", "./realmlp_natural_local_out"))
OUT_DIR.mkdir(exist_ok=True, parents=True)

IS_SMOKE = False
SMOKE = IS_SMOKE or os.environ.get("SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2


def _find_one(pattern: str) -> Path:
    for p in KAGGLE_INPUT.rglob(pattern):
        return p
    raise FileNotFoundError(f"no match for {pattern} under {KAGGLE_INPUT}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ========================= data + features =========================
def load_data():
    log("loading train / test / orig")
    train = pd.read_csv(_find_one("train.csv"))
    test = pd.read_csv(_find_one("test.csv"))
    orig = pd.read_csv(_find_one("irrigation_prediction.csv"))
    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    for df in (train, test):
        df.drop(columns=["id"], inplace=True, errors="ignore")
    if SMOKE:
        log("SMOKE=1 — subsampling")
        train = train.sample(20_000, random_state=SEED).reset_index(drop=True)
        test_sub = test.sample(10_000, random_state=SEED)
        test_ids = test_ids[test_sub.index.to_numpy()]
        test = test_sub.reset_index(drop=True)
        orig = orig.sample(2_000, random_state=SEED).reset_index(drop=True)
    # Natural-cal parity via subsampling: pytabkit's RealMLP_TD_Classifier.fit()
    # doesn't support sample_weight= (verified in smoke v1). Subsample orig to
    # ORIG_ROW_WEIGHT fraction so 5000 unit-weight rows replicate the gradient
    # contribution of 10000 × 0.5-weight rows in expectation.
    n_orig_keep = max(1, int(round(len(orig) * ORIG_ROW_WEIGHT)))
    orig = orig.sample(n=n_orig_keep, random_state=SEED).reset_index(drop=True)
    log(f"  train={len(train):,}  test={len(test):,}  "
        f"orig={len(orig):,} (subsampled @ ORIG_ROW_WEIGHT={ORIG_ROW_WEIGHT})")
    return train, test, orig, test_ids


def build_features(train, test, orig):
    NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon",
            "Electrical_Conductivity", "Temperature_C", "Humidity",
            "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
            "Field_Area_hectare", "Previous_Irrigation_mm"]
    CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

    for c in NUMS:
        train[f"{c}_raw"] = train[c].astype(np.float32)
        test[f"{c}_raw"]  = test[c].astype(np.float32)
        orig[f"{c}_raw"]  = orig[c].astype(np.float32)

    combined = pd.concat([train, test, orig])
    for c in CATS + NUMS:
        combined[c], _ = combined[c].factorize()
    s1 = len(train); s2 = s1 + len(test)
    train = combined.iloc[:s1].copy()
    test = combined.iloc[s1:s2].drop(columns=[TARGET]).copy()
    orig = combined.iloc[s2:].copy()

    PAIR_SRC = ["Soil_Moisture", "Crop_Growth_Stage", "Temperature_C",
                "Mulching_Used", "Wind_Speed_kmh", "Rainfall_mm"]
    combo_cols: list[str] = []
    log(f"  building {len(list(combinations(PAIR_SRC, 2)))} pair combos")
    for c1, c2 in combinations(PAIR_SRC, 2):
        name = f"{c1}_x_{c2}"
        tr = train[c1].astype(str) + "_" + train[c2].astype(str)
        te = test[c1].astype(str)  + "_" + test[c2].astype(str)
        og = orig[c1].astype(str)  + "_" + orig[c2].astype(str)
        combined_pair = pd.concat([tr, te, og], ignore_index=True)
        codes, _ = pd.factorize(combined_pair)
        if pd.Series(codes).nunique() > len(codes) // 2:
            log(f"  skipped {name} (nunique > N/2)")
            continue
        train[name] = codes[:s1]
        test[name] = codes[s1:s2]
        orig[name] = codes[s2:]
        combo_cols.append(name)
    log(f"  kept {len(combo_cols)} combo cols")

    return train, test, orig, NUMS, CATS, combo_cols


# ========================= training loop =========================
def run_cv(train, test, orig, NUMS, CATS, combo_cols) -> dict:
    y = train[TARGET].to_numpy()
    y_orig = orig[TARGET].to_numpy()
    n_train = len(train)

    base_cols = CATS + NUMS + [f"{c}_raw" for c in NUMS]
    te_cols = combo_cols

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((n_train, 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_ba = []
    folds_completed = 0
    t_start = time.time()
    FOLD1_KILL_SEC = 22 * 60 if not SMOKE else 10 * 60
    TOTAL_KILL_SEC = 55 * 60 if not SMOKE else 15 * 60

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)
        X_or = orig.copy().reset_index(drop=True)
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Per-fold multiclass TargetEncoder cv=5 — natural-cal mechanism.
        # Fit on synth train ONLY (orig labels are rule-perfect; including
        # them in TE fit would bias encoded probs toward the rule rather
        # than the synth-side flip distribution).
        log(f"  fitting TargetEncoder(cv={TE_CV}) on {len(te_cols)} combos")
        te = TargetEncoder(target_type="multiclass", cv=TE_CV,
                           random_state=SEED)
        te_tr = te.fit_transform(X_tr[te_cols], y_tr)
        te_va = te.transform(X_va[te_cols])
        te_te = te.transform(X_te[te_cols])
        te_or = te.transform(X_or[te_cols])

        n_te = te_tr.shape[1]
        te_names = [f"te_{i}" for i in range(n_te)]
        def _frame(base_df, te_arr):
            return pd.concat([
                base_df[base_cols].reset_index(drop=True),
                pd.DataFrame(te_arr, columns=te_names),
            ], axis=1)
        tr_frame = _frame(X_tr, te_tr)
        or_frame = _frame(X_or, te_or)
        va_frame = _frame(X_va, te_va)
        te_frame = _frame(X_te, te_te)

        # Concat ORIG into training pool (rawashishsin pattern). orig was
        # already subsampled to ORIG_ROW_WEIGHT fraction in load_data() so
        # all rows enter at unit weight (= equivalent to original count
        # × ORIG_ROW_WEIGHT in expectation).
        train_frame = pd.concat([tr_frame, or_frame], axis=0,
                                ignore_index=True)
        y_train = np.concatenate([y_tr, y_orig])
        log(f"  combined train: {len(train_frame):,} (synth {len(tr_idx):,} "
            f"+ orig {len(or_frame):,}; orig pre-subsampled @ "
            f"ORIG_ROW_WEIGHT={ORIG_ROW_WEIGHT})")

        cat_like = CATS + NUMS
        for frame in (train_frame, va_frame, te_frame):
            for c in cat_like:
                frame[c] = frame[c].astype("category")

        log(f"  fitting RealMLP_TD on {train_frame.shape[1]} features")
        t0 = time.time()
        model = RealMLP_TD_Classifier(
            n_cv=1, n_refit=0, n_ens=1,
            device="cuda",
            val_metric_name="class_error",
            n_epochs=(3 if SMOKE else 40),
            random_state=SEED,
            verbosity=1,
        )
        model.fit(train_frame, y_train, val_idxs=None,
                  cat_col_names=cat_like)
        log(f"  RealMLP fit in {time.time() - t0:.1f}s")

        proba_va = model.predict_proba(va_frame)
        proba_te = model.predict_proba(te_frame)
        oof[va_idx] = proba_va.astype(np.float32)
        test_pred += proba_te.astype(np.float32) / N_FOLDS

        pred_va = proba_va.argmax(axis=1)
        ba = balanced_accuracy_score(y_va, pred_va)
        fold_ba.append(float(ba))
        folds_completed = fold
        elapsed = time.time() - t_start
        log(f"  fold {fold} argmax bal_acc = {ba:.5f}  "
            f"(elapsed {elapsed/60:.1f}m)")

        if fold == 1 and elapsed > FOLD1_KILL_SEC:
            log(f"!! FOLD-1 WALL-TIME KILL: {elapsed/60:.1f}m")
            break
        if elapsed > TOTAL_KILL_SEC:
            log(f"!! TOTAL WALL-TIME KILL: {elapsed/60:.1f}m  "
                f"({fold}/{N_FOLDS} folds done)")
            break

    if folds_completed and folds_completed < N_FOLDS:
        test_pred *= N_FOLDS / folds_completed

    nonzero_mask = oof.sum(axis=1) > 0
    overall_argmax = (
        balanced_accuracy_score(y[nonzero_mask],
                                oof[nonzero_mask].argmax(axis=1))
        if nonzero_mask.sum() > 0 else 0.0
    )
    log(f"=== OOF argmax = {overall_argmax:.5f} "
        f"({folds_completed}/{N_FOLDS} folds, mean fold "
        f"{np.mean(fold_ba) if fold_ba else 0:.5f})")
    return dict(
        oof=oof, test=test_pred,
        overall_argmax=float(overall_argmax),
        fold_ba=fold_ba,
        folds_completed=folds_completed,
    )


# ========================= log-bias tune =========================
def tune_log_bias(oof, y, prior, eps=1e-9):
    log_oof = np.log(np.clip(oof, eps, 1.0))
    bias = -np.log(prior)
    cc = np.bincount(y, minlength=3)

    def score(b):
        pred = (log_oof + b).argmax(1)
        per_cls = np.zeros(3)
        for k in range(3):
            per_cls[k] = ((pred == k) & (y == k)).sum() / max(cc[k], 1)
        return float(per_cls.mean())

    best = score(bias)
    grid_default = np.linspace(-3.0, 3.0, 61)
    grid_high = np.linspace(-3.0, 6.0, 91)
    for _ in range(25):
        improved = False
        for k in range(3):
            grid = grid_high if k == 2 else grid_default
            scores = []
            base = bias.copy()
            for g in grid:
                base[k] = bias[k] + g
                scores.append(score(base))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


# ========================= main =========================
def main():
    train, test, orig, test_ids = load_data()
    train, test, orig, NUMS, CATS, combo_cols = build_features(
        train, test, orig,
    )
    y = train[TARGET].to_numpy().astype(np.int64)

    result = run_cv(train, test, orig, NUMS, CATS, combo_cols)
    oof = result["oof"]
    test_pred = result["test"]
    folds_completed = result["folds_completed"]

    prior = np.bincount(y, minlength=3) / len(y)
    nonzero_mask = oof.sum(axis=1) > 0
    if nonzero_mask.sum() > 0:
        bias, tuned = tune_log_bias(oof[nonzero_mask], y[nonzero_mask], prior)
    else:
        bias, tuned = -np.log(prior), 0.0

    # Natural-cal diagnostic: drift from -log(prior).
    drift = bias - (-np.log(prior))
    drift_max = float(np.abs(drift).max())
    natcal_pass = drift_max <= 0.3
    log(f"tuned OOF bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")
    log(f"  -log(prior) = {(-np.log(prior)).round(4).tolist()}")
    log(f"  drift       = {drift.round(4).tolist()}  max|drift|={drift_max:.4f}")
    log(f"  natural-cal verdict: {'PASS' if natcal_pass else 'FAIL'} "
        f"(threshold |drift|<=0.3 each class)")

    np.save(OUT_DIR / "oof_realmlp_natural.npy", oof)
    np.save(OUT_DIR / "test_realmlp_natural.npy", test_pred)
    log(f"wrote oof_realmlp_natural.npy + test_realmlp_natural.npy")

    pred = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in pred]})
    sub_path = OUT_DIR / "submission_realmlp_natural_tuned.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")

    json_out = {
        "overall_argmax_bal_acc": result["overall_argmax"],
        "tuned_log_bias_bal_acc": tuned,
        "log_bias": bias.tolist(),
        "drift_from_neg_log_prior": drift.tolist(),
        "max_abs_drift": drift_max,
        "natural_cal_pass": natcal_pass,
        "fold_ba": result["fold_ba"],
        "n_folds": N_FOLDS,
        "folds_completed": folds_completed,
        "seed": SEED,
        "smoke": bool(SMOKE),
        "natcal_knobs": {
            "ORIG_ROW_WEIGHT": ORIG_ROW_WEIGHT,
            "TE_CV": TE_CV,
            "n_ens": 1,
            "n_epochs": 3 if SMOKE else 40,
        },
    }
    (OUT_DIR / "realmlp_natural_results.json").write_text(
        json.dumps(json_out, indent=2)
    )
    log(f"wrote realmlp_natural_results.json")


if __name__ == "__main__":
    main()
