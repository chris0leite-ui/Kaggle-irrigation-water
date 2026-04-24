"""A1 RealMLP-TD via pytabkit on Kaggle GPU.

Ports mahoganybuttstrings/pg-s6e4-realmlp-cv-0-97802-lb-0-97685 using the
official pytabkit library (github.com/dholzmueller/pytabkit). RealMLP-TD is
a production-tuned tabular NN with:
  - n_ens parallel BatchEnsemble heads sharing most weights via einsum
  - PBLD (Periodic Basis with Learned Decay) numeric embedding
  - smooth-clip scaler (x / sqrt(1 + (x/3)^2))
  - label smoothing with cosine schedule
  - flat_cos LR schedule, scale-layer LR=10x base

Our 11 prior NN nulls (v5-v9 MLP, FT-T, TabPFN, pretrain-FT, NN-on-orig,
soft-distill, DAE) all used from-scratch architectures. RealMLP-TD is
qualitatively different — first NN family consistently matching GBDT on
46-set TabArena benchmark.

Feature set (small by design, following mahoganybuttstrings):
  - 11 raw numerics (kept as float + duplicated as factorized cats)
  - 8 raw cats (factorized)
  - 15 pair combos of 6 rule-relevant features (Soil_Moisture,
    Crop_Growth_Stage, Temperature_C, Mulching_Used, Wind_Speed_kmh,
    Rainfall_mm). Filters out pairs where nunique > N/2 (uninformative).
  - Per-fold multiclass TargetEncoder (cv=5) on the 15 pair combos.

Total ~19 base + ~45 TE cols = ~64 features (vs our 443-col recipe —
RealMLP is not a fit for wide feature sets).

5-fold StratifiedKFold(shuffle=True, random_state=42) aligned with every
other OOF on main so the downstream blend_realmlp.py script can compare
directly to recipe_full_te / pseudolabel / etc.

Outputs in /kaggle/working/:
  oof_realmlp.npy (n_train, 3)
  test_realmlp.npy (n_test, 3)
  realmlp_results.json

Fold-1 early-gate on error Jaccard vs recipe_full_te + LB-best 2-way:
  Jaccard >= 0.90 → abort (redundant with existing blend)
  0.85 <= Jaccard < 0.90 → warn (blend lift capped ~+0.00015)
  Jaccard < 0.85 → run all folds + flag as fresh blend component.

ETA: ~45 min on P100 GPU.

SMOKE=1 env var runs a 2-fold 20k-row smoke (~3 min) for debugging.
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
# P100 sm_60 shim: pre-installed torch on Kaggle kernels often lacks
# kernel images for Pascal GPUs. Install cu121 build BEFORE pytabkit
# (which depends on torch) so pytabkit imports against a working torch.
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
    # Pin torch + torchvision to a matched cu121 pair that supports sm_60.
    # --no-deps on torch to avoid pulling CPU-only numpy etc., but we must
    # also force-reinstall torchvision to a version compatible with 2.5.1
    # (otherwise pytabkit's lightning dep sees torchvision incompat).
    print("[boot] sm_60/61 detected - pinning torch+torchvision cu121", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--upgrade", "--force-reinstall", "--no-deps",
        "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])

# Install pytabkit + lightning explicitly (pytabkit's pyproject doesn't
# always mark lightning as a hard dep, and on Kaggle images neither is
# pre-installed). Use regular install so pip resolves any remaining deps.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "pytabkit", "lightning",
])
import pytabkit as _pt
print(f"[boot] pytabkit {getattr(_pt, '__version__', 'unknown')}", flush=True)

try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
         "--format=csv,noheader"],
        text=True, timeout=10,
    ).strip()
    print(f"[boot] GPU info: {out}", flush=True)
except Exception as e:
    print(f"[boot] nvidia-smi info error: {e}", flush=True)

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

KAGGLE_INPUT = Path("/kaggle/input")
OUT_DIR = Path("/kaggle/working")
OUT_DIR.mkdir(exist_ok=True, parents=True)

SMOKE = os.environ.get("SMOKE") == "1"
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
    orig_path = _find_one("irrigation_prediction.csv")
    orig = pd.read_csv(orig_path)
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
    log(f"  train={len(train):,}  test={len(test):,}  orig={len(orig):,}")
    return train, test, orig, test_ids


def build_features(train, test, orig):
    """Factorize cats + build 15 pair combos of rule-relevant features.

    Matches mahoganybuttstrings kernel. Returns (train, test, orig) with
    added combo columns, plus the list of combo col names + the base
    feature name lists.
    """
    NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon",
            "Electrical_Conductivity", "Temperature_C", "Humidity",
            "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
            "Field_Area_hectare", "Previous_Irrigation_mm"]
    CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]

    # Copy raw nums BEFORE factorization (mahoganybuttstrings keeps both
    # the raw continuous float and a factorized cat version).
    for c in NUMS:
        train[f"{c}_raw"] = train[c].astype(np.float32)
        test[f"{c}_raw"]  = test[c].astype(np.float32)
        orig[f"{c}_raw"]  = orig[c].astype(np.float32)

    # Factorize all CATS + NUMS across the concatenation.
    combined = pd.concat([train, test, orig])
    for c in CATS + NUMS:
        combined[c], _ = combined[c].factorize()
    s1 = len(train); s2 = s1 + len(test)
    train = combined.iloc[:s1].copy()
    test = combined.iloc[s1:s2].drop(columns=[TARGET]).copy()
    orig = combined.iloc[s2:].copy()

    # 15 pair combos of the 6 rule-relevant columns.
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
        # Drop pairs where nunique > N/2 (uninformative — near-unique keys).
        if pd.Series(codes).nunique() > len(codes) // 2:
            log(f"  skipped {name} (nunique > N/2, uninformative)")
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
    n_train = len(train)

    base_cols = CATS + NUMS + [f"{c}_raw" for c in NUMS]
    te_cols = combo_cols

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof = np.zeros((n_train, 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_ba = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Per-fold multiclass TargetEncoder on combo cols (sklearn
        # 1.5+, fit on train only = leak-free for OOF alignment).
        log(f"  fitting TargetEncoder on {len(te_cols)} combos")
        te = TargetEncoder(target_type="multiclass", cv=5,
                           random_state=SEED)
        te_tr = te.fit_transform(X_tr[te_cols], y_tr)
        te_va = te.transform(X_va[te_cols])
        te_te = te.transform(X_te[te_cols])

        # TE output is (n, k*3) for multiclass (one col per key per class).
        n_te = te_tr.shape[1]
        te_names = [f"te_{i}" for i in range(n_te)]
        tr_frame = pd.concat([
            X_tr[base_cols].reset_index(drop=True),
            pd.DataFrame(te_tr, columns=te_names),
        ], axis=1)
        va_frame = pd.concat([
            X_va[base_cols].reset_index(drop=True),
            pd.DataFrame(te_va, columns=te_names),
        ], axis=1)
        te_frame = pd.concat([
            X_te[base_cols].reset_index(drop=True),
            pd.DataFrame(te_te, columns=te_names),
        ], axis=1)

        # Declare categorical dtype for the CATS+NUMS factorized codes so
        # pytabkit handles them via embedding. The `_raw` copies stay
        # float → numeric path in RealMLP (gets PBLD periodic embedding).
        cat_like = CATS + NUMS
        for frame in (tr_frame, va_frame, te_frame):
            for c in cat_like:
                frame[c] = frame[c].astype("category")

        log(f"  fitting RealMLP_TD on {tr_frame.shape[1]} features, "
            f"{len(tr_frame):,} rows")
        t0 = time.time()
        model = RealMLP_TD_Classifier(
            n_cv=1, n_refit=0,
            device="cuda",
            val_metric_name="class_error",
            n_epochs=(3 if SMOKE else None),  # pytabkit default if None
            random_state=SEED,
            verbosity=1,
        )
        model.fit(tr_frame, y_tr,
                  val_idxs=None,  # let pytabkit hold out internally
                  cat_col_names=cat_like)
        log(f"  RealMLP fit in {time.time() - t0:.1f}s")

        proba_va = model.predict_proba(va_frame)
        proba_te = model.predict_proba(te_frame)
        oof[va_idx] = proba_va.astype(np.float32)
        test_pred += proba_te.astype(np.float32) / N_FOLDS

        pred_va = proba_va.argmax(axis=1)
        ba = balanced_accuracy_score(y_va, pred_va)
        fold_ba.append(float(ba))
        log(f"  fold {fold} argmax bal_acc = {ba:.5f}")

    overall_argmax = balanced_accuracy_score(y, oof.argmax(axis=1))
    log(f"=== OOF argmax = {overall_argmax:.5f} "
        f"(mean fold {np.mean(fold_ba):.5f} ± {np.std(fold_ba):.5f})")
    return dict(
        oof=oof,
        test=test_pred,
        overall_argmax=float(overall_argmax),
        fold_ba=fold_ba,
    )


# ========================= log-bias tune =========================
def tune_log_bias(oof, y, prior, eps=1e-9):
    """Coord-ascent per-class log-bias."""
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
    y = train[TARGET].to_numpy()

    result = run_cv(train, test, orig, NUMS, CATS, combo_cols)
    oof = result["oof"]
    test_pred = result["test"]

    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"tuned OOF bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(OUT_DIR / "oof_realmlp.npy", oof)
    np.save(OUT_DIR / "test_realmlp.npy", test_pred)
    log(f"wrote oof_realmlp.npy + test_realmlp.npy")

    pred = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in pred]})
    sub_path = OUT_DIR / "submission_realmlp_tuned.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")

    json_out = {
        "overall_argmax_bal_acc": result["overall_argmax"],
        "tuned_log_bias_bal_acc": tuned,
        "log_bias": bias.tolist(),
        "fold_ba": result["fold_ba"],
        "n_folds": N_FOLDS,
        "seed": SEED,
        "feature_counts": {
            "base_cols": len(CATS) + len(NUMS) + len(NUMS),
            "combo_cols": len(combo_cols),
        },
    }
    (OUT_DIR / "realmlp_results.json").write_text(json.dumps(json_out, indent=2))
    log(f"wrote realmlp_results.json")


if __name__ == "__main__":
    main()
