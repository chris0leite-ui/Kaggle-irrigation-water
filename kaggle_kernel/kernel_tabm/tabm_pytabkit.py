"""TabM-D via pytabkit on Kaggle GPU.

15th NN-family lever-existence test. TabM (ICLR 2025, Gorishniy et al.)
is a BatchEnsemble MLP + parameter-efficient ensembling. Mirrors the
RealMLP_TD_Classifier kernel exactly except the model swap. mikhailnaumov
public kernel imports `TabM_D_Classifier` from pytabkit but never
instantiates it; this is the first execution.

Three pushes per CLAUDE.md SMOKE-first + 1h GPU cap rules:
  1. SMOKE  (IS_SMOKE=True): 2-fold × 20k × 3 epochs (~5 min)
  2. PROBE  (IS_PROBE=True): 1 fold × full data × full epochs
  3. PROD   (both False):    5 folds × full data, fold-1 abort gate

Fold-1 abort gate (production only):
  - argmax bal_acc < 0.97  → ABORT (close to RealMLP n_ens=1's 0.96978
    fold-1; below this means TabM doesn't reach RealMLP's standalone bar
    and is unlikely to clear the magnitude trap on a blend)
  - wall-time > 20 min     → ABORT (fold-1 wall-time safety net)

Outputs in /kaggle/working/:
  oof_tabm.npy         (n_train, 3)
  test_tabm.npy        (n_test, 3)
  tabm_results.json    (per-fold metrics + abort reason if any)
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
from pytabkit import TabM_D_Classifier


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

KAGGLE_INPUT = Path("/kaggle/input")
_DEFAULT_OUT = Path("/kaggle/working")
if _DEFAULT_OUT.exists() or _DEFAULT_OUT.parent.exists():
    OUT_DIR = _DEFAULT_OUT
else:
    OUT_DIR = Path(os.environ.get("SMOKE_OUTPUT_DIR", "./tabm_local_out"))
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Top-level toggles for Kaggle pushes (env vars aren't settable at push).
IS_SMOKE = False  # SMOKE v2 PASSED — moved to PROBE
IS_PROBE = True   # 1 fold × full data × full epochs (~25-35 min ETA)
SMOKE = IS_SMOKE or os.environ.get("SMOKE") == "1"
PROBE = IS_PROBE or os.environ.get("PROBE") == "1"

if SMOKE:
    N_FOLDS = 2
elif PROBE:
    N_FOLDS = 1  # outer loop still iterates; we just stop after 1

# Fold-1 standalone bal_acc floor (production gate).
FOLD1_FLOOR = 0.97  # RealMLP n_ens=1 was 0.96978; below = unlikely to lift


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
    # try multiple casings for the original dataset name
    orig_path = None
    for pat in ("irrigation_prediction.csv", "Irrigation_Prediction*.csv",
                "irrigation*.csv"):
        try:
            orig_path = _find_one(pat)
            break
        except FileNotFoundError:
            continue
    if orig_path is None:
        raise FileNotFoundError("no irrigation original CSV found")
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
    """Same 19 raw + 15 pair combos as RealMLP kernel for apples-to-apples."""
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

    # In PROBE mode, we still build a 5-split skf so the held-out fold
    # rows align with every other OOF on disk (StratifiedKFold seed=42).
    skf = StratifiedKFold(
        n_splits=5 if (PROBE or not SMOKE) else 2,
        shuffle=True, random_state=SEED,
    )

    oof = np.zeros((n_train, 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_ba = []
    folds_completed = 0
    abort_reason = None
    t_start = time.time()
    FOLD1_KILL_SEC = 20 * 60 if not SMOKE else 10 * 60
    TOTAL_KILL_SEC = 55 * 60 if not SMOKE else 15 * 60

    max_folds = 1 if PROBE else (2 if SMOKE else 5)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y), 1):
        if fold > max_folds:
            log(f"PROBE/SMOKE limit reached: stopping after fold {fold-1}")
            break
        log(f"=== fold {fold}/{max_folds} ===")
        X_tr = train.iloc[tr_idx].copy()
        X_va = train.iloc[va_idx].copy()
        X_te = test.copy()
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        log(f"  fitting TargetEncoder on {len(te_cols)} combos")
        te = TargetEncoder(target_type="multiclass", cv=2,
                           random_state=SEED)
        te_tr = te.fit_transform(X_tr[te_cols], y_tr)
        te_va = te.transform(X_va[te_cols])
        te_te = te.transform(X_te[te_cols])

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

        cat_like = CATS + NUMS
        for frame in (tr_frame, va_frame, te_frame):
            for c in cat_like:
                frame[c] = frame[c].astype("category")

        log(f"  fitting TabM_D on {tr_frame.shape[1]} features, "
            f"{len(tr_frame):,} rows")
        t0 = time.time()
        # TabM_D has BatchEnsemble built in (tabm_k controls width;
        # default 32). No n_ens kwarg — pytabkit's TabM_D_Classifier
        # signature differs from RealMLP_TD_Classifier.
        # SMOKE: tabm_k=8 + 3 epochs (cuts compute ~4x for end-to-end check).
        # PROBE/PROD: default tabm_k=32 + 25 epochs (CLAUDE.md 1h GPU cap).
        kwargs = dict(
            n_cv=1, n_refit=0,
            device="cuda",
            val_metric_name="class_error",
            random_state=SEED,
            verbosity=1,
        )
        if SMOKE:
            kwargs["n_epochs"] = 3
            kwargs["tabm_k"] = 8
        else:
            kwargs["n_epochs"] = 25
        model = TabM_D_Classifier(**kwargs)
        model.fit(tr_frame, y_tr,
                  val_idxs=None,
                  cat_col_names=cat_like)
        log(f"  TabM fit in {time.time() - t0:.1f}s")

        proba_va = model.predict_proba(va_frame)
        proba_te = model.predict_proba(te_frame)
        oof[va_idx] = proba_va.astype(np.float32)
        test_pred += proba_te.astype(np.float32) / max_folds

        pred_va = proba_va.argmax(axis=1)
        ba = balanced_accuracy_score(y_va, pred_va)
        fold_ba.append(float(ba))
        folds_completed = fold
        elapsed = time.time() - t_start
        log(f"  fold {fold} argmax bal_acc = {ba:.5f}  "
            f"(elapsed {elapsed/60:.1f}m)")

        # Persist partial results AFTER each fold so an abort still
        # leaves usable artefacts.
        np.save(OUT_DIR / "oof_tabm.npy", oof)
        np.save(OUT_DIR / "test_tabm.npy", test_pred)
        log(f"  partial save: oof_tabm.npy + test_tabm.npy")

        # -------------- fold-1 abort gates (production only) --------------
        if fold == 1 and not SMOKE and not PROBE:
            if ba < FOLD1_FLOOR:
                abort_reason = (f"fold-1 argmax bal_acc {ba:.5f} < "
                                f"floor {FOLD1_FLOOR} - ABORT")
                log(f"!! {abort_reason}")
                break
            if elapsed > FOLD1_KILL_SEC:
                abort_reason = (f"fold-1 wall {elapsed/60:.1f}m > "
                                f"{FOLD1_KILL_SEC/60:.0f}m - ABORT")
                log(f"!! {abort_reason}")
                break
            log(f"  fold-1 GATE PASSED (bal_acc {ba:.5f} >= "
                f"{FOLD1_FLOOR}, wall {elapsed/60:.1f}m)")

        if elapsed > TOTAL_KILL_SEC:
            abort_reason = (f"total wall {elapsed/60:.1f}m > "
                            f"{TOTAL_KILL_SEC/60:.0f}m - "
                            f"{fold}/{max_folds} folds done")
            log(f"!! {abort_reason}")
            break

    if folds_completed and folds_completed < max_folds and max_folds > 1:
        test_pred *= max_folds / folds_completed

    if folds_completed > 0:
        nonzero_mask = oof.sum(axis=1) > 0
        if nonzero_mask.sum() > 0:
            overall_argmax = balanced_accuracy_score(
                y[nonzero_mask], oof[nonzero_mask].argmax(axis=1),
            )
        else:
            overall_argmax = 0.0
    else:
        overall_argmax = 0.0
    log(f"=== OOF argmax = {overall_argmax:.5f} "
        f"({folds_completed}/{max_folds} folds, "
        f"mean fold {np.mean(fold_ba) if fold_ba else 0:.5f} ± "
        f"{np.std(fold_ba) if fold_ba else 0:.5f})")
    return dict(
        oof=oof,
        test=test_pred,
        overall_argmax=float(overall_argmax),
        fold_ba=fold_ba,
        folds_completed=folds_completed,
        max_folds=max_folds,
        abort_reason=abort_reason,
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
    log(f"tuned OOF bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}  "
        f"({folds_completed}/{result['max_folds']} folds scored)")

    np.save(OUT_DIR / "oof_tabm.npy", oof)
    np.save(OUT_DIR / "test_tabm.npy", test_pred)
    log(f"wrote oof_tabm.npy + test_tabm.npy")

    pred = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(axis=1)
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in pred]})
    sub_path = OUT_DIR / "submission_tabm_tuned.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")

    json_out = {
        "overall_argmax_bal_acc": result["overall_argmax"],
        "tuned_log_bias_bal_acc": tuned,
        "log_bias": bias.tolist(),
        "fold_ba": result["fold_ba"],
        "n_folds": result["max_folds"],
        "folds_completed": folds_completed,
        "seed": SEED,
        "smoke": bool(SMOKE),
        "probe": bool(PROBE),
        "abort_reason": result["abort_reason"],
        "feature_counts": {
            "base_cols": len(CATS) + len(NUMS) + len(NUMS),
            "combo_cols": len(combo_cols),
        },
    }
    (OUT_DIR / "tabm_results.json").write_text(json.dumps(json_out, indent=2))
    log(f"wrote tabm_results.json")


if __name__ == "__main__":
    main()
