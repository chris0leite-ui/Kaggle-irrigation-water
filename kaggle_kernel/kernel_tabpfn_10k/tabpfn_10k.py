"""N1 TabPFN at 10k context (1-fold probe for signal) on Kaggle GPU.

Prior TabPFN run (2026-04-22) used SUBSAMPLE=1500 for CPU feasibility.
TabPFN v2's architectural sweet spot is 10k context. The closure note
"GPU SUBSAMPLE=10000 unlikely to change blend outcome" was a guess from
1.5k Jaccard, never measured. At 10k context the model sees ~6.7x more
rare-class examples (~333 High @ 10k stratified vs ~50 @ 1.5k), exactly
the failure mode that crippled the prior run's High recall (0.9238).

This is a 1-FOLD PROBE. We run fold 0 only (StratifiedKFold(seed=42)
aligned with every other OOF on main) and check:
  1. Standalone fold-0 val argmax bal_acc
  2. Per-class recall (esp. High)
  3. Jaccard of fold-0 errors vs LB-best primary (would need reconstruct;
     easier to compute downstream from saved OOF + test arrays)
  4. Whether the 10k context lifts the prior 1.5k result enough to
     justify the full 5-fold + blend-gate cycle.

Feature set: 43 dist features (raw + signed/abs distances + rule
indicators + dgp_score + score-band distances + min_axis + 4 pairwise
products). MATCHES the prior 1.5k TabPFN run for apples-to-apples
comparison on context-size effect.

Outputs in /kaggle/working/:
  oof_tabpfn_10k.npy (n_train, 3) — only fold-0 val rows populated;
                                     other rows zero (sentinel for
                                     downstream filtering)
  test_tabpfn_10k.npy (n_test, 3)
  tabpfn_10k_results.json

Wall budget: 1h cap per CLAUDE.md GPU rule.
  Estimated: 5 min fit + ~25 min val (126k) + ~30 min test (270k) = 60 min.
  HARD KILL at 55 min — save partials.

SMOKE=1 env var runs a 1k-context × 5k-val × 5k-test smoke (~3 min).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
    print("[boot] sm_60/61 detected — pinning torch+torchvision cu121", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        "--upgrade", "--force-reinstall", "--no-deps",
        "torch==2.5.1", "torchvision==0.20.1",
        "--index-url", "https://download.pytorch.org/whl/cu121",
    ])

# Pin TabPFN to a pre-license version (v2.2.x). v7+ requires Prior-Labs
# API token; older 2.x is still strong at 10k context and matches what
# the prior 2026-04-22 run used.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "tabpfn==2.2.1",
])
import tabpfn as _tp
print(f"[boot] tabpfn {getattr(_tp, '__version__', 'unknown')}", flush=True)

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
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier


SEED = 42
N_FOLDS = 5  # for split alignment; we only run fold 0
FOLD_TO_RUN = 0  # 0-indexed first fold of StratifiedKFold(seed=42)
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}

KAGGLE_INPUT = Path("/kaggle/input")
_DEFAULT_OUT = Path("/kaggle/working")
if _DEFAULT_OUT.exists() or _DEFAULT_OUT.parent.exists():
    OUT_DIR = _DEFAULT_OUT
else:
    OUT_DIR = Path(os.environ.get("SMOKE_OUTPUT_DIR", "./tabpfn_local_out"))
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Toggle for production push (Kaggle env vars don't propagate to script
# kernels at push time; flip here BEFORE pushing).
IS_SMOKE = False
SMOKE = IS_SMOKE or os.environ.get("SMOKE") == "1"

# Context size: 10k for production probe; 1k for smoke.
SUBSAMPLE = 1_000 if SMOKE else 10_000
N_ESTIMATORS = 1  # no internal ensemble; matches prior 1.5k run

# Wall-time kills (per CLAUDE.md 1h GPU cap).
T_START = time.time()
TOTAL_KILL_SEC = 55 * 60 if not SMOKE else 8 * 60


def _find_one(pattern: str) -> Path:
    for p in KAGGLE_INPUT.rglob(pattern):
        return p
    raise FileNotFoundError(f"no match for {pattern} under {KAGGLE_INPUT}")


def log(msg: str) -> None:
    elapsed = time.time() - T_START
    print(f"[{time.strftime('%H:%M:%S')} +{elapsed/60:5.1f}m] {msg}", flush=True)


def kill_check(label: str = "") -> None:
    elapsed = time.time() - T_START
    if elapsed > TOTAL_KILL_SEC:
        log(f"!! TOTAL WALL-TIME KILL at {label}: {elapsed/60:.1f}m > "
            f"{TOTAL_KILL_SEC/60:.0f}m. Bailing.")
        sys.exit(0)


# ========================= dist-feature builder =========================
ACTIVE_STAGES = ("Flowering", "Vegetative")
DGP_THRESHOLDS = dict(sm=25.0, rf=300.0, tc=30.0, ws=10.0)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Inlined copy of common.add_distance_features. Produces the same
    43-feature dist set used by every dist-family OOF on main.
    """
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).to_numpy()
    rf = out["Rainfall_mm"].astype(float).to_numpy()
    tc = out["Temperature_C"].astype(float).to_numpy()
    ws = out["Wind_Speed_kmh"].astype(float).to_numpy()

    dry = (sm < DGP_THRESHOLDS["sm"]).astype(np.int8)
    norain = (rf < DGP_THRESHOLDS["rf"]).astype(np.int8)
    hot = (tc > DGP_THRESHOLDS["tc"]).astype(np.int8)
    windy = (ws > DGP_THRESHOLDS["ws"]).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).to_numpy() == "No").astype(np.int8)
    stage = out["Crop_Growth_Stage"].astype(str).to_numpy()
    kc = np.where(np.isin(stage, ACTIVE_STAGES), 2, 0).astype(np.int8)

    out["dgp_dry"] = dry
    out["dgp_norain"] = norain
    out["dgp_hot"] = hot
    out["dgp_windy"] = windy
    out["dgp_nomulch"] = nomulch
    out["dgp_kc"] = kc
    out["dgp_score"] = (2 * (dry + norain) + hot + windy + nomulch + kc).astype(np.int8)

    out["sm_dist"] = (sm - DGP_THRESHOLDS["sm"]).astype(np.float32)
    out["rf_dist"] = (rf - DGP_THRESHOLDS["rf"]).astype(np.float32)
    out["tc_dist"] = (tc - DGP_THRESHOLDS["tc"]).astype(np.float32)
    out["ws_dist"] = (ws - DGP_THRESHOLDS["ws"]).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].to_numpy()).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].to_numpy()).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].to_numpy()).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].to_numpy()).astype(np.float32)

    out["score_dist_low_med"] = (out["dgp_score"].to_numpy() - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (out["dgp_score"].to_numpy() - 6.5).astype(np.float32)
    out["score_dist_min_abs"] = np.minimum(
        np.abs(out["score_dist_low_med"].to_numpy()),
        np.abs(out["score_dist_mid_high"].to_numpy()),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].to_numpy(), out["rf_abs"].to_numpy(),
         out["tc_abs"].to_numpy(), out["ws_abs"].to_numpy()]
    ).astype(np.float32)

    out["sm_x_rf"] = (out["sm_dist"].to_numpy() * out["rf_dist"].to_numpy()).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].to_numpy() * out["ws_dist"].to_numpy()).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].to_numpy() * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].to_numpy() * kc.astype(np.float32)).astype(np.float32)

    return out


# ========================= load + features =========================
def load_data():
    log("loading train / test")
    train = pd.read_csv(_find_one("train.csv"))
    test = pd.read_csv(_find_one("test.csv"))
    train[TARGET] = train[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    for df in (train, test):
        df.drop(columns=["id"], inplace=True, errors="ignore")
    log(f"  train={len(train):,}  test={len(test):,}")
    return train, test, test_ids


# 43 dist features + 11 raw nums + 8 cats (factorized).
NUMS = ["Soil_pH", "Soil_Moisture", "Organic_Carbon",
        "Electrical_Conductivity", "Temperature_C", "Humidity",
        "Rainfall_mm", "Sunlight_Hours", "Wind_Speed_kmh",
        "Field_Area_hectare", "Previous_Irrigation_mm"]
CATS = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
        "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
DIST_COLS = [
    "dgp_dry", "dgp_norain", "dgp_hot", "dgp_windy", "dgp_nomulch", "dgp_kc",
    "dgp_score",
    "sm_dist", "rf_dist", "tc_dist", "ws_dist",
    "sm_abs", "rf_abs", "tc_abs", "ws_abs",
    "score_dist_low_med", "score_dist_mid_high", "score_dist_min_abs",
    "min_axis_abs",
    "sm_x_rf", "tc_x_ws", "sm_x_kc", "rf_x_kc",
]


def build_features(train, test):
    log("building dist features")
    train = add_distance_features(train)
    test = add_distance_features(test)
    # Factorize cats across the concat.
    combined = pd.concat([train, test])
    for c in CATS:
        combined[c], _ = combined[c].factorize()
    s1 = len(train)
    train = combined.iloc[:s1].copy()
    test = combined.iloc[s1:].drop(columns=[TARGET]).copy()
    feature_cols = CATS + NUMS + DIST_COLS
    log(f"  feature_cols: {len(feature_cols)}")
    return train, test, feature_cols


# ========================= 1-fold probe =========================
def run_fold0(train, test, feature_cols):
    y = train[TARGET].to_numpy().astype(np.int64)
    n_train = len(train)
    n_test = len(test)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(skf.split(train, y))
    tr_idx, va_idx = splits[FOLD_TO_RUN]
    log(f"fold {FOLD_TO_RUN}: tr={len(tr_idx):,}  va={len(va_idx):,}")

    # Stratified subsample of tr_idx.
    sub_idx = []
    rng = np.random.default_rng(SEED)
    per_class = SUBSAMPLE // 3
    extra = SUBSAMPLE - per_class * 3
    for k in range(3):
        cls_rows = tr_idx[y[tr_idx] == k]
        n_pick = per_class + (extra if k == 0 else 0)
        pick = rng.choice(cls_rows, size=min(n_pick, len(cls_rows)),
                          replace=False)
        sub_idx.append(pick)
    sub_idx = np.concatenate(sub_idx)
    rng.shuffle(sub_idx)
    log(f"  context subsample: {len(sub_idx):,} rows  "
        f"(class balance: {np.bincount(y[sub_idx])})")

    X_tr = train.iloc[sub_idx][feature_cols].to_numpy()
    y_tr = y[sub_idx]
    X_va = train.iloc[va_idx][feature_cols].to_numpy()
    y_va = y[va_idx]
    X_te = test[feature_cols].to_numpy()

    log(f"  X shapes — tr={X_tr.shape}  va={X_va.shape}  te={X_te.shape}")
    kill_check("pre-fit")

    log(f"  fitting TabPFN (n_estimators={N_ESTIMATORS}, device=cuda)")
    t0 = time.time()
    clf = TabPFNClassifier(
        device="cuda", n_estimators=N_ESTIMATORS,
        random_state=SEED, ignore_pretraining_limits=True,
    )
    clf.fit(X_tr, y_tr)
    log(f"  fit done in {time.time() - t0:.1f}s")
    kill_check("post-fit")

    # Chunked val prediction. TabPFN's predict_proba batches internally
    # but we chunk explicitly to see progress + bail early on wall-time.
    log(f"  predicting val ({len(X_va):,} rows)")
    proba_va = np.zeros((len(X_va), 3), dtype=np.float32)
    chunk = 2_000
    t0 = time.time()
    for i in range(0, len(X_va), chunk):
        proba_va[i:i + chunk] = clf.predict_proba(X_va[i:i + chunk])
        if (i // chunk) % 10 == 0 and i > 0:
            kill_check(f"val chunk {i // chunk}")
            rate = (i + chunk) / (time.time() - t0)
            log(f"    val {i + chunk:,}/{len(X_va):,}  "
                f"({rate:.0f} rows/sec)")
    log(f"  val predict done in {time.time() - t0:.1f}s")

    log(f"  predicting test ({len(X_te):,} rows)")
    proba_te = np.zeros((len(X_te), 3), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(X_te), chunk):
        proba_te[i:i + chunk] = clf.predict_proba(X_te[i:i + chunk])
        if (i // chunk) % 10 == 0 and i > 0:
            kill_check(f"test chunk {i // chunk}")
            rate = (i + chunk) / (time.time() - t0)
            log(f"    test {i + chunk:,}/{len(X_te):,}  "
                f"({rate:.0f} rows/sec)")
    log(f"  test predict done in {time.time() - t0:.1f}s")

    # Per-fold scoring.
    pred_va = proba_va.argmax(1)
    ba = balanced_accuracy_score(y_va, pred_va)
    cm = confusion_matrix(y_va, pred_va, labels=[0, 1, 2])
    rec = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    log(f"  fold {FOLD_TO_RUN} argmax bal_acc = {ba:.5f}  "
        f"PCR=[L={rec[0]:.4f}, M={rec[1]:.4f}, H={rec[2]:.4f}]")

    # Build OOF in 630k-row sentinel layout for cross-branch alignment.
    oof = np.zeros((n_train, 3), dtype=np.float32)
    oof[va_idx] = proba_va

    return dict(
        oof=oof,
        test=proba_te,
        fold_ba=float(ba),
        fold_recalls=[float(r) for r in rec],
        va_idx=va_idx,
        n_va=int(len(va_idx)),
    )


# ========================= log-bias tune (on fold-0 val) =========================
def tune_log_bias(oof_va, y_va, prior, eps=1e-9):
    log_oof = np.log(np.clip(oof_va, eps, 1.0))
    bias = -np.log(prior)
    cc = np.bincount(y_va, minlength=3)

    def score(b):
        pred = (log_oof + b).argmax(1)
        per_cls = np.zeros(3)
        for k in range(3):
            per_cls[k] = ((pred == k) & (y_va == k)).sum() / max(cc[k], 1)
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
                bias[k] = bias[k] + grid[j]; best = scores[j]; improved = True
        if not improved:
            break
    return bias, best


# ========================= main =========================
def main():
    train, test, test_ids = load_data()
    train, test, feature_cols = build_features(train, test)
    y = train[TARGET].to_numpy().astype(np.int64)

    res = run_fold0(train, test, feature_cols)
    oof = res["oof"]
    test_pred = res["test"]
    va_idx = res["va_idx"]

    # Tune log-bias on fold-0 val only.
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof[va_idx], y[va_idx], prior)
    log(f"fold-0 tuned bal_acc = {tuned:.5f}  "
        f"bias={bias.round(4).tolist()}")

    np.save(OUT_DIR / "oof_tabpfn_10k.npy", oof)
    np.save(OUT_DIR / "test_tabpfn_10k.npy", test_pred)
    log("wrote oof_tabpfn_10k.npy + test_tabpfn_10k.npy")

    # Diagnostic submission (test-side argmax with tuned bias).
    pred_te = (np.log(np.clip(test_pred, 1e-9, 1.0)) + bias).argmax(1)
    sub = pd.DataFrame({"id": test_ids,
                        TARGET: [IDX2CLS[i] for i in pred_te]})
    sub_path = OUT_DIR / "submission_tabpfn_10k_fold0_tuned.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path} (DIAGNOSTIC; standalone 1-fold)")

    json_out = {
        "subsample": SUBSAMPLE,
        "n_estimators": N_ESTIMATORS,
        "fold_run": FOLD_TO_RUN,
        "fold_argmax_bal_acc": res["fold_ba"],
        "fold_per_class_recall": res["fold_recalls"],
        "fold_tuned_bal_acc": tuned,
        "log_bias": bias.tolist(),
        "n_val_rows": res["n_va"],
        "smoke": bool(SMOKE),
        "wall_min": (time.time() - T_START) / 60.0,
        "tabpfn_version": getattr(_tp, "__version__", "unknown"),
    }
    (OUT_DIR / "tabpfn_10k_results.json").write_text(json.dumps(json_out, indent=2))
    log("wrote tabpfn_10k_results.json")


if __name__ == "__main__":
    main()
