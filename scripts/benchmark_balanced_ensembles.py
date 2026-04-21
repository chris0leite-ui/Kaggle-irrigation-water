"""Balanced-ensemble methods on the DGP-enriched feature set.

Reference ceiling: LGBM+DGP tuned log-bias = 0.97271 (scripts/benchmark_dgp.py).

The hypothesis is that per-tree (or per-base-learner) undersampling of the
majority class changes the *split decisions* during training, rather than
just reweighting the loss. Under balanced accuracy that can matter when the
Medium <-> High boundary is the bottleneck, because each tree now sees the
High class at 33% instead of 3.3%.

Models (all from imbalanced-learn):
  - BalancedRandomForestClassifier: RF where each tree is trained on a
    class-balanced subsample (undersample majority to minority size).
  - EasyEnsembleClassifier: ensemble of AdaBoost models, each on a
    class-balanced random subsample of the training set.
  - RUSBoostClassifier: AdaBoost where each iteration random-undersamples
    to a balanced subset before fitting the weak learner.

Everything else (features, folds, seed, decision-rule sweep) matches
scripts/benchmark_dgp.py so the OOF bal_acc number is apples-to-apples.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
    RUSBoostClassifier,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

OUT_DIR = Path("submissions")
ART_DIR = Path("scripts/artifacts")
OUT_DIR.mkdir(exist_ok=True)
ART_DIR.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_dgp_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float)
    rm = out["Rainfall_mm"].astype(float)
    tc = out["Temperature_C"].astype(float)
    ws = out["Wind_Speed_kmh"].astype(float)
    out["dgp_dry"] = (sm < 25).astype(np.int8)
    out["dgp_norain"] = (rm < 300).astype(np.int8)
    out["dgp_hot"] = (tc > 30).astype(np.int8)
    out["dgp_windy"] = (ws > 10).astype(np.int8)
    out["dgp_nomulch"] = (out["Mulching_Used"].astype(str) == "No").astype(np.int8)
    out["dgp_kc"] = np.where(
        out["Crop_Growth_Stage"].astype(str).isin(["Flowering", "Vegetative"]),
        2, 0,
    ).astype(np.int8)
    out["dgp_score"] = (
        2 * (out["dgp_dry"] + out["dgp_norain"])
        + (out["dgp_hot"] + out["dgp_windy"] + out["dgp_nomulch"])
        + out["dgp_kc"]
    ).astype(np.int8)
    out["dgp_dist_moist"] = sm - 25.0
    out["dgp_dist_rain"] = rm - 300.0
    out["dgp_dist_temp"] = tc - 30.0
    out["dgp_dist_wind"] = ws - 10.0
    out["dgp_abs_moist"] = out["dgp_dist_moist"].abs()
    out["dgp_abs_rain"] = out["dgp_dist_rain"].abs()
    out["dgp_abs_temp"] = out["dgp_dist_temp"].abs()
    out["dgp_abs_wind"] = out["dgp_dist_wind"].abs()
    return out


log("loading data")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")

tr = add_dgp_features(tr)
te = add_dgp_features(te)

num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c not in (TARGET, ID)]
cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]

for c in cat_cols:
    mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
    tr[c] = tr[c].map(mapping).astype("int32")
    te[c] = te[c].map(mapping).astype("int32")

feature_cols = num_cols + cat_cols
log(f"features ({len(feature_cols)})")

X = tr[feature_cols].to_numpy(dtype=np.float32)
X_test = te[feature_cols].to_numpy(dtype=np.float32)
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)
log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")


def build_model(name: str):
    if name == "brf":
        return BalancedRandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=50,
            sampling_strategy="all",
            replacement=True,
            bootstrap=False,
            n_jobs=-1,
            random_state=SEED,
        )
    if name == "easy":
        # default (stumps) collapses on this 3-class problem; use a deeper
        # inner AdaBoost so each balanced subsample trains a real classifier.
        inner = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=SEED),
            n_estimators=40,
            learning_rate=0.3,
            random_state=SEED,
        )
        return EasyEnsembleClassifier(
            n_estimators=10,
            estimator=inner,
            sampling_strategy="not majority",
            n_jobs=-1,
            random_state=SEED,
        )
    if name == "rusb":
        # SAMME stumps give bal_acc=0.333 on this task; max_depth=5 trees
        # give a properly boosting sequence.
        return RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=SEED),
            n_estimators=200,
            learning_rate=0.3,
            algorithm="SAMME",
            sampling_strategy="not majority",
            random_state=SEED,
        )
    raise ValueError(name)


MODEL_NAMES = ["brf", "easy", "rusb"]
MODEL_TITLES = {"brf": "BalancedRF", "easy": "EasyEnsemble", "rusb": "RUSBoost"}


def score_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray) -> tuple[float, np.ndarray]:
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))

    def score_bias(bias):
        return balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))

    bias = -np.log(prior)
    best = score_bias(bias)
    grid = np.linspace(-2.5, 2.5, 51)
    for _ in range(20):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(score_bias(base))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return best, bias


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_splits = list(skf.split(X, y))

all_results: dict[str, dict] = {}

for mname in MODEL_NAMES:
    log(f"=== {MODEL_TITLES[mname]} ===")
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(fold_splits):
        t0 = time.time()
        model = build_model(mname)
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X[va_idx])
        test_pred += model.predict_proba(X_test) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], oof[va_idx].argmax(axis=1))
        log(
            f"  fold {fold+1}/{N_FOLDS}  bal_acc(argmax)={fold_bal:.5f}  "
            f"({time.time()-t0:.1f}s)"
        )

    results = [
        {"name": f"{MODEL_TITLES[mname]} argmax", "bal_acc": balanced_accuracy_score(y, oof.argmax(axis=1))},
        {
            "name": f"{MODEL_TITLES[mname]} prior-reweight",
            "bal_acc": balanced_accuracy_score(y, (oof / prior).argmax(axis=1)),
        },
    ]
    best, bias = score_log_bias(oof, y, prior)
    results.append({"name": f"{MODEL_TITLES[mname]} tuned log-bias", "bal_acc": best})
    log(f"  best bias = {dict(zip(CLASSES, bias.round(4)))}")

    # save OOF / test for later blending
    np.save(ART_DIR / f"oof_{mname}.npy", oof)
    np.save(ART_DIR / f"test_{mname}.npy", test_pred)

    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias).argmax(axis=1))
    log(f"  confusion matrix (tuned):\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    all_results[mname] = {
        "title": MODEL_TITLES[mname],
        "log_bias": bias.tolist(),
        "results": results,
        "cm_tuned": cm.tolist(),
    }

    # submission CSVs
    log_test = np.log(np.clip(test_pred, 1e-9, 1.0))
    tuned_test_idx = (log_test + bias).argmax(axis=1)
    pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in tuned_test_idx]}).to_csv(
        OUT_DIR / f"submission_{mname}_tuned.csv", index=False
    )

print("\n=== summary (OOF balanced accuracy) ===")
print(f"  Reference (LGBM+DGP tuned log-bias)      0.97271")
for mname in MODEL_NAMES:
    for r in all_results[mname]["results"]:
        print(f"  {r['name']:<45s}  {r['bal_acc']:.5f}")

with open(ART_DIR / "bench_balanced_ensembles.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "class_priors": prior.tolist(),
            "feature_cols": feature_cols,
            "reference_lgbm_dgp_tuned": 0.9727126135013285,
            "models": all_results,
        },
        f,
        indent=2,
    )
log("artefacts written")
