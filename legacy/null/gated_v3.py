"""Gated v3: meta-stacking + hard-gate over saved OOFs (no retraining).

v1 (soft blend of rule + main) tied LGBM+DGP at 0.9725 because both
sides of the blend already agree on clean rows. v2 (soft blend of
rule + flipped-only specialist) collapsed to 0.867 because the
specialist is OOD on clean rows. The flip-detector diagnostic showed
the residual signal IS there (AUC 0.899, flip-direction 99.4% bal on
flipped rows), so the problem is the hand-coded blending, not the
signal.

v3 tries two mechanisms that handle the OOD problem correctly:

  A. Hard gate:   predict rule if P_flip < tau else specialist.argmax.
                  Sweep tau on OOF, pick best.
  B. Meta-LGBM:   train a final LGBM on meta-features
                  [P_main(3), P_spec(3), P_flip(1), rule_oh(3), rule_int(1)]
                  using the same 5-fold split. Lets the meta model
                  learn when to route rule -> specialist based on P_flip.

Also re-evaluates the v1/v2 soft blends for comparison, all using
saved OOF artefacts:

  - oof_lgbm_dgp.npy / test_lgbm_dgp.npy  (LGBM+DGP, OOF 0.9727)
  - oof_flip_v2.npy  / test_flip_v2.npy   (binary flip detector)
  - oof_spec_v2.npy  / test_spec_v2.npy   (3-class specialist trained
                                           on flipped rows only)

All use seed=42 5-fold StratifiedKFold so OOFs are row-aligned.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

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


def dgp_rule_int(df: pd.DataFrame) -> np.ndarray:
    sm = df["Soil_Moisture"].astype(float).values
    rm = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    um = df["Mulching_Used"].astype(str).values
    stg = df["Crop_Growth_Stage"].astype(str).values
    dry = (sm < 25).astype(int)
    norain = (rm < 300).astype(int)
    hot = (tc > 30).astype(int)
    windy = (ws > 10).astype(int)
    nomulch = (um == "No").astype(int)
    kc = np.where(np.isin(stg, ["Flowering", "Vegetative"]), 2, 0)
    s = 2 * (dry + norain) + (hot + windy + nomulch) + kc
    return np.where(s <= 3, 0, np.where(s <= 6, 1, 2)).astype(np.int32)


def tune_bias(probs: np.ndarray, y: np.ndarray, prior: np.ndarray) -> tuple[float, np.ndarray]:
    log_p = np.log(np.clip(probs, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_p + bias).argmax(axis=1))
    grid = np.linspace(-2.5, 2.5, 51)
    for _ in range(20):
        improved = False
        for k in range(len(CLASSES)):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_p + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return float(best), bias


log("loading data + saved OOFs")
tr = pd.read_csv("data/train.csv")
te = pd.read_csv("data/test.csv")
y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
prior = np.bincount(y) / len(y)

rule_tr = dgp_rule_int(tr)
rule_te = dgp_rule_int(te)

oof_main = np.load(ART_DIR / "oof_lgbm_dgp.npy")
test_main = np.load(ART_DIR / "test_lgbm_dgp.npy")
oof_flip = np.load(ART_DIR / "oof_flip_v2.npy")
test_flip = np.load(ART_DIR / "test_flip_v2.npy")
oof_spec = np.load(ART_DIR / "oof_spec_v2.npy")
test_spec = np.load(ART_DIR / "test_spec_v2.npy")

log(f"shapes: main={oof_main.shape}  flip={oof_flip.shape}  spec={oof_spec.shape}")
log(f"test shapes: main={test_main.shape}  flip={test_flip.shape}  spec={test_spec.shape}")
log(f"priors: {dict(zip(CLASSES, prior.round(4)))}")

rule_bal = balanced_accuracy_score(y, rule_tr)
main_argmax_bal = balanced_accuracy_score(y, oof_main.argmax(axis=1))
main_tuned_bal, main_bias = tune_bias(oof_main, y, prior)
log(f"rule-only          bal={rule_bal:.5f}")
log(f"LGBM+DGP argmax    bal={main_argmax_bal:.5f}")
log(f"LGBM+DGP tuned     bal={main_tuned_bal:.5f}  bias={main_bias.round(3).tolist()}")


# === Candidate A: hard gate ================================================
log("\n=== Candidate A: hard-gate sweep (rule if P_flip<tau else specialist) ===")
spec_argmax = oof_spec.argmax(axis=1)
hard_results = []
for tau in np.linspace(0.05, 0.95, 19):
    pred = np.where(oof_flip > tau, spec_argmax, rule_tr)
    bal = balanced_accuracy_score(y, pred)
    raw = float((pred == y).mean())
    hard_results.append({"tau": float(tau), "bal_acc": float(bal), "raw": raw})
    log(f"  tau={tau:.2f}  bal={bal:.5f}  raw={raw:.5f}")
best_hard = max(hard_results, key=lambda r: r["bal_acc"])
log(f"best hard gate: tau={best_hard['tau']:.2f}  bal={best_hard['bal_acc']:.5f}")


# === Candidate B: soft blend rule + main ===================================
rule_oh_tr = np.eye(len(CLASSES))[rule_tr]
soft_main = (1 - oof_flip[:, None]) * rule_oh_tr + oof_flip[:, None] * oof_main
softB_argmax_bal = balanced_accuracy_score(y, soft_main.argmax(axis=1))
softB_tuned_bal, softB_bias = tune_bias(soft_main, y, prior)
log(f"\nsoft(rule+main)  argmax={softB_argmax_bal:.5f}  tuned={softB_tuned_bal:.5f}")


# === Candidate C: soft blend rule + spec (known-broken control) ===========
soft_spec = (1 - oof_flip[:, None]) * rule_oh_tr + oof_flip[:, None] * oof_spec
softC_argmax_bal = balanced_accuracy_score(y, soft_spec.argmax(axis=1))
softC_tuned_bal, softC_bias = tune_bias(soft_spec, y, prior)
log(f"soft(rule+spec)  argmax={softC_argmax_bal:.5f}  tuned={softC_tuned_bal:.5f}")


# === Candidate D: meta-LGBM stacking =======================================
log("\n=== Candidate D: meta-LGBM over [main(3), spec(3), flip(1), rule_oh(3), rule_int(1)] ===")


def make_meta(main: np.ndarray, spec: np.ndarray, flip: np.ndarray, rule_int: np.ndarray) -> np.ndarray:
    rule_oh = np.eye(len(CLASSES))[rule_int]
    return np.concatenate(
        [main, spec, flip[:, None], rule_oh, rule_int[:, None].astype(np.float64)],
        axis=1,
    )


X_meta = make_meta(oof_main, oof_spec, oof_flip, rule_tr)
X_meta_test = make_meta(test_main, test_spec, test_flip, rule_te)
log(f"meta shapes: train={X_meta.shape}  test={X_meta_test.shape}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_meta = np.zeros((len(y), len(CLASSES)), dtype=np.float64)
test_meta = np.zeros((len(te), len(CLASSES)), dtype=np.float64)

params_meta = dict(
    objective="multiclass",
    num_class=len(CLASSES),
    metric="multi_logloss",
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=1,
    min_data_in_leaf=500,
    verbose=-1,
    seed=SEED,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_meta, y)):
    t0 = time.time()
    dtr = lgb.Dataset(X_meta[tr_idx], label=y[tr_idx])
    dva = lgb.Dataset(X_meta[va_idx], label=y[va_idx], reference=dtr)
    m = lgb.train(
        params_meta, dtr, num_boost_round=2000,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
    )
    oof_meta[va_idx] = m.predict(X_meta[va_idx], num_iteration=m.best_iteration)
    test_meta += m.predict(X_meta_test, num_iteration=m.best_iteration) / N_FOLDS
    fb = balanced_accuracy_score(y[va_idx], oof_meta[va_idx].argmax(axis=1))
    log(f"  fold {fold+1}/{N_FOLDS}  best_iter={m.best_iteration}  bal={fb:.5f}  ({time.time()-t0:.1f}s)")

meta_argmax_bal = balanced_accuracy_score(y, oof_meta.argmax(axis=1))
meta_tuned_bal, meta_bias = tune_bias(oof_meta, y, prior)
log(f"meta argmax={meta_argmax_bal:.5f}  tuned={meta_tuned_bal:.5f}  bias={meta_bias.round(3).tolist()}")


# === summary ===============================================================
print("\n=== gated_v3 summary (OOF balanced accuracy) ===")
rows = [
    ("rule-only", rule_bal),
    ("LGBM+DGP argmax", main_argmax_bal),
    ("LGBM+DGP tuned", main_tuned_bal),
    (f"hard-gate (tau={best_hard['tau']:.2f})", best_hard["bal_acc"]),
    ("soft(rule+main) argmax", softB_argmax_bal),
    ("soft(rule+main) tuned", softB_tuned_bal),
    ("soft(rule+spec) argmax", softC_argmax_bal),
    ("soft(rule+spec) tuned", softC_tuned_bal),
    ("meta-LGBM argmax", meta_argmax_bal),
    ("meta-LGBM tuned", meta_tuned_bal),
]
w = max(len(name) for name, _ in rows)
for name, val in rows:
    print(f"  {name:<{w}s}  {val:.5f}")


# === pick best + write submission =========================================
candidates = {
    "LGBM+DGP tuned": (main_tuned_bal, "main"),
    f"hard-gate tau={best_hard['tau']:.2f}": (best_hard["bal_acc"], "hard"),
    "soft(rule+main) tuned": (softB_tuned_bal, "softB"),
    "meta-LGBM tuned": (meta_tuned_bal, "meta"),
}
best_name = max(candidates, key=lambda k: candidates[k][0])
best_bal, best_tag = candidates[best_name]
log(f"\nBEST: {best_name}  bal={best_bal:.5f}")

if best_tag == "main":
    log_p = np.log(np.clip(test_main, 1e-9, 1.0))
    test_idx = (log_p + main_bias).argmax(axis=1)
elif best_tag == "hard":
    tau = best_hard["tau"]
    test_idx = np.where(test_flip > tau, test_spec.argmax(axis=1), rule_te)
elif best_tag == "softB":
    rule_oh_te = np.eye(len(CLASSES))[rule_te]
    soft_te = (1 - test_flip[:, None]) * rule_oh_te + test_flip[:, None] * test_main
    log_p = np.log(np.clip(soft_te, 1e-9, 1.0))
    test_idx = (log_p + softB_bias).argmax(axis=1)
elif best_tag == "meta":
    log_p = np.log(np.clip(test_meta, 1e-9, 1.0))
    test_idx = (log_p + meta_bias).argmax(axis=1)
else:
    raise ValueError(best_tag)

pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in test_idx]}).to_csv(
    OUT_DIR / "submission_gated_v3.csv", index=False,
)
log(f"wrote {OUT_DIR}/submission_gated_v3.csv")

np.save(ART_DIR / "oof_meta_v3.npy", oof_meta)
np.save(ART_DIR / "test_meta_v3.npy", test_meta)
with open(ART_DIR / "gated_v3_results.json", "w") as f:
    json.dump(
        {
            "seed": SEED,
            "n_folds": N_FOLDS,
            "rule_bal": float(rule_bal),
            "main_argmax_bal": float(main_argmax_bal),
            "main_tuned_bal": float(main_tuned_bal),
            "main_bias": main_bias.tolist(),
            "hard_gate_sweep": hard_results,
            "best_hard_gate": best_hard,
            "softB_argmax_bal": float(softB_argmax_bal),
            "softB_tuned_bal": float(softB_tuned_bal),
            "softB_bias": softB_bias.tolist(),
            "softC_argmax_bal": float(softC_argmax_bal),
            "softC_tuned_bal": float(softC_tuned_bal),
            "meta_argmax_bal": float(meta_argmax_bal),
            "meta_tuned_bal": float(meta_tuned_bal),
            "meta_bias": meta_bias.tolist(),
            "best_name": best_name,
            "best_bal": float(best_bal),
        },
        f,
        indent=2,
    )
log(f"artefacts saved to {ART_DIR}/gated_v3_results.json")
