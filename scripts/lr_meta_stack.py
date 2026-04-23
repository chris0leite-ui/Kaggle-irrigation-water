"""Logistic-regression meta-stacker over saved OOF probabilities.

Following the Ali Afzal kernel (LB 0.977) which uses multinomial LR on
stacked per-class OOF probs. We have 15+ OOFs available; a learned
meta-model can pick up signal the greedy heuristic misses.

Feature construction:
  For each base model, concatenate its 3 per-class probs -> 3 meta-features
  per model. Plus a sparse one-hot of dgp_score (0..9) and rule_pred (0..2)
  as "hand-crafted" meta features the kernel uses too.

Model: LogisticRegression(multi_class='multinomial', class_weight='balanced',
C=1.0, solver='lbfgs', max_iter=500). Trained with 5-fold CV aligned on
seed=42 folds to produce meta-OOF. Tuned log-bias on meta-OOF.

Decision: pick best-OOF model (recipe_full_te or meta). Emit submission
if meta-OOF > recipe OOF by >= +5e-5.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# Candidate OOFs for the meta stack. Skip extremely weak ones (tabpfn at 0.95)
# and overfit ones (binhigh family).
COMPONENTS = [
    ("recipe_full_te",   "oof_recipe_full_te.npy",               "test_recipe_full_te.npy"),
    ("greedy_full_bank", "oof_greedy_full_bank_6way.npy",        "test_greedy_full_bank_6way.npy"),
    ("digit_xgb",        "oof_xgb_dist_digits.npy",              "test_xgb_dist_digits.npy"),
    ("digits_ote",       "oof_xgb_dist_digits_ote_digits.npy",   "test_xgb_dist_digits_ote_digits.npy"),
    ("digits_pairs",     "oof_xgb_dist_digits_ote_digits_pairs.npy", "test_xgb_dist_digits_ote_digits_pairs.npy"),
    ("cat_ote",          "oof_xgb_dist_digits_ote.npy",          "test_xgb_dist_digits_ote.npy"),
    ("lgbm_digit_ote",   "oof_lgbm_dist_digits_ote.npy",         "test_lgbm_dist_digits_ote.npy"),
    ("xgb_nonrule",      "oof_xgb_nonrule.npy",                  "test_xgb_nonrule.npy"),
    ("xgb_corn",         "oof_xgb_corn.npy",                     "test_xgb_corn.npy"),
    ("hybrid_lgbmxgb",   "oof_hybrid_lgbmxgb_blend.npy",         "test_hybrid_lgbmxgb_blend.npy"),
]


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray):
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(axis=1))
    grid = np.linspace(-3.0, 3.0, 61)
    for _ in range(25):
        improved = False
        for k in range(len(CLASSES)):
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


def load_components():
    comps = {}
    for name, oof_name, test_name in COMPONENTS:
        op = ART / oof_name
        tp = ART / test_name
        if not op.exists() or not tp.exists():
            log(f"  skip {name}: missing")
            continue
        comps[name] = {
            "oof": np.load(op),
            "test": np.load(tp),
        }
    return comps


def main() -> None:
    log("loading components")
    comps = load_components()
    log(f"loaded {len(comps)}: {sorted(comps.keys())}")

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    n_train = len(tr)
    n_test = len(te)

    # Stack per-class probs side by side: shape (n, 3 * k_models).
    X_train = np.hstack([comps[n]["oof"] for n in comps])
    X_test = np.hstack([comps[n]["test"] for n in comps])
    log(f"meta feature shape: train {X_train.shape}, test {X_test.shape}")

    # 5-fold CV aligned on seed=42 StratifiedKFold.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros((n_train, len(CLASSES)), dtype=np.float64)
    meta_test = np.zeros((n_test, len(CLASSES)), dtype=np.float64)

    log("training 5-fold LR meta-stacker")
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(n_train), y)):
        t0 = time.time()
        lr = LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=SEED,
            n_jobs=-1,
        )
        lr.fit(X_train[tr_idx], y[tr_idx])
        meta_oof[va_idx] = lr.predict_proba(X_train[va_idx])
        meta_test += lr.predict_proba(X_test) / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], meta_oof[va_idx].argmax(axis=1))
        log(f"  fold {fold+1}/{N_FOLDS}  bal_acc(argmax)={fold_bal:.5f}  "
            f"({time.time()-t0:.1f}s)")

    argmax_bal = balanced_accuracy_score(y, meta_oof.argmax(axis=1))
    bias, tuned_bal = tune_log_bias(meta_oof, y, prior)

    cm = confusion_matrix(
        y, (np.log(np.clip(meta_oof, 1e-9, 1.0)) + bias).argmax(axis=1)
    )
    log(f"OOF CM:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # Compare against recipe_full_te standalone (LB-best anchor).
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    recipe_oof_tuned = recipe_res["tuned_log_bias_bal_acc"]

    print("\n=== LR meta-stacker OOF ===")
    print(f"  argmax          : {argmax_bal:.5f}")
    print(f"  tuned log-bias  : {tuned_bal:.5f}")
    print(f"  recipe_full_te  : {recipe_oof_tuned:.5f}  (reference anchor)")
    print(f"  Δ vs recipe    : {tuned_bal - recipe_oof_tuned:+.5f}")

    np.save(ART / "oof_lr_meta_stack.npy", meta_oof)
    np.save(ART / "test_lr_meta_stack.npy", meta_test)
    with open(ART / "lr_meta_stack_results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "n_meta_features": X_train.shape[1],
            "components": list(comps.keys()),
            "log_bias": bias.tolist(),
            "argmax_bal_acc": float(argmax_bal),
            "tuned_bal_acc": float(tuned_bal),
            "recipe_oof_tuned": recipe_oof_tuned,
            "delta_vs_recipe": float(tuned_bal - recipe_oof_tuned),
        }, f, indent=2)

    # Emit submission only if Δ vs recipe standalone >= +5e-5.
    delta = tuned_bal - recipe_oof_tuned
    if delta >= 5e-5:
        preds = (np.log(np.clip(meta_test, 1e-9, 1.0)) + bias).argmax(axis=1)
        sub = SUB / "submission_lr_meta_stack.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub, index=False
        )
        log(f"wrote {sub}  Δ = {delta:+.5f}")
    else:
        log(f"no submission: Δ {delta:+.5f} below +5e-5 emit gate")


if __name__ == "__main__":
    main()
