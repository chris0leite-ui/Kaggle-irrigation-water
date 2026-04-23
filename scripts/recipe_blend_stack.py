"""XGB-recipe + CAT-recipe: log-blend α sweep + LR meta-stacker.

Consumes OOF/test arrays from:
    scripts/artifacts/{oof,test}_recipe_full_te.npy   (XGB leg, LB 0.97939)
    scripts/artifacts/{oof,test}_recipe_catboost.npy  (CAT leg, this scaffold)

Outputs three candidates, each evaluated at the XGB-recipe's log-bias
(fixed — we don't retune to avoid binhigh-style selection overfit):

    1. log-blend α sweep: α ∈ {0.025 .. 0.6}.
    2. prob-blend α sweep (arithmetic, not geometric).
    3. LR stacking meta-learner (multinomial LR on concat([P_xgb, P_cat])).

Emits:
    submissions/submission_recipe_blend_w{alpha}.csv     (peak log-blend)
    submissions/submission_recipe_lr_stack.csv           (LR stacker)
    scripts/artifacts/recipe_blend_stack_results.json

Gate: submit only if fixed-bias Δ vs XGB-recipe OOF ≥ +5e-4.
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
from common import fast_bal_acc, tune_log_bias  # noqa: E402

SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fixed_bias_ba(probs: np.ndarray, y: np.ndarray, bias: np.ndarray) -> float:
    lp = np.log(np.clip(probs, 1e-9, 1.0))
    return fast_bal_acc(y.astype(np.int32), (lp + bias).argmax(1))


def log_blend(p1: np.ndarray, p2: np.ndarray, w1: float) -> np.ndarray:
    l = w1 * np.log(np.clip(p1, 1e-9, 1.0)) + (1 - w1) * np.log(np.clip(p2, 1e-9, 1.0))
    l -= l.max(1, keepdims=True)
    e = np.exp(l)
    return e / e.sum(1, keepdims=True)


def prob_blend(p1: np.ndarray, p2: np.ndarray, w1: float) -> np.ndarray:
    return w1 * p1 + (1 - w1) * p2


def main():
    # Load both legs.
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias = np.array(recipe_res["log_bias"])
    xgb_oof = np.load(ART / "oof_recipe_full_te.npy")
    xgb_test = np.load(ART / "test_recipe_full_te.npy")

    cat_oof_path = ART / "oof_recipe_catboost.npy"
    cat_test_path = ART / "test_recipe_catboost.npy"
    if not cat_oof_path.exists():
        log(f"MISSING {cat_oof_path}; run scripts/recipe_catboost.py first")
        return
    cat_oof = np.load(cat_oof_path)
    cat_test = np.load(cat_test_path)

    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)

    # Anchor XGB OOF + sanity.
    xgb_ba = fixed_bias_ba(xgb_oof, y, bias)
    cat_ba = fixed_bias_ba(cat_oof, y, bias)
    log(f"XGB-recipe OOF at its bias  = {xgb_ba:.5f}")
    log(f"CAT-recipe OOF at XGB bias  = {cat_ba:.5f}")

    # Error Jaccard + counts — diagnostic for blend potential.
    xgb_err = (np.log(np.clip(xgb_oof, 1e-9, 1.0)) + bias).argmax(1) != y
    cat_err = (np.log(np.clip(cat_oof, 1e-9, 1.0)) + bias).argmax(1) != y
    jacc = (xgb_err & cat_err).sum() / max(1, (xgb_err | cat_err).sum())
    log(f"Error counts: XGB={xgb_err.sum()}  CAT={cat_err.sum()}  "
        f"Jaccard={jacc:.4f}")
    if jacc >= 0.90:
        log("  Jaccard ≥ 0.90: blend ceiling ~+0.00015 expected")
    elif jacc < 0.65:
        log("  Jaccard < 0.65: decent orthogonality; check error magnitude")

    # ---------- (1) log-blend α sweep (fixed-bias AND tuned-bias diagnostic)
    alpha_grid = np.array(
        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
         0.70, 0.75, 0.80, 0.85, 0.90]
    )  # α = weight on XGB (anchor)
    log("\n--- log-blend sweep (α = weight on XGB) ---")
    log("  fixed-bias == XGB's bias; tuned = coord-ascent on blend (DIAGNOSTIC ONLY)")
    log_sweep = []
    for a in alpha_grid:
        b = log_blend(xgb_oof, cat_oof, a)
        ba_fixed = fixed_bias_ba(b, y, bias)
        _, ba_tuned = tune_log_bias(b, y.astype(np.int32), prior)
        log_sweep.append((float(a), ba_fixed, ba_tuned))
        log(f"  α={a:.2f}  fixed={ba_fixed:.5f}  tuned={ba_tuned:.5f}  "
            f"Δf={ba_fixed - xgb_ba:+.5f}  Δt={ba_tuned - xgb_ba:+.5f}")
    best_log_idx = int(np.argmax([s[1] for s in log_sweep]))
    best_log_alpha, best_log_ba, best_log_tuned = log_sweep[best_log_idx]
    best_log_tuned_alpha = log_sweep[int(np.argmax([s[2] for s in log_sweep]))][0]
    best_log_tuned_ba = max(s[2] for s in log_sweep)
    log(f"log-blend fixed-bias peak: α={best_log_alpha:.2f}  OOF={best_log_ba:.5f}  "
        f"Δ={best_log_ba - xgb_ba:+.5f}")
    log(f"log-blend tuned-bias peak (DIAG): α={best_log_tuned_alpha:.2f}  "
        f"OOF={best_log_tuned_ba:.5f}  Δ={best_log_tuned_ba - xgb_ba:+.5f}")

    # ---------- (2) prob-blend α sweep
    log("\n--- prob-blend sweep (arithmetic) ---")
    prob_sweep = []
    for a in alpha_grid:
        b = prob_blend(xgb_oof, cat_oof, a)
        ba_fixed = fixed_bias_ba(b, y, bias)
        _, ba_tuned = tune_log_bias(b, y.astype(np.int32), prior)
        prob_sweep.append((float(a), ba_fixed, ba_tuned))
        log(f"  α={a:.2f}  fixed={ba_fixed:.5f}  tuned={ba_tuned:.5f}  "
            f"Δf={ba_fixed - xgb_ba:+.5f}  Δt={ba_tuned - xgb_ba:+.5f}")
    best_prob_idx = int(np.argmax([s[1] for s in prob_sweep]))
    best_prob_alpha, best_prob_ba, best_prob_tuned = prob_sweep[best_prob_idx]
    best_prob_tuned_alpha = prob_sweep[int(np.argmax([s[2] for s in prob_sweep]))][0]
    best_prob_tuned_ba = max(s[2] for s in prob_sweep)
    log(f"prob-blend fixed-bias peak: α={best_prob_alpha:.2f}  OOF={best_prob_ba:.5f}  "
        f"Δ={best_prob_ba - xgb_ba:+.5f}")
    log(f"prob-blend tuned-bias peak (DIAG): α={best_prob_tuned_alpha:.2f}  "
        f"OOF={best_prob_tuned_ba:.5f}  Δ={best_prob_tuned_ba - xgb_ba:+.5f}")

    # ---------- (3) LR meta-stacker (honest inner 5-fold stacking)
    log("\n--- LR meta-stacker (inner 5-fold stacked features) ---")
    # Features: [P_xgb(3), P_cat(3), log P_xgb(3), log P_cat(3)] = 12.
    # Log-odds features help the LR separate high-confidence disagreements.
    X_meta = np.hstack([
        xgb_oof, cat_oof,
        np.log(np.clip(xgb_oof, 1e-9, 1.0)),
        np.log(np.clip(cat_oof, 1e-9, 1.0)),
    ]).astype(np.float32)

    X_test_meta = np.hstack([
        xgb_test, cat_test,
        np.log(np.clip(xgb_test, 1e-9, 1.0)),
        np.log(np.clip(cat_test, 1e-9, 1.0)),
    ]).astype(np.float32)

    # Inner 5-fold CV to produce honest meta OOF on the SAME split as the base
    # models — this gives apples-to-apples comparison with log/prob blends.
    inner = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    meta_oof = np.zeros_like(xgb_oof)
    meta_test = np.zeros_like(xgb_test)
    for i, (tr_i, va_i) in enumerate(inner.split(X_meta, y), 1):
        lr = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000, class_weight="balanced",
            random_state=SEED,
        )
        lr.fit(X_meta[tr_i], y[tr_i])
        meta_oof[va_i] = lr.predict_proba(X_meta[va_i]).astype(np.float32)
        meta_test += lr.predict_proba(X_test_meta).astype(np.float32) / N_FOLDS
        log(f"  inner fold {i} done")

    meta_ba_fixed = fixed_bias_ba(meta_oof, y, bias)
    # Also report tuned-bias diagnostic (UPPER BOUND — don't use for submission).
    _, meta_tuned = tune_log_bias(meta_oof, y.astype(np.int32), prior)
    log(f"LR-stack OOF @ XGB bias     = {meta_ba_fixed:.5f}  "
        f"Δ={meta_ba_fixed - xgb_ba:+.5f}")
    log(f"LR-stack OOF @ tuned bias   = {meta_tuned:.5f}  (diagnostic, not for submit)")

    # ---------- emit submissions
    def maybe_submit(name: str, test_probs: np.ndarray, oof_ba: float,
                     label: str) -> str | None:
        delta = oof_ba - xgb_ba
        if delta < 5e-4:
            log(f"  [skip] {name}: Δ={delta:+.5f} below +5e-4 gate")
            return None
        preds = (np.log(np.clip(test_probs, 1e-9, 1.0)) + bias).argmax(1)
        path = SUB / f"submission_recipe_{name}.csv"
        pd.DataFrame({
            "id": te["id"], TARGET: [IDX2CLS[i] for i in preds]
        }).to_csv(path, index=False)
        log(f"  [emit] {name}: Δ={delta:+.5f}  {path}")
        return str(path)

    log("\n--- emit candidates (gate Δ ≥ +5e-4) ---")
    best_log_test = log_blend(xgb_test, cat_test, best_log_alpha)
    best_prob_test = prob_blend(xgb_test, cat_test, best_prob_alpha)

    log_sub = maybe_submit(
        f"log_blend_a{int(best_log_alpha*100):02d}",
        best_log_test, best_log_ba, "log"
    )
    prob_sub = maybe_submit(
        f"prob_blend_a{int(best_prob_alpha*100):02d}",
        best_prob_test, best_prob_ba, "prob"
    )
    stack_sub = maybe_submit("lr_stack", meta_test, meta_ba_fixed, "LR")

    # ---------- dump results
    cm_best = confusion_matrix(
        y,
        (np.log(np.clip(log_blend(xgb_oof, cat_oof, best_log_alpha), 1e-9, 1.0))
         + bias).argmax(1),
    )
    log(f"\nlog-blend@α={best_log_alpha:.2f} confusion matrix:\n"
        f"{pd.DataFrame(cm_best, index=CLASSES, columns=CLASSES)}")

    out = dict(
        xgb_oof=xgb_ba, cat_oof=cat_ba,
        xgb_bias=bias.tolist(),
        error_jaccard=float(jacc),
        log_sweep=log_sweep,
        prob_sweep=prob_sweep,
        best_log_alpha=best_log_alpha, best_log_oof=best_log_ba,
        best_prob_alpha=best_prob_alpha, best_prob_oof=best_prob_ba,
        lr_stack_oof_fixed_bias=meta_ba_fixed,
        lr_stack_tuned_bias_diagnostic=meta_tuned,
        submissions=dict(
            log_blend=log_sub, prob_blend=prob_sub, lr_stack=stack_sub,
        ),
    )
    np.save(ART / "oof_recipe_lr_stack.npy", meta_oof)
    np.save(ART / "test_recipe_lr_stack.npy", meta_test)
    with open(ART / "recipe_blend_stack_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote {ART}/recipe_blend_stack_results.json")


if __name__ == "__main__":
    main()
