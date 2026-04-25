"""Item 2: recipe_full_te XGB retrained with multiclass focal-loss objective.

Rationale (from session-open brainstorm):
    The LB-best 3-way teacher hits the per-class recall Pareto frontier
    [0.9949, 0.9685, 0.9774] given the current OOF bank. Post-hoc
    overrides (detector, router, meta-stack) all nulled because they
    re-arrange the same bank. Focal loss is the one untried
    *training-time* lever — it changes the ERROR DISTRIBUTION produced
    by the base learner, not the calibration of an existing one.

Mechanism:
    Standard CE weights every row equally. Focal weights hard rows
    (low p_y) more than confident rows (high p_y) via (1-p_y)^γ.
    Under imbalance + heavy class-balanced sample weights, our recipe
    XGB is already biased toward the rare High class — but it treats
    every row within each class equally. Focal adds a SECOND axis of
    up-weighting: the hard boundary-band rows where flip signal lives.

Knobs (env vars):
    GAMMA       — focal γ (default 2.0; 0 == softmax CE baseline)
    ALPHA_HIGH  — High-class α multiplier on top of balanced (default 1.0)
    FOCAL_SMOKE — 1 → 20k rows / 2 folds / few iters (default 0)
    SMOKE       — same semantics

Outputs:
    scripts/artifacts/oof_recipe_focal_g<GAMMA>_aH<ALPHA>.npy
    scripts/artifacts/test_recipe_focal_g<GAMMA>_aH<ALPHA>.npy
    scripts/artifacts/recipe_focal_g<GAMMA>_aH<ALPHA>_results.json
    submissions/submission_recipe_focal_<suffix>.csv

Diagnostic at the end:
    - Jaccard vs LB-best 3-way teacher (fixed recipe bias, fixed teacher
      bias). Gate: Jaccard < 0.80 AND errors ≤ teacher errors for a
      plausible blend candidate.
    - Per-class recall delta vs teacher.
    - Fixed-bias blend sweep vs teacher for a first-look α sweep.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, log_blend, tune_log_bias  # noqa: E402
from recipe_focal_obj import make_focal_multi_obj  # noqa: E402
from recipe_full_te import load_and_engineer, TARGET, IDX2CLS, CLS_MAP  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
N_FOLDS = 5

GAMMA = float(os.environ.get("GAMMA", "2.0"))
ALPHA_HIGH_MULT = float(os.environ.get("ALPHA_HIGH", "1.0"))
SMOKE = os.environ.get("SMOKE") == "1" or os.environ.get("FOCAL_SMOKE") == "1"
if SMOKE:
    N_FOLDS = 2

SUFFIX = f"_g{GAMMA:g}_aH{ALPHA_HIGH_MULT:g}".replace(".", "p")
ART = Path("scripts/artifacts"); ART.mkdir(exist_ok=True, parents=True)
SUB = Path("submissions"); SUB.mkdir(exist_ok=True, parents=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def feval_bal_acc(preds, dmat):
    """Validation metric: macro-recall (argmax, no log-bias tune)."""
    y = dmat.get_label().astype(np.int64)
    # preds is (N, K) probs for multi:softprob
    if preds.ndim == 1:
        preds = preds.reshape(len(y), 3)
    pred = preds.argmax(axis=1)
    return "bal_acc", float(balanced_accuracy_score(y, pred))


def run_cv_focal(train, test, info, a_ote=1.0):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    y = train[TARGET].to_numpy().astype(np.int64)

    numeric_feats = (info["nums"] + info["tres"] + info["logits"]
                     + info["freq"] + info["orig_stats"]
                     + info.get("dae_embed", [])
                     + info.get("extra_domain", [])
                     + info.get("extra_decimal", [])
                     + info.get("gby", []))

    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

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
        y_tr = y[tr_idx]
        y_va = y[va_idx]

        # Per-class α: balanced (inverse-freq, normalized) × High multiplier.
        n_cls = np.bincount(y_tr, minlength=3).astype(np.float64)
        alpha = (len(y_tr) / (3.0 * np.maximum(n_cls, 1.0)))
        alpha[2] *= ALPHA_HIGH_MULT
        log(f"  α = {alpha.tolist()}  γ = {GAMMA}")

        dtr = xgb.DMatrix(X_tr[feat_cols].values, label=y_tr)
        dva = xgb.DMatrix(X_va[feat_cols].values, label=y_va)
        dte = xgb.DMatrix(X_te[feat_cols].values)

        params = dict(
            objective="multi:softprob", num_class=3,
            tree_method="hist", max_bin=256 if SMOKE else 1024,
            max_depth=4, max_leaves=30,
            learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=2, reg_alpha=5, reg_lambda=5,
            nthread=-1, verbosity=0,
        )
        obj = make_focal_multi_obj(gamma=GAMMA, alpha=alpha, K=3)

        log(f"  training XGB on {len(feat_cols)} features, "
            f"{len(X_tr):,} rows")
        booster = xgb.train(
            params, dtr,
            num_boost_round=300 if SMOKE else 3000,
            obj=obj,
            custom_metric=feval_bal_acc,
            evals=[(dva, "va")],
            early_stopping_rounds=50 if SMOKE else 200,
            verbose_eval=200,
            maximize=True,  # we maximize bal_acc (the custom metric)
        )

        oof[va_idx] = booster.predict(dva).astype(np.float32)
        test_pred += booster.predict(dte).astype(np.float32) / N_FOLDS

        bal = balanced_accuracy_score(y_va, oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  "
            f"best_iter = {booster.best_iteration}")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}  "
        f"(mean fold {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
    return dict(oof=oof, test=test_pred, fold_scores=fold_scores,
                overall_argmax=float(overall))


def save_outputs(result, y, test_ids, info, train):
    np.save(ART / f"oof_recipe_focal{SUFFIX}.npy", result["oof"])
    np.save(ART / f"test_recipe_focal{SUFFIX}.npy", result["test"])

    # Tune log bias on OOF
    prior = np.bincount(y) / len(y)
    bias, tuned = tune_log_bias(result["oof"], y, prior)
    log(f"tuned log-bias = {bias.tolist()}  bal_acc = {tuned:.5f}")

    # Per-class recall at tuned bias
    log_oof = np.log(np.clip(result["oof"], 1e-9, 1.0)) + bias
    pred = log_oof.argmax(1)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred)
    rec = cm.diagonal() / cm.sum(axis=1)
    log(f"per-class recall Low={rec[0]:.5f} Med={rec[1]:.5f} High={rec[2]:.5f}")

    out_meta = {
        "seed": SEED,
        "n_folds": N_FOLDS,
        "gamma": GAMMA,
        "alpha_high_mult": ALPHA_HIGH_MULT,
        "fold_scores_argmax": result["fold_scores"],
        "overall_argmax_bal_acc": result["overall_argmax"],
        "tuned_log_bias_bal_acc": float(tuned),
        "log_bias": bias.tolist(),
        "per_class_recall_tuned": rec.tolist(),
        "smoke": SMOKE,
    }
    with open(ART / f"recipe_focal{SUFFIX}_results.json", "w") as f:
        json.dump(out_meta, f, indent=2)

    # Submission at tuned bias
    log_test = np.log(np.clip(result["test"], 1e-9, 1.0)) + bias
    test_pred_labels = [IDX2CLS[int(i)] for i in log_test.argmax(1)]
    sub = pd.DataFrame({"id": test_ids, "Irrigation_Need": test_pred_labels})
    sub_path = SUB / f"submission_recipe_focal{SUFFIX}.csv"
    sub.to_csv(sub_path, index=False)
    log(f"wrote {sub_path}")


def blend_gate_vs_teacher(oof_focal, test_focal, y):
    """Diagnostic: Jaccard + error count + fixed-bias sweep vs LB-best 3-way.

    Teacher = log_blend of [recipe, pseudo_s1, pseudo_s7] at (0.25, 0.35, 0.40)
    using recipe's fixed tuned bias.
    """
    from common import load_oof_pair
    try:
        o_rec, t_rec = load_oof_pair("recipe_full_te")
        o_s1, t_s1 = load_oof_pair("recipe_pseudolabel")
        o_s7, t_s7 = load_oof_pair("recipe_pseudolabel_seed7labeler")
    except Exception as e:
        log(f"skipping blend gate (component OOFs missing): {e}")
        return {}

    with open(ART / "recipe_full_te_results.json") as f:
        bias = np.array(json.load(f)["log_bias"], dtype=np.float64)

    w = np.array([0.25, 0.35, 0.40])
    o_teacher = log_blend([o_rec, o_s1, o_s7], w)
    t_teacher = log_blend([t_rec, t_s1, t_s7], w)

    def argmax_at_bias(p, b):
        return (np.log(np.clip(p, 1e-9, 1.0)) + b).argmax(1)

    cc = np.bincount(y, minlength=3)

    teach_pred = argmax_at_bias(o_teacher, bias)
    focal_pred = argmax_at_bias(oof_focal, bias)
    teach_bal = fast_bal_acc(y, teach_pred, class_counts=cc)
    focal_bal = fast_bal_acc(y, focal_pred, class_counts=cc)

    teach_err = teach_pred != y
    focal_err = focal_pred != y
    inter = (teach_err & focal_err).sum()
    union = (teach_err | focal_err).sum()
    jacc = inter / max(union, 1)
    log(f"teacher OOF bal_acc @ recipe bias = {teach_bal:.5f}  errs={int(teach_err.sum()):,}")
    log(f"focal   OOF bal_acc @ recipe bias = {focal_bal:.5f}  errs={int(focal_err.sum()):,}")
    log(f"Jaccard(focal vs teacher) = {jacc:.4f}")
    log(f"  blend-gate: Jaccard<0.80 AND errs<=teacher → PLAUSIBLE")

    # Fixed-bias log-blend sweep vs teacher
    best = (0.0, teach_bal)
    results = []
    for alpha in np.linspace(0, 0.5, 11):
        if alpha == 0:
            o_mix = o_teacher
        else:
            o_mix = log_blend([o_teacher, oof_focal],
                              np.array([1 - alpha, alpha]))
        pred = argmax_at_bias(o_mix, bias)
        bal = fast_bal_acc(y, pred, class_counts=cc)
        results.append((float(alpha), float(bal)))
        if bal > best[1]:
            best = (float(alpha), float(bal))

    log(f"blend sweep vs teacher (fixed recipe bias):")
    for a, b in results:
        log(f"  α={a:.3f}  bal_acc={b:.5f}")
    log(f"peak α={best[0]:.3f}  bal={best[1]:.5f}  Δ={best[1]-teach_bal:+.5f}")

    diag = dict(
        teacher_oof_bal=teach_bal,
        teacher_errs=int(teach_err.sum()),
        focal_oof_bal=focal_bal,
        focal_errs=int(focal_err.sum()),
        jaccard_vs_teacher=jacc,
        blend_sweep=results,
        peak_alpha=best[0],
        peak_bal=best[1],
        delta_vs_teacher=best[1] - teach_bal,
    )
    with open(ART / f"recipe_focal{SUFFIX}_blend_gate.json", "w") as f:
        json.dump(diag, f, indent=2)
    return diag


def main():
    log(f"config: γ={GAMMA}  αHigh={ALPHA_HIGH_MULT}  smoke={SMOKE}  "
        f"suffix={SUFFIX!r}")
    train, test, info, test_ids = load_and_engineer()
    result = run_cv_focal(train, test, info, a_ote=1.0)
    y = train[TARGET].to_numpy().astype(np.int64)
    save_outputs(result, y, test_ids, info, train)
    if not SMOKE:
        blend_gate_vs_teacher(result["oof"], result["test"], y)


if __name__ == "__main__":
    main()
