"""Per-score log-bias tuning on greedy+nonrule OOF.

Hypothesis: the tuned log-bias is currently 3 global params
[0.132, 0.569, 3.401]. Errors cluster at scores 3 (4,849 rule-wrong)
and 6 (3,541 rule-wrong), so 81% of error mass lives in just 2 of the
10 score bins. Global bias is a compromise across bins. Score-
conditional bias (10 bins × 3 classes = 30 params) can operate at the
exact granularity where the errors live.

Overfit protection: nested CV. For each outer fold f, tune the 30-
param bias using rows NOT in fold f (as tuning signal), then apply
the tuned bias to fold-f rows. Union the 5 outer-fold-specific
predictions -> honest OOF.

Base: LB-best = log_blend2(nonrule, greedy, 0.15) at fixed global
bias [0.132, 0.569, 3.401], OOF 0.97421.

Outputs:
  scripts/artifacts/per_score_bias_results.json
  scripts/artifacts/oof_per_score_biased.npy  (post-bias preds as 3-probs)
  scripts/artifacts/test_per_score_biased.npy
  submissions/submission_greedy_nonrule_per_score_bias.csv (if lift)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold


SEED = 42
N_FOLDS = 5
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")
N_SCORES = 10  # dgp_score is 0..9

ART = Path("scripts/artifacts")
OUT = Path("submissions")
OUT.mkdir(exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    s = la + lb
    s -= s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


def compute_dgp_score(df):
    sm = df["Soil_Moisture"].astype(float).values
    rf = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = df["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)
    return (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)


def apply_per_score_bias(log_p, scores, bias_per_score):
    """log_p: (n, 3), scores: (n,), bias_per_score: (N_SCORES, 3)."""
    per_row = bias_per_score[scores]  # (n, 3)
    return (log_p + per_row).argmax(axis=1)


def tune_per_score_bias(log_p, y, scores, init_bias,
                        grid=None, max_iters=8):
    """Coord-ascent over 30 params (N_SCORES x 3) maximizing global bal_acc."""
    if grid is None:
        grid = np.linspace(-1.5, 1.5, 31)
    bias = np.tile(init_bias, (N_SCORES, 1)).astype(np.float64)  # (10, 3)
    best = balanced_accuracy_score(y, apply_per_score_bias(log_p, scores, bias))
    for it in range(max_iters):
        improved = False
        for s in range(N_SCORES):
            for k in range(3):
                base = bias.copy()
                bals = []
                for g in grid:
                    base[s, k] = bias[s, k] + g
                    bals.append(balanced_accuracy_score(
                        y, apply_per_score_bias(log_p, scores, base)
                    ))
                j = int(np.argmax(bals))
                if bals[j] > best + 1e-7:
                    bias[s, k] = bias[s, k] + grid[j]
                    best = bals[j]
                    improved = True
        log(f"    iter {it+1}  bal_acc={best:.6f}  improved={improved}")
        if not improved:
            break
    return bias, float(best)


def main():
    log("loading greedy + nonrule OOFs + greedy bias")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    test_greedy = np.load(ART / "test_greedy_blend.npy")
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")
    greedy_res = json.loads((ART / "greedy_binhigh_minimal_results.json").read_text())
    global_bias = np.array(greedy_res["greedy_bias"])
    log(f"  global bias = {global_bias.round(4).tolist()}")

    log("loading train/test for score computation")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")
    y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
    scores_tr = compute_dgp_score(tr)
    scores_te = compute_dgp_score(te)
    log(f"  train shape {tr.shape}  score counts (train): "
        f"{np.bincount(scores_tr, minlength=N_SCORES).tolist()}")

    # LB-best base = log_blend2(nonrule, greedy, alpha=0.15)
    log("rebuilding greedy+nonrule base (LB-best composition at alpha=0.15)")
    oof_base = log_blend2(oof_nonrule, oof_greedy, 0.15)
    test_base = log_blend2(test_nonrule, test_greedy, 0.15)
    log_p_oof = np.log(np.clip(oof_base, 1e-9, 1.0))
    log_p_test = np.log(np.clip(test_base, 1e-9, 1.0))

    # Reference: global bias applied to the base
    base_ba = balanced_accuracy_score(y,
        apply_per_score_bias(log_p_oof, scores_tr, np.tile(global_bias, (N_SCORES, 1))))
    log(f"  greedy+nonrule @ global bias (ref) bal_acc = {base_ba:.5f}")

    # ------------------ NESTED CV -------------------
    # For each outer fold f:
    #   tune per-score bias on rows NOT in fold f  (honest — no self-leak)
    #   apply tuned bias to fold f rows
    # Union all 5 folds -> nested-CV OOF prediction vector.
    log("NESTED CV: tune per-score bias on 4/5 folds, apply to held-out fold")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    nested_preds = -np.ones(len(y), dtype=np.int64)
    nested_bias_per_fold = []
    global_tile = np.tile(global_bias, (N_SCORES, 1)).astype(np.float64)
    for f, (tune_idx, apply_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        t0 = time.time()
        log(f"--- outer fold {f+1}/{N_FOLDS}  tune on {len(tune_idx)}  apply to {len(apply_idx)} ---")
        bias_f, bal_tune = tune_per_score_bias(
            log_p_oof[tune_idx], y[tune_idx], scores_tr[tune_idx],
            init_bias=global_bias,
        )
        nested_bias_per_fold.append(bias_f.tolist())
        nested_preds[apply_idx] = apply_per_score_bias(
            log_p_oof[apply_idx], scores_tr[apply_idx], bias_f
        )
        apply_ba = balanced_accuracy_score(y[apply_idx], nested_preds[apply_idx])
        log(f"  fold {f+1}  tune bal_acc={bal_tune:.5f}  "
            f"apply bal_acc={apply_ba:.5f}  ({time.time()-t0:.0f}s)")

    assert (nested_preds >= 0).all()
    nested_ba = balanced_accuracy_score(y, nested_preds)
    log(f"NESTED-CV OOF bal_acc = {nested_ba:.5f}  (vs global-bias {base_ba:.5f}  "
        f"Δ = {nested_ba - base_ba:+.5f})")

    # ------------------ FULL-FIT bias for TEST TIME -------------------
    # Tune once on ALL training OOF -> final bias vector used at test time.
    # For test predictions we don't have nested structure available (no
    # "outer fold" on test), so we use the globally-tuned bias. This is
    # the standard approach: nested CV gives the honest OOF score, the
    # full-fit params ship to production / test.
    log("FULL-FIT: tune per-score bias on ALL 630k OOF (for test inference)")
    full_bias, full_tune_ba = tune_per_score_bias(
        log_p_oof, y, scores_tr, init_bias=global_bias,
    )
    log(f"  full-fit tune bal_acc = {full_tune_ba:.5f}  "
        f"(training-data self-optimum, expected slight overfit vs nested)")

    # Diagnostic: show full-fit bias vs global bias per score
    log("per-score bias vs global (full-fit):")
    for s in range(N_SCORES):
        delta = full_bias[s] - global_bias
        log(f"  score {s} (n={int((scores_tr == s).sum()):6d}): "
            f"Low={full_bias[s,0]:+.3f} (Δ {delta[0]:+.3f})  "
            f"Med={full_bias[s,1]:+.3f} (Δ {delta[1]:+.3f})  "
            f"High={full_bias[s,2]:+.3f} (Δ {delta[2]:+.3f})")

    # OOF probs post-bias (for potential downstream blending)
    # We use the FULL-FIT bias for the OOF-probs-as-feature view, and the
    # NESTED predictions as the honest accuracy estimate.
    oof_biased_probs = oof_base.copy()
    # Apply full-fit per-score bias in logit space -> re-softmax
    adj_logp = log_p_oof + full_bias[scores_tr]
    exp_a = np.exp(adj_logp - adj_logp.max(axis=1, keepdims=True))
    oof_biased_probs = exp_a / exp_a.sum(axis=1, keepdims=True)

    adj_logp_te = log_p_test + full_bias[scores_te]
    exp_te = np.exp(adj_logp_te - adj_logp_te.max(axis=1, keepdims=True))
    test_biased_probs = exp_te / exp_te.sum(axis=1, keepdims=True)

    np.save(ART / "oof_per_score_biased.npy", oof_biased_probs)
    np.save(ART / "test_per_score_biased.npy", test_biased_probs)

    # Confusion matrix at nested operating point
    cm = confusion_matrix(y, nested_preds)
    log(f"Nested OOF confusion matrix:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    # Action decision
    delta = nested_ba - base_ba
    results = {
        "global_bias": global_bias.tolist(),
        "full_fit_bias": full_bias.tolist(),
        "nested_bias_per_fold": nested_bias_per_fold,
        "base_oof_bal_acc": float(base_ba),
        "nested_oof_bal_acc": float(nested_ba),
        "full_fit_tune_bal_acc": float(full_tune_ba),
        "delta_nested_vs_base": float(delta),
    }

    if delta < 1e-5:
        log(f"NO LIFT (Δ = {delta:+.5f}) — null result, no submission")
        results["action"] = "no_submission"
    elif delta < 3e-4:
        log(f"Δ = {delta:+.5f} below +0.0003 threshold — borderline")
        # Still emit submission for offline inspection
        preds = apply_per_score_bias(log_p_test, scores_te, full_bias)
        sub = OUT / "submission_greedy_nonrule_per_score_bias.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote borderline {sub}")
        results["action"] = "borderline_no_submit"
        results["submission_path"] = str(sub)
    else:
        # Apply full-fit bias to TEST preds
        preds = apply_per_score_bias(log_p_test, scores_te, full_bias)
        sub = OUT / "submission_greedy_nonrule_per_score_bias.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(sub, index=False)
        log(f"wrote {sub}")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub)

    with open(ART / "per_score_bias_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/per_score_bias_results.json")


if __name__ == "__main__":
    main()
