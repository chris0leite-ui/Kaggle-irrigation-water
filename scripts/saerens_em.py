"""Saerens-2002 EM posterior correction.

Given:
  - P_src(y|x): the model's OOF posterior on the labeled training set
  - Pi_src(y): the class prior in the labeled training set
  - P_src(y|x) applied to unlabeled test rows (or val rows we treat as
    an unlabeled "target" distribution for the calibration study)

Iterate EM to recover Pi_tgt(y) and P_tgt(y|x) under the label-shift
assumption P(x|y) is invariant between src and tgt (only the class
prior changes). Saerens, Latinne, Decaestecker (2002):
  P_tgt^(t+1)(y|x_i) = [P_src(y|x_i) * Pi_tgt^(t)(y) / Pi_src(y)]
                       / sum_y' [same]
  Pi_tgt^(t+1)(y)   = (1/N) sum_i P_tgt^(t+1)(y|x_i)

Because balanced accuracy = macro-recall, the Bayes-optimal decision
rule under a UNIFORM target prior (= pi_uniform = [1/3, 1/3, 1/3])
is argmax over EM-corrected posteriors at pi_tgt = [1/3, 1/3, 1/3].
This is the principled replacement for the heuristic log-bias
[1.43, 1.47, 3.40] currently tuned via coordinate-ascent on OOF.

What this script does:
  1. Build the LB-best 2-way blend as the source posterior.
  2. Compute three candidates:
     (a) EM-corrected OOF with target prior FIXED at [1/3, 1/3, 1/3]
         — this is the textbook balanced-accuracy-optimal decision.
     (b) EM-corrected OOF with target prior ESTIMATED by EM on
         the TEST set (unsupervised label-shift).
     (c) Tuned log-bias baseline for reference.
  3. Report OOF balanced accuracy for each.
  4. If (a) or (b) beats baseline at the fixed-bias gate, save
     corresponding test-set predictions and an emit-ready CSV.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "scripts" / "artifacts"
SUB = ROOT / "submissions"
CLASSES = ("Low", "Medium", "High")
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}


def log_blend(a: np.ndarray, b: np.ndarray, w: float = 0.5) -> np.ndarray:
    eps = 1e-12
    la = np.log(np.clip(a, eps, 1.0))
    lb = np.log(np.clip(b, eps, 1.0))
    z = w * la + (1.0 - w) * lb
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)


def em_posterior(P_src: np.ndarray, pi_src: np.ndarray,
                 pi_tgt_init: np.ndarray | None = None,
                 fix_target_prior: np.ndarray | None = None,
                 max_iter: int = 500,
                 tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray, int]:
    """EM label-shift correction.

    If `fix_target_prior` is given, uses that prior and only updates
    posteriors (no prior estimation — used for the fixed-uniform case).
    Otherwise, estimates pi_tgt iteratively from P_src.
    """
    P_src = np.clip(P_src, 1e-12, 1.0)
    if fix_target_prior is not None:
        pi_tgt = np.asarray(fix_target_prior, dtype=np.float64)
        ratio = pi_tgt / pi_src
        num = P_src * ratio
        P_tgt = num / num.sum(axis=1, keepdims=True)
        return P_tgt, pi_tgt, 1
    # EM
    pi_tgt = pi_src.copy() if pi_tgt_init is None else np.asarray(pi_tgt_init, dtype=np.float64).copy()
    for it in range(1, max_iter + 1):
        ratio = pi_tgt / pi_src
        num = P_src * ratio
        P_tgt = num / num.sum(axis=1, keepdims=True)
        pi_new = P_tgt.mean(axis=0)
        if np.abs(pi_new - pi_tgt).max() < tol:
            pi_tgt = pi_new
            return P_tgt, pi_tgt, it
        pi_tgt = pi_new
    return P_tgt, pi_tgt, max_iter


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray,
                  grid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Simple coord-ascent log-bias to match the existing pipeline."""
    if grid is None:
        grid = np.linspace(-1.0, 5.0, 121)  # 0.05 spacing
    eps = 1e-12
    log_oof = np.log(np.clip(oof, eps, 1.0))
    bias = np.zeros(3)
    best = balanced_accuracy_score(y, (log_oof + bias).argmax(1))
    improved = True
    iters = 0
    while improved and iters < 20:
        improved = False
        iters += 1
        for c in range(3):
            for v in grid:
                b_try = bias.copy()
                b_try[c] = v
                sc = balanced_accuracy_score(y, (log_oof + b_try).argmax(1))
                if sc > best + 1e-7:
                    best = sc
                    bias = b_try
                    improved = True
    return bias, best


def main() -> None:
    t0 = time.time()
    train = pd.read_csv(DATA / "train.csv")
    y = train["Irrigation_Need"].map(CLS_MAP).to_numpy()
    pi_src = np.bincount(y, minlength=3).astype(np.float64) / len(y)
    print(f"[load] train={len(train):,}  pi_src={pi_src.round(4).tolist()}")

    recipe = np.load(ART / "oof_recipe_full_te.npy")
    pseudo = np.load(ART / "oof_recipe_pseudolabel.npy")
    test_recipe = np.load(ART / "test_recipe_full_te.npy")
    test_pseudo = np.load(ART / "test_recipe_pseudolabel.npy")
    oof_blend = log_blend(recipe, pseudo, 0.5)
    test_blend = log_blend(test_recipe, test_pseudo, 0.5)
    print(f"[teacher] OOF blend shape {oof_blend.shape}  test blend shape {test_blend.shape}")

    # -- Baseline: argmax of blend ---------------------------
    argmax_base = balanced_accuracy_score(y, oof_blend.argmax(1))
    print(f"[baseline] OOF argmax bal_acc = {argmax_base:.6f}")

    # -- Baseline: recipe's tuned log-bias (= LB-best config) --
    recipe_bias = np.array([1.4324, 1.4689, 3.4008])
    adj = np.log(np.clip(oof_blend, 1e-12, 1.0)) + recipe_bias
    bias_base = balanced_accuracy_score(y, adj.argmax(1))
    print(f"[baseline] OOF recipe-bias bal_acc = {bias_base:.6f} (bias={recipe_bias.round(4).tolist()})")

    # -- Baseline: re-fit log-bias on OOF ---------------------
    fit_bias, fit_bal = tune_log_bias(oof_blend, y, pi_src)
    print(f"[baseline] OOF re-tuned log-bias bal_acc = {fit_bal:.6f} (bias={fit_bias.round(4).tolist()})")

    # -- (a) EM with fixed uniform target prior ---------------
    pi_uniform = np.array([1/3, 1/3, 1/3])
    oof_em_uniform, _, _ = em_posterior(oof_blend, pi_src, fix_target_prior=pi_uniform)
    bal_em_uniform = balanced_accuracy_score(y, oof_em_uniform.argmax(1))
    print(f"[EM-fixed-uniform] OOF bal_acc = {bal_em_uniform:.6f}  "
          f"Δ vs baseline argmax = {bal_em_uniform - argmax_base:+.6f}")

    # -- (b) EM with prior estimated from OOF itself (sanity; should
    # recover pi_src since OOF labels are IID with train) ----
    oof_em_self, pi_self, it_self = em_posterior(oof_blend, pi_src, pi_tgt_init=pi_src)
    print(f"[EM-self-oof] converged pi_tgt={pi_self.round(4).tolist()}  iters={it_self}")

    # -- (c) EM with prior estimated from TEST (unsup label-shift) --
    test_em, pi_test, it_test = em_posterior(test_blend, pi_src, pi_tgt_init=pi_src)
    print(f"[EM-on-test] converged pi_tgt={pi_test.round(4).tolist()}  iters={it_test}")
    # Apply the test-estimated prior to OOF (this tests whether
    # target-prior correction helps the decision even on OOF).
    oof_em_testprior, _, _ = em_posterior(oof_blend, pi_src, fix_target_prior=pi_test)
    bal_em_testprior = balanced_accuracy_score(y, oof_em_testprior.argmax(1))
    print(f"[EM-test-prior on OOF] OOF bal_acc = {bal_em_testprior:.6f}  "
          f"Δ vs baseline argmax = {bal_em_testprior - argmax_base:+.6f}")

    # -- Two-step: EM correct + re-tune bias on top ----------
    bias_unif, bal_unif_bias = tune_log_bias(oof_em_uniform, y, pi_uniform)
    print(f"[EM-uniform + bias] OOF bal_acc = {bal_unif_bias:.6f} "
          f"bias={bias_unif.round(4).tolist()}")

    # Save artefacts
    np.save(ART / "oof_em_uniform.npy", oof_em_uniform)
    np.save(ART / "test_em_uniform.npy",
            em_posterior(test_blend, pi_src, fix_target_prior=pi_uniform)[0])
    np.save(ART / "test_em_testprior.npy",
            em_posterior(test_blend, pi_src, fix_target_prior=pi_test)[0])

    # Per-class recall table for the best EM variant
    from sklearn.metrics import confusion_matrix
    best_name, best_oof = ("EM-uniform", oof_em_uniform) if bal_em_uniform >= bal_em_testprior \
        else ("EM-test-prior", oof_em_testprior)
    cm = confusion_matrix(y, best_oof.argmax(1))
    recall = cm.diagonal() / cm.sum(axis=1)
    print(f"[{best_name}] per-class recall  Low={recall[0]:.4f}  "
          f"Medium={recall[1]:.4f}  High={recall[2]:.4f}")

    results = dict(
        baseline_argmax=float(argmax_base),
        baseline_recipe_bias=float(bias_base),
        baseline_retuned_bias=float(fit_bal),
        retuned_bias=fit_bias.tolist(),
        em_uniform=float(bal_em_uniform),
        em_test_prior=float(bal_em_testprior),
        em_uniform_plus_bias=float(bal_unif_bias),
        em_uniform_plus_bias_vec=bias_unif.tolist(),
        pi_src=pi_src.tolist(),
        pi_test_estimated=pi_test.tolist(),
        em_test_iters=int(it_test),
        per_class_recall_best={
            "variant": best_name,
            "Low": float(recall[0]),
            "Medium": float(recall[1]),
            "High": float(recall[2]),
        },
        elapsed_sec=float(time.time() - t0),
    )
    (ART / "saerens_em_results.json").write_text(json.dumps(results, indent=2))

    # Build a submission from the best OOF variant (using the best-tuned
    # bias on that variant).
    test_best = em_posterior(test_blend, pi_src,
                             fix_target_prior=pi_uniform
                             if best_name == "EM-uniform" else pi_test)[0]
    log_test = np.log(np.clip(test_best, 1e-12, 1.0))
    # No extra bias for pure EM; that's the whole point.
    pred = log_test.argmax(1)

    sample = pd.read_csv(DATA / "sample_submission.csv")
    sub = pd.DataFrame({
        "id": sample["id"].values,
        "Irrigation_Need": [CLASSES[i] for i in pred],
    })
    sub_path = SUB / f"submission_saerens_{best_name.lower().replace('-', '_')}.csv"
    sub.to_csv(sub_path, index=False)
    print(f"[write] {sub_path}  dist={dict(sub['Irrigation_Need'].value_counts())}")

    print(f"[done] {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
