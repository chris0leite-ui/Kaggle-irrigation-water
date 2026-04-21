"""Soft-prob blending and meta-stacking over saved OOF/test probs.

Runs five ensemble strategies on any OOF/test NPYs present in
scripts/artifacts/:

  1. Per-model standalone tuned OOF (reference baseline)
  2. Pairwise prob-space + log-space α-sweep (replicates the
     original blend_lgbm_xgb_dist lever)
  3. Equal-weight multi-model prob average + log average
  4. Greedy forward selection: start with the best standalone, add
     the model whose log-blend at the OOF-best α most improves tuned
     bal_acc; stop when no further add helps
  5. Ridge + LogReg meta-stacker on concatenated OOF probs
     (optionally with dgp_score / rule one-hots as additional
     stacking features)

Every candidate is evaluated on OOF bal_acc with coord-ascent
log-bias tuning so results are directly comparable to the logged
numbers in CLAUDE.md. Only the globally best candidate writes a
submission; all per-strategy scores are saved to
scripts/artifacts/blend_ensemble_results.json.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}

# Map of display name -> (oof_npy, test_npy). We load whatever is
# present on disk and skip the rest with a warning.
CANDIDATES = {
    "lgbm_baseline":    ("oof_lgbm_baseline.npy",     "test_lgbm_baseline.npy"),
    "lgbm_dgp":         ("oof_lgbm_dgp.npy",          "test_lgbm_dgp.npy"),
    "lgbm_dist_bag":    ("oof_lgbm_dist_bag.npy",     "test_lgbm_dist_bag.npy"),
    "xgb_dist":         ("oof_xgb_dist.npy",          "test_xgb_dist.npy"),
    "xgb_routed_v3":    ("oof_xgb_dist_routed_v3.npy", "test_xgb_dist_routed_v3.npy"),
    "xgb_spec_678":     ("oof_xgb_spec_678.npy",      "test_xgb_spec_678.npy"),
    "xgb_hybrid":       ("oof_xgb_hybrid_routed_spec.npy", "test_xgb_hybrid_routed_spec.npy"),
    "xgb_hybrid_v3":    ("oof_xgb_hybrid_v3.npy",     "test_xgb_hybrid_v3.npy"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def per_class_recall(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out = {}
    for i, c in enumerate(CLASSES):
        mask = (y == i)
        out[c] = float(((pred == i) & mask).sum() / max(mask.sum(), 1))
    return out


def fast_bal_acc(y: np.ndarray, pred: np.ndarray, n_class: int = 3,
                 class_counts: np.ndarray | None = None) -> float:
    """Vectorized macro-recall — ~30x faster than sklearn for 630k rows."""
    if class_counts is None:
        class_counts = np.bincount(y, minlength=n_class)
    hit = np.zeros(n_class, dtype=np.int64)
    matches = (pred == y)
    for k in range(n_class):
        hit[k] = matches[y == k].sum()
    rec = hit / np.maximum(class_counts, 1)
    return float(rec.mean())


def tune_log_bias(oof: np.ndarray, y: np.ndarray, prior: np.ndarray,
                  high_grid_wide: bool = True, coarse: bool = False):
    """Coord-ascent log-bias. If high_grid_wide, extend the search
    range for the High class to [-3, 6] since the optimum is typically
    around +3.4 on this dataset (see CLAUDE.md). If coarse, use a
    smaller grid + fewer outer iters (for inner-loop sweeps).
    """
    log_oof = np.log(np.clip(oof, 1e-9, 1.0))
    bias = -np.log(prior)
    cc = np.bincount(y, minlength=3)
    best = fast_bal_acc(y, (log_oof + bias).argmax(axis=1), class_counts=cc)
    if coarse:
        grid_default = np.linspace(-2.0, 2.0, 21)
        grid_high = np.linspace(-1.0, 5.0, 25)
        max_iter = 6
    else:
        grid_default = np.linspace(-3.0, 3.0, 61)
        grid_high = np.linspace(-3.0, 6.0, 91) if high_grid_wide else grid_default
        max_iter = 25
    for _ in range(max_iter):
        improved = False
        for k in range(len(CLASSES)):
            grid = grid_high if k == 2 else grid_default  # k=2 is High
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(fast_bal_acc(y, (log_oof + base).argmax(axis=1), class_counts=cc))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, best


def load_present() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {name: (oof, test)} for every pair whose npys exist."""
    out = {}
    for name, (f_oof, f_test) in CANDIDATES.items():
        p_oof = ART / f_oof
        p_test = ART / f_test
        if p_oof.exists() and p_test.exists():
            oof = np.load(p_oof)
            test = np.load(p_test)
            # normalize rows to sum 1 in case any are logits
            if oof.min() < 0 or oof.max() > 1.0 + 1e-6:
                oof = np.exp(oof - oof.max(axis=1, keepdims=True))
                oof = oof / oof.sum(axis=1, keepdims=True)
                test = np.exp(test - test.max(axis=1, keepdims=True))
                test = test / test.sum(axis=1, keepdims=True)
            out[name] = (oof, test)
            log(f"loaded {name}  oof={oof.shape}  test={test.shape}")
        else:
            log(f"SKIP   {name}  (missing {f_oof} or {f_test})")
    return out


def prob_blend(oofs: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    w = weights / weights.sum()
    out = np.zeros_like(oofs[0])
    for wi, o in zip(w, oofs):
        out += wi * o
    return out


def log_blend(oofs: list[np.ndarray], weights: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    w = weights / weights.sum()
    logits = np.zeros_like(oofs[0])
    for wi, o in zip(w, oofs):
        logits += wi * np.log(np.clip(o, eps, 1.0))
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p


def pairwise_sweep(oof_a: np.ndarray, oof_b: np.ndarray,
                   test_a: np.ndarray, test_b: np.ndarray,
                   y: np.ndarray, prior: np.ndarray):
    """Return (best_alpha, best_tuned, space, oof_blend, test_blend, bias)."""
    alphas = np.linspace(0.0, 1.0, 11)
    best = (None, -1.0, None, None, None, None)
    for space in ("prob", "log"):
        for a in alphas:
            if space == "prob":
                blend = a * oof_a + (1 - a) * oof_b
                tblend = a * test_a + (1 - a) * test_b
            else:
                blend = log_blend([oof_a, oof_b], np.array([a, 1 - a]))
                tblend = log_blend([test_a, test_b], np.array([a, 1 - a]))
            bias, tuned = tune_log_bias(blend, y, prior, coarse=True)
            if tuned > best[1]:
                best = (float(a), float(tuned), space, blend, tblend, bias)
    # refine winner with full tune
    _, refined = tune_log_bias(best[3], y, prior)
    bias_ref, _ = tune_log_bias(best[3], y, prior)
    best = (best[0], float(refined), best[2], best[3], best[4], bias_ref)
    return best


def greedy_forward(oofs: dict[str, np.ndarray], tests: dict[str, np.ndarray],
                   y: np.ndarray, prior: np.ndarray):
    """Start with best standalone; iteratively add the component that
    most improves log-blended tuned bal_acc at the OOF-best α.
    Returns list of (name, weight, cumulative_tuned) and the final
    (oof_blend, test_blend, bias).
    """
    # standalone scores
    scores = {}
    for name, o in oofs.items():
        _, tuned = tune_log_bias(o, y, prior)
        scores[name] = tuned
    start = max(scores, key=scores.get)
    path = [(start, 1.0, scores[start])]
    current_oofs = [oofs[start]]
    current_tests = [tests[start]]
    current_weights = [1.0]
    remaining = set(oofs) - {start}
    while remaining:
        best_new = (None, -1.0, None, None, None)
        for cand in remaining:
            # add with equal weight first, then sweep mixing weight
            oofs_try = current_oofs + [oofs[cand]]
            tests_try = current_tests + [tests[cand]]
            # sweep mixing weight of the new model (coarse, then refine)
            best_w = (None, -1.0, None, None)
            for wnew in np.linspace(0.1, 0.9, 9):
                ws = np.array(current_weights + [wnew * sum(current_weights) / (1 - wnew)])
                blend_oof = log_blend(oofs_try, ws)
                blend_test = log_blend(tests_try, ws)
                _, tuned = tune_log_bias(blend_oof, y, prior, coarse=True)
                if tuned > best_w[1]:
                    best_w = (float(wnew), float(tuned), blend_oof, blend_test)
            if best_w[1] > best_new[1]:
                best_new = (cand, best_w[1], best_w[0], best_w[2], best_w[3])
        cand, tuned, wnew, blend_oof, blend_test = best_new
        prev = path[-1][2]
        if tuned <= prev + 1e-6:
            log(f"  stop: adding {cand} gives {tuned:.5f} <= {prev:.5f}")
            break
        ws_abs = np.array(current_weights + [wnew * sum(current_weights) / (1 - wnew)])
        # store normalized
        ws_norm = ws_abs / ws_abs.sum()
        current_weights = ws_norm.tolist()
        current_oofs.append(oofs[cand])
        current_tests.append(tests[cand])
        path.append((cand, float(ws_norm[-1]), float(tuned)))
        remaining.remove(cand)
        log(f"  add {cand:20s} w={ws_norm[-1]:.3f}  tuned={tuned:.5f}")
    # build final
    final_oof = log_blend(current_oofs, np.array(current_weights))
    final_test = log_blend(current_tests, np.array(current_weights))
    bias, tuned = tune_log_bias(final_oof, y, prior)
    return path, final_oof, final_test, bias, tuned


def meta_stack(oofs: dict[str, np.ndarray], tests: dict[str, np.ndarray],
               y: np.ndarray, prior: np.ndarray, extra_feat: np.ndarray | None = None,
               extra_test: np.ndarray | None = None):
    """Fit a logistic-regression meta-model on concat([P_1, ..., P_k])
    + optional extra features. Uses 5-fold CV to produce OOF meta
    predictions that don't leak.
    """
    from sklearn.model_selection import StratifiedKFold

    names = list(oofs.keys())
    X_oof = np.concatenate([oofs[n] for n in names], axis=1)
    X_test = np.concatenate([tests[n] for n in names], axis=1)
    if extra_feat is not None:
        X_oof = np.concatenate([X_oof, extra_feat], axis=1)
        X_test = np.concatenate([X_test, extra_test], axis=1)

    meta_oof = np.zeros((len(y), 3))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # class_weight='balanced' upweights High (~30× Low); produces
    # probabilities already biased toward bal-acc optimum. Post-hoc
    # log-bias tuning still applied and typically finds near-zero
    # residual bias.
    for f, (tr, va) in enumerate(skf.split(X_oof, y)):
        lr = LogisticRegression(
            multi_class="multinomial", solver="lbfgs", C=1.0,
            class_weight="balanced", max_iter=1000, n_jobs=1)
        lr.fit(X_oof[tr], y[tr])
        meta_oof[va] = lr.predict_proba(X_oof[va])
        log(f"  meta fold {f+1}/5 done")
    # refit on all oof for test pred
    lr = LogisticRegression(
        multi_class="multinomial", solver="lbfgs", C=1.0,
        class_weight="balanced", max_iter=1000, n_jobs=1)
    lr.fit(X_oof, y)
    meta_test = lr.predict_proba(X_test)

    bias, tuned = tune_log_bias(meta_oof, y, prior)
    return meta_oof, meta_test, bias, tuned, names


def main():
    SUB.mkdir(exist_ok=True)
    ART.mkdir(exist_ok=True, parents=True)

    tr = pd.read_csv("data/train.csv", usecols=[ID, TARGET])
    te_ids = pd.read_csv("data/test.csv", usecols=[ID])[ID].values
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"train={len(y)}  test={len(te_ids)}  prior={prior.round(4)}")

    loaded = load_present()
    if len(loaded) < 2:
        log(f"need >= 2 OOFs, have {len(loaded)}; exiting.")
        return
    oofs = {n: v[0] for n, v in loaded.items()}
    tests = {n: v[1] for n, v in loaded.items()}

    results: dict = {"standalone": {}, "pairwise": [], "equal_weight": {},
                     "greedy": None, "meta_stack": None}

    # (1) standalone
    log("\n=== standalone tuned OOF (with per-class recall) ===")
    for name, o in oofs.items():
        bias, tuned = tune_log_bias(o, y, prior)
        pred = (np.log(np.clip(o, 1e-9, 1.0)) + bias).argmax(axis=1)
        pcr = per_class_recall(y, pred)
        results["standalone"][name] = {
            "tuned": float(tuned),
            "recall_Low": pcr["Low"],
            "recall_Medium": pcr["Medium"],
            "recall_High": pcr["High"],
            "bias": bias.tolist(),
        }
        log(f"  {name:20s}  bal={tuned:.5f}  "
            f"rec_L={pcr['Low']:.4f} rec_M={pcr['Medium']:.4f} "
            f"rec_H={pcr['High']:.4f}  bias={np.round(bias, 2).tolist()}")

    # (2) pairwise α sweep
    log("\n=== pairwise prob + log blend ===")
    for a, b in combinations(oofs.keys(), 2):
        best_alpha, best_tuned, space, _, _, _ = pairwise_sweep(
            oofs[a], oofs[b], tests[a], tests[b], y, prior)
        results["pairwise"].append({
            "a": a, "b": b, "alpha": best_alpha,
            "space": space, "tuned": best_tuned,
        })
        log(f"  {a:18s} x {b:18s}  alpha={best_alpha:.2f} space={space} tuned={best_tuned:.5f}")

    # (3) equal-weight multi
    log("\n=== equal-weight prob/log across all components ===")
    ws = np.ones(len(oofs))
    o_list = list(oofs.values()); t_list = list(tests.values())
    o_avg = prob_blend(o_list, ws);  t_avg = prob_blend(t_list, ws)
    o_geo = log_blend(o_list, ws);   t_geo = log_blend(t_list, ws)
    _, b_avg = tune_log_bias(o_avg, y, prior)
    _, b_geo = tune_log_bias(o_geo, y, prior)
    results["equal_weight"]["prob_mean"] = float(b_avg)
    results["equal_weight"]["log_mean"] = float(b_geo)
    log(f"  prob mean of {len(oofs)}: tuned={b_avg:.5f}")
    log(f"  log  mean of {len(oofs)}: tuned={b_geo:.5f}")

    # (4) greedy forward
    log("\n=== greedy forward-selection ===")
    path, g_oof, g_test, g_bias, g_tuned = greedy_forward(oofs, tests, y, prior)
    results["greedy"] = {
        "path": [{"name": n, "weight": w, "tuned": t} for n, w, t in path],
        "final_tuned": float(g_tuned),
    }
    log(f"  greedy final: {g_tuned:.5f}")

    # (5) logistic meta-stack
    log("\n=== logistic-regression meta-stack ===")
    m_oof, m_test, m_bias, m_tuned, m_names = meta_stack(oofs, tests, y, prior)
    results["meta_stack"] = {
        "components": m_names,
        "tuned": float(m_tuned),
    }
    log(f"  meta-stack tuned: {m_tuned:.5f}")

    # pick best and write submission
    candidates = []
    best_standalone = max(results["standalone"].values())
    candidates.append(("standalone_best", best_standalone, None, None, None))
    for r in results["pairwise"]:
        candidates.append(("pair_" + r["a"] + "_" + r["b"], r["tuned"], None, None, None))
    candidates.append(("equal_prob", b_avg, o_avg, t_avg, None))
    candidates.append(("equal_log", b_geo, o_geo, t_geo, None))
    candidates.append(("greedy", g_tuned, g_oof, g_test, g_bias))
    candidates.append(("meta_stack", m_tuned, m_oof, m_test, m_bias))

    candidates.sort(key=lambda r: r[1], reverse=True)
    log("\n=== leaderboard (OOF tuned bal_acc, with High recall) ===")
    for name, score, oof_b, _, _ in candidates:
        if oof_b is None:
            log(f"  {name:40s}  bal={score:.5f}")
            continue
        bias, _ = tune_log_bias(oof_b, y, prior)
        pred = (np.log(np.clip(oof_b, 1e-9, 1.0)) + bias).argmax(axis=1)
        pcr = per_class_recall(y, pred)
        log(f"  {name:40s}  bal={score:.5f}  "
            f"rec_L={pcr['Low']:.4f} rec_M={pcr['Medium']:.4f} "
            f"rec_H={pcr['High']:.4f}")

    # write submissions for every configuration that beats best standalone by >= 0.0003
    threshold = best_standalone + 3e-4
    for name, score, oof_b, test_b, bias in candidates:
        if oof_b is None or score < threshold:
            continue
        if bias is None:
            bias, _ = tune_log_bias(oof_b, y, prior)
        pred_idx = (np.log(np.clip(test_b, 1e-9, 1.0)) + bias).argmax(axis=1)
        fname = f"submission_blend_{name}.csv"
        pd.DataFrame({ID: te_ids, TARGET: [IDX2CLS[i] for i in pred_idx]}).to_csv(
            SUB / fname, index=False
        )
        log(f"  wrote {fname}  OOF={score:.5f}")

    # always also re-emit the winner with a stable filename
    win = candidates[0]
    if win[2] is not None:
        bias = win[4]
        if bias is None:
            bias, _ = tune_log_bias(win[2], y, prior)
        pred_idx = (np.log(np.clip(win[3], 1e-9, 1.0)) + bias).argmax(axis=1)
        pd.DataFrame({ID: te_ids, TARGET: [IDX2CLS[i] for i in pred_idx]}).to_csv(
            SUB / "submission_blend_ensemble_best.csv", index=False
        )
        log(f"  wrote submission_blend_ensemble_best.csv  (winner={win[0]}  OOF={win[1]:.5f})")

    with open(ART / "blend_ensemble_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log("\ndone.")


if __name__ == "__main__":
    main()
