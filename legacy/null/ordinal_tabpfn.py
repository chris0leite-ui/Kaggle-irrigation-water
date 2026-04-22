"""TabPFN v2 as a new blend leg (brainstorm suggestion #1).

Hypothesis: TabPFN is a tabular foundation model pretrained on millions
of synthetic tabular DGPs — exactly the regime the host used to
generate the 630k synthetic train set. Every NN we've tried (v5-v9,
FT-Transformer, pretrain-finetune) was trained from scratch and
plateaued with error-magnitudes that defeated the blend math even
when fold-1 Jaccard looked promising. TabPFN starts from a prior
over tabular DGPs and conditions on our 10k-sized "training set"
each fold — the pretraining distribution may place the host's NN
DGP within its support.

Architecture:
  - TabPFN v2 handles up to ~10k training rows + arbitrary test
    rows per inference call. Full 504k train rows per fold is
    impossible — use 10k stratified subsample per fold (preserves
    class prior).
  - 5-fold StratifiedKFold(seed=42) — OOF-aligned with every other
    saved .npy. Val rows predicted against the subsample context.
  - 43-feature dist set (same as benchmark_dist.py / ordinal_corn).
  - n_estimators=2-4 (internal TabPFN ensemble over feature
    permutations). More = higher quality, more compute.

Protocol (learned from every prior null):
  1. Fold-1 error Jaccard GATE versus greedy + xgb_nonrule
     (LB-best components). Kill if either Jaccard >= 0.90 (TabPFN
     collapsed to the same decision surface as trees).
  2. Standalone tuned OOF.
  3. Fixed-bias log-blend sweep vs greedy (0.97375) and LB-best
     (0.97421). Emit submission only if sweep vs LB-best >= +0.0002.
  4. Save oof + test .npy for cross-branch reuse.

Artefacts:
  scripts/artifacts/oof_tabpfn.npy
  scripts/artifacts/test_tabpfn.npy
  scripts/artifacts/tabpfn_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

import os
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")       # force HTTPS fallback (xet CAS has outages)

import torch
torch.set_num_threads(16)

from tabpfn import TabPFNClassifier


SEED = 42
N_FOLDS = 5
SUBSAMPLE = 1_500            # stratified per-fold training rows for TabPFN
N_ESTIMATORS = 1             # internal TabPFN ensemble count (compute vs quality tradeoff)
TEST_BATCH = 5_000           # predict test in chunks to cap memory

# CPU-only timing notes (actual fold-1 timing at SUBSAMPLE=3000 was
# pacing ~45 min per fold on 16-core, way beyond smoke extrapolation —
# halved subsample + chunk size to target ~15 min per fold, accept
# some accuracy loss vs TabPFN sweet spot (10k)).
TARGET = "Irrigation_Need"
ID = "id"
CLASSES = ["Low", "Medium", "High"]
CLS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLS = {i: c for c, i in CLS2IDX.items()}
ACTIVE_STAGES = ("Flowering", "Vegetative")

ART = Path("scripts/artifacts")
OUT = Path("submissions")
ART.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def add_distance_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sm = out["Soil_Moisture"].astype(float).values
    rf = out["Rainfall_mm"].astype(float).values
    tc = out["Temperature_C"].astype(float).values
    ws = out["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8)
    norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8)
    windy = (ws > 10).astype(np.int8)
    nomulch = (out["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    stage_str = out["Crop_Growth_Stage"].astype(str).values
    kc = np.where(np.isin(stage_str, ACTIVE_STAGES), 2, 0).astype(np.int8)
    out["sm_dist"] = (sm - 25).astype(np.float32)
    out["rf_dist"] = (rf - 300).astype(np.float32)
    out["tc_dist"] = (tc - 30).astype(np.float32)
    out["ws_dist"] = (ws - 10).astype(np.float32)
    out["sm_abs"] = np.abs(out["sm_dist"].values).astype(np.float32)
    out["rf_abs"] = np.abs(out["rf_dist"].values).astype(np.float32)
    out["tc_abs"] = np.abs(out["tc_dist"].values).astype(np.float32)
    out["ws_abs"] = np.abs(out["ws_dist"].values).astype(np.float32)
    out["dry"] = dry
    out["norain"] = norain
    out["hot"] = hot
    out["windy"] = windy
    out["nomulch"] = nomulch
    out["kc_active"] = (kc > 0).astype(np.int8)
    score = (2 * (dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)
    out["dgp_score"] = score
    out["rule_pred"] = np.where(score <= 3, 0, np.where(score <= 6, 1, 2)).astype(np.int8)
    out["score_dist_low_mid"] = (score.astype(np.float32) - 3.5).astype(np.float32)
    out["score_dist_mid_high"] = (score.astype(np.float32) - 6.5).astype(np.float32)
    out["min_boundary_dist"] = np.minimum(
        np.abs(out["score_dist_low_mid"].values),
        np.abs(out["score_dist_mid_high"].values),
    ).astype(np.float32)
    out["min_axis_abs"] = np.minimum.reduce(
        [out["sm_abs"].values, out["rf_abs"].values,
         out["tc_abs"].values, out["ws_abs"].values]
    ).astype(np.float32)
    out["sm_x_rf"] = (out["sm_dist"].values * out["rf_dist"].values).astype(np.float32)
    out["tc_x_ws"] = (out["tc_dist"].values * out["ws_dist"].values).astype(np.float32)
    out["sm_x_kc"] = (out["sm_dist"].values * kc.astype(np.float32)).astype(np.float32)
    out["rf_x_kc"] = (out["rf_dist"].values * kc.astype(np.float32)).astype(np.float32)
    return out


def tune_log_bias(oof, y, prior):
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
                scores.append(balanced_accuracy_score(y, (log_oof + base).argmax(axis=1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]
                best = scores[j]
                improved = True
        if not improved:
            break
    return bias, float(best)


def log_blend2(p_a, p_b, w_a):
    la = w_a * np.log(np.clip(p_a, 1e-9, 1.0))
    lb = (1 - w_a) * np.log(np.clip(p_b, 1e-9, 1.0))
    logs = la + lb
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    return e / e.sum(1, keepdims=True)


def err_jaccard(pred_a, pred_b, y):
    e_a = (pred_a != y)
    e_b = (pred_b != y)
    inter = np.logical_and(e_a, e_b).sum()
    union = np.logical_or(e_a, e_b).sum()
    return float(inter / max(union, 1)), int(e_a.sum()), int(e_b.sum())


def predict_test_chunked(clf, X_test_np, label, batch_size=TEST_BATCH):
    n = len(X_test_np)
    out = np.zeros((n, len(CLASSES)), dtype=np.float64)
    n_batches = (n + batch_size - 1) // batch_size
    t_start = time.time()
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        bt = time.time()
        out[i:j] = clf.predict_proba(X_test_np[i:j])
        if (i // batch_size) % 5 == 0 or j == n:
            elapsed = time.time() - t_start
            k = i // batch_size + 1
            eta = (n_batches - k) * (elapsed / k) if k > 0 else 0
            log(f"    [{label}] batch {k}/{n_batches}  "
                f"batch_dt={time.time()-bt:.1f}s  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
    return out


def main() -> None:
    log("loading data")
    tr = pd.read_csv("data/train.csv")
    te = pd.read_csv("data/test.csv")

    log("building 43-feature dist set")
    tr = add_distance_features(tr)
    te = add_distance_features(te)

    num_cols = tr.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in (TARGET, ID)]
    cat_cols = [c for c in tr.columns if c not in num_cols + [TARGET, ID]]
    for c in cat_cols:
        mapping = {v: i for i, v in enumerate(sorted(tr[c].unique()))}
        tr[c] = tr[c].map(mapping).astype("int32")
        te[c] = te[c].map(mapping).astype("int32")

    feat_cols = num_cols + cat_cols
    X = tr[feat_cols].values.astype(np.float32)
    X_test = te[feat_cols].values.astype(np.float32)
    y = tr[TARGET].map(CLS2IDX).values.astype(np.int32)
    prior = np.bincount(y) / len(y)
    log(f"features: {len(feat_cols)} ({len(num_cols)} num + {len(cat_cols)} cat)")
    log(f"class priors: {dict(zip(CLASSES, prior.round(4)))}")
    log(f"TabPFN config: SUBSAMPLE={SUBSAMPLE}, N_ESTIMATORS={N_ESTIMATORS}")

    log("loading reference OOFs for Jaccard gate")
    oof_greedy = np.load(ART / "oof_greedy_blend.npy")
    oof_nonrule = np.load(ART / "oof_xgb_nonrule.npy")
    greedy_bias = np.array([
        0.13244116323609723, 0.568946691946548, 3.400768902044088
    ])
    greedy_tuned_oof = 0.973746084242468
    oof_lbbest = log_blend2(oof_nonrule, oof_greedy, 0.15)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((len(tr), len(CLASSES)), dtype=np.float64)
    test_pred = np.zeros((len(te), len(CLASSES)), dtype=np.float64)
    fold_times = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        t0 = time.time()
        # Stratified subsample of the training fold to fit TabPFN's cap.
        rng = np.random.RandomState(SEED + fold)
        X_tr_sub, _, y_tr_sub, _ = train_test_split(
            X[tr_idx], y[tr_idx],
            train_size=SUBSAMPLE, stratify=y[tr_idx], random_state=SEED + fold,
        )

        clf = TabPFNClassifier(
            device="cpu",
            n_estimators=N_ESTIMATORS,
            ignore_pretraining_limits=True,
            random_state=SEED + fold,
        )
        clf.fit(X_tr_sub, y_tr_sub)
        t_fit = time.time() - t0

        t0 = time.time()
        va_pred = predict_test_chunked(clf, X[va_idx], label=f"fold{fold+1}-val")
        t_val = time.time() - t0

        t0 = time.time()
        te_pred = predict_test_chunked(clf, X_test, label=f"fold{fold+1}-test")
        t_te = time.time() - t0

        oof[va_idx] = va_pred
        test_pred += te_pred / N_FOLDS
        fold_bal = balanced_accuracy_score(y[va_idx], va_pred.argmax(axis=1))
        fold_times.append(t_fit + t_val + t_te)
        log(f"  fold {fold+1}/{N_FOLDS}  bal_acc(argmax)={fold_bal:.5f}  "
            f"fit={t_fit:.1f}s val={t_val:.1f}s test={t_te:.1f}s")

        # Jaccard gate after fold 1 only.
        if fold == 0:
            pred_tabpfn = va_pred.argmax(axis=1)
            pred_greedy = oof_greedy[va_idx].argmax(axis=1)
            pred_lbbest = oof_lbbest[va_idx].argmax(axis=1)
            yv = y[va_idx]
            j_g, n_e_tab, n_e_g = err_jaccard(pred_tabpfn, pred_greedy, yv)
            j_lb, _, n_e_lb = err_jaccard(pred_tabpfn, pred_lbbest, yv)
            log(f"  fold-1 Jaccard gate:")
            log(f"    vs greedy: J={j_g:.4f}  errs tabpfn={n_e_tab}  errs greedy={n_e_g}")
            log(f"    vs lbbest: J={j_lb:.4f}  errs tabpfn={n_e_tab}  errs lbbest={n_e_lb}")
            if j_g >= 0.90 or j_lb >= 0.90:
                log(f"  ABORT: fold-1 Jaccard >= 0.90 — TabPFN mimicking tree ensemble, "
                    f"blend will be null.")
                results = {
                    "aborted": True,
                    "reason": f"fold1_jaccard_too_high j_greedy={j_g:.4f} j_lbbest={j_lb:.4f}",
                    "fold1_errs_tabpfn": n_e_tab,
                    "fold1_errs_greedy": n_e_g,
                    "fold1_errs_lbbest": n_e_lb,
                    "fold1_bal_acc": float(fold_bal),
                }
                with open(ART / "tabpfn_results.json", "w") as f:
                    json.dump(results, f, indent=2)
                return

    log(f"total wall time per fold (mean) = {np.mean(fold_times):.1f}s")

    argmax_bal = balanced_accuracy_score(y, oof.argmax(axis=1))
    bias_tabpfn, tuned_bal = tune_log_bias(oof, y, prior)
    log(f"TabPFN standalone  argmax={argmax_bal:.5f}  tuned={tuned_bal:.5f}")
    log(f"  bias = {dict(zip(CLASSES, bias_tabpfn.round(4)))}")

    cm = confusion_matrix(y, (np.log(np.clip(oof, 1e-9, 1.0)) + bias_tabpfn).argmax(axis=1))
    log(f"standalone CM:\n{pd.DataFrame(cm, index=CLASSES, columns=CLASSES)}")

    np.save(ART / "oof_tabpfn.npy", oof)
    np.save(ART / "test_tabpfn.npy", test_pred)

    # Full Jaccard diagnostic (all folds).
    j_g_full, e_tab, e_g = err_jaccard(oof.argmax(axis=1), oof_greedy.argmax(axis=1), y)
    j_lb_full, _, e_lb = err_jaccard(oof.argmax(axis=1), oof_lbbest.argmax(axis=1), y)
    log(f"full-OOF Jaccard vs greedy = {j_g_full:.4f}  "
        f"(errs tab={e_tab}  greedy={e_g})")
    log(f"full-OOF Jaccard vs lbbest = {j_lb_full:.4f}  "
        f"(errs tab={e_tab}  lbbest={e_lb})")

    test_greedy = np.load(ART / "test_greedy_blend.npy")
    test_nonrule = np.load(ART / "test_xgb_nonrule.npy")
    test_lbbest = log_blend2(test_nonrule, test_greedy, 0.15)
    lp_lb = np.log(np.clip(oof_lbbest, 1e-9, 1.0))
    lbbest_tuned_oof = float(balanced_accuracy_score(
        y, (lp_lb + greedy_bias).argmax(axis=1)
    ))

    log(f"baselines (fixed-bias, reproduced):")
    log(f"  greedy         OOF = {greedy_tuned_oof:.5f}")
    log(f"  greedy+nonrule OOF = {lbbest_tuned_oof:.5f}")

    results = {
        "aborted": False,
        "subsample": SUBSAMPLE,
        "n_estimators": N_ESTIMATORS,
        "fold_wall_times": fold_times,
        "argmax_bal": float(argmax_bal),
        "tuned_bal": float(tuned_bal),
        "bias_tabpfn": bias_tabpfn.tolist(),
        "full_oof_jaccard_vs_greedy": j_g_full,
        "full_oof_jaccard_vs_lbbest": j_lb_full,
        "greedy_tuned_oof": greedy_tuned_oof,
        "lbbest_tuned_oof": lbbest_tuned_oof,
        "sweep_vs_greedy": [],
        "sweep_vs_lbbest": [],
    }

    grid = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    log("sweep vs greedy (fixed greedy bias)")
    for alpha in grid:
        blend = log_blend2(oof, oof_greedy, alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + greedy_bias).argmax(axis=1))
        results["sweep_vs_greedy"].append({
            "alpha": alpha, "oof": float(ba),
            "delta": float(ba - greedy_tuned_oof),
        })
        log(f"  alpha={alpha:.3f}  OOF={ba:.5f}  Δ={ba-greedy_tuned_oof:+.5f}")

    log("sweep vs LB-best greedy+nonrule (fixed greedy bias)")
    for alpha in grid:
        blend = log_blend2(oof, oof_lbbest, alpha)
        lp = np.log(np.clip(blend, 1e-9, 1.0))
        ba = balanced_accuracy_score(y, (lp + greedy_bias).argmax(axis=1))
        results["sweep_vs_lbbest"].append({
            "alpha": alpha, "oof": float(ba),
            "delta": float(ba - lbbest_tuned_oof),
        })
        log(f"  alpha={alpha:.3f}  OOF={ba:.5f}  Δ={ba-lbbest_tuned_oof:+.5f}")

    best_g = max(results["sweep_vs_greedy"], key=lambda d: d["oof"])
    best_lb = max(results["sweep_vs_lbbest"], key=lambda d: d["oof"])
    results["best_vs_greedy"] = best_g
    results["best_vs_lbbest"] = best_lb
    log(f"best vs greedy:  alpha={best_g['alpha']}  OOF={best_g['oof']:.5f}  "
        f"Δ={best_g['delta']:+.5f}")
    log(f"best vs lbbest:  alpha={best_lb['alpha']}  OOF={best_lb['oof']:.5f}  "
        f"Δ={best_lb['delta']:+.5f}")

    if best_lb["alpha"] > 0 and best_lb["delta"] >= 2e-4:
        blend_test = log_blend2(test_pred, test_lbbest, best_lb["alpha"])
        lp_test = np.log(np.clip(blend_test, 1e-9, 1.0))
        preds = (lp_test + greedy_bias).argmax(axis=1)
        sub_path = OUT / "submission_lbbest_tabpfn_blend.csv"
        pd.DataFrame({ID: te[ID], TARGET: [IDX2CLS[i] for i in preds]}).to_csv(
            sub_path, index=False
        )
        log(f"wrote {sub_path}  (OOF lift +{best_lb['delta']:.5f} — LB probe candidate)")
        results["action"] = "ready_to_submit"
        results["submission_path"] = str(sub_path)
    elif best_lb["alpha"] > 0 and best_lb["delta"] > 0:
        log(f"OOF lift +{best_lb['delta']:.5f} below +0.0002 threshold — null.")
        results["action"] = "below_threshold_no_submit"
    else:
        log("sweep strictly <= baseline — null result; no submission.")
        results["action"] = "no_submission"

    with open(ART / "tabpfn_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"wrote {ART}/tabpfn_results.json")


if __name__ == "__main__":
    main()
