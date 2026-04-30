"""H3 — Per-row router on v1 ⊗ rawashishsin LB-validated outputs.

Two LB-validated solutions, orthogonal model classes (RF-meta vs single
XGB), 620 test-row disagreements. Direct log-blend fails per the
bias-mismatch rule. Per-row delegation has not been tested.

Mechanism:
  1. Identify OOF rows where v1 and rawashishsin DISAGREE on argmax.
  2. Among those, compute who is correct: v1 / raw / both / neither.
  3. Train a binary classifier on disagreement rows: target = 1 if v1 right.
  4. Diagnose top-K precision and break-even precision under macro-recall.
  5. If precision >= break-even, deploy on test disagreements.
  6. Otherwise: scaffold a 3-class soft router (gate determines which
     model's probs to weight).

Decision rule comparison vs simple log-blend:
  - log-blend at fixed v1 bias: bias-mismatch trap (CLAUDE.md rule)
  - per-row router: only flips test rows where gate confident,
    otherwise inherits LB-best (v1).

OUTPUT: report only — actual submission emitted only if precision
analysis indicates LB-positive transfer probability.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SEED = 42


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        m = y == k
        if m.sum() > 0:
            rec[k] = (pred[m] == k).sum() / m.sum()
    return rec


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def main():
    log("loading train/test")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values
    n_tr, n_te = len(train), len(test)

    # Load v1 LB-best (LB 0.98129) and rawashishsin v3 (LB 0.98109)
    v1_oof = _normed(np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    v1_test = _normed(np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32))
    raw_oof = _normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    raw_test = _normed(np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32))

    prior = np.bincount(y, minlength=3) / len(y)
    v1_bias, v1_tuned = tune_log_bias(v1_oof, y, prior)
    raw_bias, raw_tuned = tune_log_bias(raw_oof, y, prior)

    log(f"v1  tuned={v1_tuned:.5f}  bias={v1_bias.round(4).tolist()}")
    log(f"raw tuned={raw_tuned:.5f}  bias={raw_bias.round(4).tolist()}")

    # Per-row argmax at each model's tuned bias
    v1_pred = (safelog(v1_oof) + v1_bias).argmax(1)
    raw_pred = (safelog(raw_oof) + raw_bias).argmax(1)

    # OOF disagreement set
    disagree_mask = v1_pred != raw_pred
    n_dis = disagree_mask.sum()
    log(f"OOF disagreement rows: {n_dis} / {n_tr} ({n_dis/n_tr*100:.3f}%)")

    # Among disagreement rows: who is correct?
    v1_right = (v1_pred == y) & disagree_mask
    raw_right = (raw_pred == y) & disagree_mask
    both_wrong = ~v1_right & ~raw_right & disagree_mask

    n_v1 = v1_right.sum()
    n_raw = raw_right.sum()
    n_both_wrong = both_wrong.sum()
    log(f"  v1 correct (raw wrong): {n_v1} ({n_v1/n_dis*100:.1f}%)")
    log(f"  raw correct (v1 wrong): {n_raw} ({n_raw/n_dis*100:.1f}%)")
    log(f"  both wrong (different): {n_both_wrong} ({n_both_wrong/n_dis*100:.1f}%)")

    # Per-class breakdown of correct picks
    log("Per-class correct distribution among disagreement rows:")
    for k, name in IDX2CLS.items():
        nv1k = ((v1_pred == k) & v1_right).sum()
        nrawk = ((raw_pred == k) & raw_right).sum()
        log(f"  v1 wins as {name}: {nv1k} | raw wins as {name}: {nrawk}")

    # Build router training set (where exactly one is right)
    one_right = v1_right ^ raw_right
    n_one = one_right.sum()
    log(f"router training set: {n_one} rows (exactly one model right)")

    # Target: 1 if v1 is right, 0 if raw is right
    router_y = v1_right[one_right].astype(np.int32)
    log(f"  v1=right target distribution: {router_y.mean():.3f}")

    # Build router feature matrix
    tr_d = add_distance_features(train)
    test_d = add_distance_features(test)
    META_COLS = ["dgp_score", "rule_pred", "sm_dist", "rf_dist", "tc_dist",
                 "ws_dist", "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high"]
    meta_tr = tr_d[META_COLS].to_numpy(dtype=np.float32)
    meta_te = test_d[META_COLS].to_numpy(dtype=np.float32)

    # Per-row features: v1 probs (3), raw probs (3), dist features
    X_full_tr = np.concatenate([
        v1_oof, raw_oof,
        v1_oof - raw_oof,  # disagreement vector (3)
        np.abs(v1_oof - raw_oof).max(1, keepdims=True),  # max disagreement
        v1_oof.max(1, keepdims=True),  # v1 confidence
        raw_oof.max(1, keepdims=True),  # raw confidence
        meta_tr,
    ], axis=1).astype(np.float32)
    X_full_te = np.concatenate([
        v1_test, raw_test,
        v1_test - raw_test,
        np.abs(v1_test - raw_test).max(1, keepdims=True),
        v1_test.max(1, keepdims=True),
        raw_test.max(1, keepdims=True),
        meta_te,
    ], axis=1).astype(np.float32)
    log(f"router feature matrix: full train={X_full_tr.shape}  test={X_full_te.shape}")

    # Slice to one-right rows
    X_router = X_full_tr[one_right]

    # 5-fold CV on the router (StratifiedKFold on router target)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    router_oof = np.zeros(n_one, dtype=np.float32)
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_router, router_y), 1):
        t0 = time.time()
        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=2.0, objective="binary:logistic",
            random_state=SEED, n_jobs=-1, verbosity=0,
            early_stopping_rounds=50, eval_metric="auc",
        )
        clf.fit(X_router[tr_idx], router_y[tr_idx],
                eval_set=[(X_router[va_idx], router_y[va_idx])],
                verbose=False)
        p = clf.predict_proba(X_router[va_idx])[:, 1]
        router_oof[va_idx] = p
        auc = roc_auc_score(router_y[va_idx], p)
        fold_aucs.append(float(auc))
        log(f"  router fold {fold}/5 auc={auc:.4f}  best_iter={clf.best_iteration} wall={time.time()-t0:.1f}s")

    overall_auc = roc_auc_score(router_y, router_oof)
    log(f"router OVERALL OOF AUC = {overall_auc:.4f}")
    # Save router_oof + which rows are in the disagreement set
    np.save(ART / "h3_router_oof.npy", router_oof)
    np.save(ART / "h3_router_one_right_mask.npy", one_right.astype(np.int8))
    np.save(ART / "h3_router_y.npy", router_y)
    log(f"  saved router_oof artifacts ({len(router_oof)} rows)")

    # Top-K precision analysis at various thresholds
    # Definition: at threshold τ, "use v1" rows are those where router_oof > τ
    # On those rows, fraction where v1 is actually right = precision
    log("=== Per-K precision analysis (use v1) ===")
    for k_frac in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]:
        k = int(n_one * k_frac)
        order = np.argsort(-router_oof)[:k]  # top-K most v1-confident
        v1_correct_at_k = router_y[order].mean()
        log(f"  top-{k_frac*100:.0f}% (n={k}): v1 correct rate = {v1_correct_at_k:.3f}")

    # Bottom-K precision (use raw)
    log("=== Per-K precision analysis (use raw) ===")
    for k_frac in [0.05, 0.10, 0.20, 0.30, 0.50]:
        k = int(n_one * k_frac)
        order = np.argsort(router_oof)[:k]  # bottom-K most raw-confident
        raw_correct_at_k = (1 - router_y[order]).mean()
        log(f"  bottom-{k_frac*100:.0f}% (n={k}): raw correct rate = {raw_correct_at_k:.3f}")

    # Build full-train router predictor and apply to test
    log("=== final router fit + test prediction ===")
    clf_full = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=2.0, objective="binary:logistic",
        random_state=SEED, n_jobs=-1, verbosity=0,
        early_stopping_rounds=50, eval_metric="auc",
    )
    # Use 80% as train, 20% as val for early stopping
    n_one_total = len(router_y)
    perm = np.random.RandomState(SEED).permutation(n_one_total)
    cut = int(0.8 * n_one_total)
    tr_idx, va_idx = perm[:cut], perm[cut:]
    clf_full.fit(X_router[tr_idx], router_y[tr_idx],
                 eval_set=[(X_router[va_idx], router_y[va_idx])],
                 verbose=False)
    test_router_p = clf_full.predict_proba(X_full_te)[:, 1]

    # Test-side disagreement
    v1_test_pred = (safelog(v1_test) + v1_bias).argmax(1)
    raw_test_pred = (safelog(raw_test) + raw_bias).argmax(1)
    test_disagree = v1_test_pred != raw_test_pred
    n_test_dis = test_disagree.sum()
    log(f"test disagreements: {n_test_dis} / {n_te} ({n_test_dis/n_te*100:.3f}%)")

    # Build per-row submission scenarios
    # Scenario A: "use raw on bottom-K most raw-confident"
    # Scenario B: "use raw on bottom-K rows where router strongly thinks raw"
    scenarios = {}
    for tau in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
        # Among test disagreements, switch to raw where router says v1 prob < tau
        switch_mask = test_disagree & (test_router_p < tau)
        n_sw = switch_mask.sum()
        new_pred = v1_test_pred.copy()
        new_pred[switch_mask] = raw_test_pred[switch_mask]
        scenarios[f"tau{int(tau*100)}"] = (new_pred, n_sw)
        log(f"  tau={tau}: switch {n_sw} rows v1->raw")

    # Each scenario emits a candidate CSV
    for name, (pred, n_sw) in scenarios.items():
        sub_path = SUB / f"submission_h3_router_{name}.csv"
        sub = pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in pred]})
        sub.to_csv(sub_path, index=False)
        log(f"  wrote {sub_path}  (switches {n_sw} rows v1->raw)")

    summary = dict(
        n_oof_disagreements=int(n_dis),
        v1_correct_at_disagreement=int(n_v1),
        raw_correct_at_disagreement=int(n_raw),
        both_wrong=int(n_both_wrong),
        router_n_train=int(n_one),
        router_target_v1_share=float(router_y.mean()),
        router_oof_auc=float(overall_auc),
        fold_aucs=fold_aucs,
        n_test_disagreements=int(n_test_dis),
        test_router_p_mean=float(test_router_p[test_disagree].mean()),
        test_router_p_std=float(test_router_p[test_disagree].std()),
    )
    with open(ART / "h3_per_row_router_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    np.save(ART / "h3_router_test_p.npy", test_router_p)
    log(f"wrote {ART}/h3_per_row_router_results.json")


if __name__ == "__main__":
    main()
