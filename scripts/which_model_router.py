"""Idea 4: Per-row "which-model-is-right" classifier.

Mechanism:
  - On rows where v1 (LB 0.98129) and rawashishsin v3 (LB 0.98109) disagree
    on argmax, ground truth tells us which side wins.
  - Train a classifier on OOF disagreement rows predicting P(raw_wins | x).
  - At test time, route disagreed-on rows: keep v1 by default, swap to
    rawashishsin's argmax where P(raw_wins) > tau.

Distinct from prior overrides because the base rate is roughly 50/50
(both models are ~98% accurate; on disagreements either could be right).
No rare-class precision floor.

Outputs:
  scripts/artifacts/which_model_router_results.json
  scripts/artifacts/oof_router_predictions.npy
  scripts/artifacts/test_router_decisions.npy
  submissions/submission_router_*.csv (multiple tau levels)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, fast_bal_acc  # noqa: E402

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
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def build_features(v1_p, raw_p, dist_df):
    """Engineered features for the which-model task. ~14 columns.

    Designed to expose the calibration disagreement geometry without
    adding capacity for the model to memorize raw class identity.
    """
    n = len(v1_p)
    eps = 1e-9
    v1_max = v1_p.max(1)
    raw_max = raw_p.max(1)
    v1_arg = v1_p.argmax(1)
    raw_arg = raw_p.argmax(1)

    # Entropies
    v1_H = -(v1_p * np.log(np.clip(v1_p, eps, 1.0))).sum(1)
    raw_H = -(raw_p * np.log(np.clip(raw_p, eps, 1.0))).sum(1)

    # Per-class log-prob ratio
    log_ratio = safelog(v1_p) - safelog(raw_p)

    feats = np.column_stack([
        v1_max, raw_max, v1_max - raw_max,
        v1_H, raw_H, v1_H - raw_H,
        v1_arg.astype(np.float32), raw_arg.astype(np.float32),
        (v1_arg == raw_arg).astype(np.float32),  # = 0 on disagreement rows (constant)
        log_ratio[:, 0], log_ratio[:, 1], log_ratio[:, 2],
        dist_df["dgp_score"].to_numpy(dtype=np.float32),
        dist_df["min_axis_abs"].to_numpy(dtype=np.float32) if "min_axis_abs" in dist_df.columns
            else dist_df[["sm_abs", "rf_abs", "tc_abs", "ws_abs"]].min(axis=1).to_numpy(dtype=np.float32),
        dist_df["sm_abs"].to_numpy(dtype=np.float32),
        dist_df["rf_abs"].to_numpy(dtype=np.float32),
    ]).astype(np.float32)
    feat_names = ["v1_max", "raw_max", "max_gap",
                  "v1_H", "raw_H", "H_gap",
                  "v1_arg", "raw_arg", "agree",
                  "log_r_L", "log_r_M", "log_r_H",
                  "dgp_score", "min_abs", "sm_abs", "rf_abs"]
    return feats, feat_names


def main():
    log("=== Idea 4: which-model router ===")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values

    # Load v1 LB-best (LB 0.98129) and rawashishsin v3 (LB 0.98109)
    v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy").astype(np.float32)
    raw_oof = np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32)
    raw_test = np.load(ART / "test_rawashishsin_2600.npy").astype(np.float32)

    # v1's tuned bias was [0.43, 0.87, 3.20] (CLAUDE.md)
    # rawashishsin v3's tuned bias was [-1.36, -1.19, 0.00] but reproduces a different
    # convention; we re-tune both on full OOF for apples-to-apples.
    from common import tune_log_bias
    prior = np.bincount(y, minlength=3) / len(y)
    bias_v1, _ = tune_log_bias(v1_oof, y, prior)
    bias_raw, _ = tune_log_bias(raw_oof, y, prior)
    log(f"  v1 bias  = {bias_v1.round(4).tolist()}")
    log(f"  raw bias = {bias_raw.round(4).tolist()}")

    v1_logits_oof = safelog(v1_oof) + bias_v1
    v1_logits_test = safelog(v1_test) + bias_v1
    raw_logits_oof = safelog(raw_oof) + bias_raw
    raw_logits_test = safelog(raw_test) + bias_raw

    v1_oof_p = np.exp(v1_logits_oof - v1_logits_oof.max(axis=1, keepdims=True))
    v1_oof_p /= v1_oof_p.sum(axis=1, keepdims=True)
    v1_test_p = np.exp(v1_logits_test - v1_logits_test.max(axis=1, keepdims=True))
    v1_test_p /= v1_test_p.sum(axis=1, keepdims=True)
    raw_oof_p = np.exp(raw_logits_oof - raw_logits_oof.max(axis=1, keepdims=True))
    raw_oof_p /= raw_oof_p.sum(axis=1, keepdims=True)
    raw_test_p = np.exp(raw_logits_test - raw_logits_test.max(axis=1, keepdims=True))
    raw_test_p /= raw_test_p.sum(axis=1, keepdims=True)

    v1_arg_oof = v1_oof_p.argmax(1)
    raw_arg_oof = raw_oof_p.argmax(1)
    v1_arg_test = v1_test_p.argmax(1)
    raw_arg_test = raw_test_p.argmax(1)

    dis_oof = (v1_arg_oof != raw_arg_oof)
    dis_test = (v1_arg_test != raw_arg_test)
    log(f"OOF disagreement: {dis_oof.sum():,} / {len(y):,} rows ({100*dis_oof.mean():.3f}%)")
    log(f"Test disagreement: {dis_test.sum():,} / {len(test_ids):,} rows ({100*dis_test.mean():.3f}%)")

    # Standalone macro-recall verification
    log(f"  v1 standalone bal_acc: {balanced_accuracy_score(y, v1_arg_oof):.5f}")
    log(f"  raw standalone bal_acc: {balanced_accuracy_score(y, raw_arg_oof):.5f}")

    # On disagreement rows: which side wins?
    dis_idx = np.where(dis_oof)[0]
    y_dis = y[dis_idx]
    v1_correct = (v1_arg_oof[dis_idx] == y_dis)
    raw_correct = (raw_arg_oof[dis_idx] == y_dis)
    both_wrong = (~v1_correct) & (~raw_correct)

    n_v1 = int(v1_correct.sum())
    n_raw = int(raw_correct.sum())
    n_both = int(both_wrong.sum())
    log(f"On {len(dis_idx)} disagreement rows:")
    log(f"  v1 wins:    {n_v1:>6} ({100*n_v1/len(dis_idx):.1f}%)")
    log(f"  raw wins:   {n_raw:>6} ({100*n_raw/len(dis_idx):.1f}%)")
    log(f"  both wrong: {n_both:>6} ({100*n_both/len(dis_idx):.1f}%)")

    # Build features for full data, then slice to disagreement rows
    log("building features for full train + test")
    tr_d = add_distance_features(train)
    te_d = add_distance_features(test)
    for d in (tr_d, te_d):
        d["min_axis_abs"] = d[["sm_abs", "rf_abs", "tc_abs", "ws_abs"]].min(axis=1)

    X_full_tr, feat_names = build_features(v1_oof_p, raw_oof_p, tr_d)
    X_full_te, _ = build_features(v1_test_p, raw_test_p, te_d)
    log(f"  feature matrix: tr={X_full_tr.shape}  te={X_full_te.shape}")

    # Binary target on disagreement rows: 1 = rawashishsin wins (route to raw)
    # On both_wrong rows we don't know which is "right" — can either be
    # encoded as 0 (no signal) or excluded. Try both, prefer excluding.
    target_dis = raw_correct.astype(np.int32)  # 1 = raw_wins, 0 = v1_wins or both_wrong
    keep_mask = ~both_wrong  # exclude ambiguous both-wrong rows
    log(f"keeping {keep_mask.sum()}/{len(dis_idx)} unambiguous disagreement rows")

    # Reduce to clean disagreement subset
    X_dis = X_full_tr[dis_idx][keep_mask]
    y_dis_clean = target_dis[keep_mask]
    log(f"clean disagreement set: {len(y_dis_clean)} rows  raw_wins_rate={y_dis_clean.mean():.3f}")

    # 5-fold OOF on disagreement rows. Use GBM (works well at this scale)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_router_dis = np.zeros(len(y_dis_clean), dtype=np.float32)
    fold_aucs = []
    test_router_pred = np.zeros(len(test_ids), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dis, y_dis_clean), 1):
        t0 = time.time()
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=SEED,
        )
        clf.fit(X_dis[tr_idx], y_dis_clean[tr_idx])
        oof_router_dis[va_idx] = clf.predict_proba(X_dis[va_idx])[:, 1]
        # Apply each fold's classifier to test disagreement rows; average.
        test_router_pred += clf.predict_proba(X_full_te)[:, 1] / 5
        auc = roc_auc_score(y_dis_clean[va_idx], oof_router_dis[va_idx])
        fold_aucs.append(float(auc))
        log(f"  fold {fold}/5  AUC={auc:.4f}  wall={time.time()-t0:.1f}s")

    overall_auc = float(roc_auc_score(y_dis_clean, oof_router_dis))
    log(f"=== overall router AUC = {overall_auc:.4f}  (base rate = {y_dis_clean.mean():.3f}) ===")

    # Save router predictions (full-OOF: 0 for non-disagreement rows, prob for disagreement)
    full_oof_router = np.zeros(len(y), dtype=np.float32)
    dis_clean_idx_in_full = dis_idx[keep_mask]
    full_oof_router[dis_clean_idx_in_full] = oof_router_dis
    np.save(ART / "oof_router_predictions.npy", full_oof_router)
    np.save(ART / "test_router_decisions.npy", test_router_pred)

    # Decision rule sweep: tau ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75}
    # On test rows where v1.argmax != raw.argmax AND P(raw_wins) > tau:
    #   route to rawashishsin's argmax. Otherwise keep v1.
    log("=== test-side routing sweep ===")
    sweep = []
    for tau in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        # On test: keep v1 by default, swap on disagreement rows where router
        # predicts raw wins
        new_argmax = v1_arg_test.copy()
        route_mask = dis_test & (test_router_pred > tau)
        new_argmax[route_mask] = raw_arg_test[route_mask]
        n_routed = int(route_mask.sum())

        # Validate via OOF: same logic with full_oof_router as P(raw_wins)
        new_oof_argmax = v1_arg_oof.copy()
        oof_route_mask = dis_oof & (full_oof_router > tau)
        new_oof_argmax[oof_route_mask] = raw_arg_oof[oof_route_mask]
        new_bal = balanced_accuracy_score(y, new_oof_argmax)
        v1_bal = balanced_accuracy_score(y, v1_arg_oof)
        delta = new_bal - v1_bal

        # Per-class shift
        pcr_v1 = per_class_recall(y, v1_arg_oof)
        pcr_new = per_class_recall(y, new_oof_argmax)
        pcr_delta = (pcr_new - pcr_v1).tolist()

        # Test-side class delta vs v1
        class_delta = [int((new_argmax == k).sum() - (v1_arg_test == k).sum()) for k in range(3)]

        sweep.append(dict(
            tau=tau,
            n_routed_oof=int(oof_route_mask.sum()),
            n_routed_test=n_routed,
            new_bal_oof=float(new_bal),
            delta_oof=float(delta),
            pcr_delta_oof=pcr_delta,
            test_class_delta=class_delta,
        ))
        log(f"  tau={tau:.2f}  routed_oof={int(oof_route_mask.sum())}  routed_test={n_routed}  "
            f"OOF Δ={delta:+.6f}  pcr_dL={pcr_delta[0]:+.5f} dM={pcr_delta[1]:+.5f} dH={pcr_delta[2]:+.5f}")

    # Pick best by OOF delta with PCR guardrail (-5e-4 floor each class)
    best = None
    for s in sweep:
        passes_g2 = all(d >= -5e-4 for d in s["pcr_delta_oof"])
        if passes_g2 and (best is None or s["delta_oof"] > best["delta_oof"]):
            best = s
    if best is not None:
        log(f"\nBest gate-pass tau={best['tau']:.2f}  OOF Δ={best['delta_oof']:+.6f}  routed_test={best['n_routed_test']}")
        # Build candidate submission at best tau
        tau = best["tau"]
        new_argmax = v1_arg_test.copy()
        route_mask = dis_test & (test_router_pred > tau)
        new_argmax[route_mask] = raw_arg_test[route_mask]
        labels = [IDX2CLS[i] for i in new_argmax]
        sub_path = SUB / f"submission_router_tau{int(tau*100):03d}.csv"
        pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(sub_path, index=False)
        log(f"wrote {sub_path}")

    # Always emit best-OOF-Δ submission (regardless of guardrail)
    best_unconstrained = max(sweep, key=lambda s: s["delta_oof"])
    if best is None or best_unconstrained["tau"] != best["tau"]:
        tau_u = best_unconstrained["tau"]
        new_argmax = v1_arg_test.copy()
        route_mask = dis_test & (test_router_pred > tau_u)
        new_argmax[route_mask] = raw_arg_test[route_mask]
        labels = [IDX2CLS[i] for i in new_argmax]
        sub_path = SUB / f"submission_router_tau{int(tau_u*100):03d}_unconstrained.csv"
        pd.DataFrame({"id": test_ids, TARGET: labels}).to_csv(sub_path, index=False)
        log(f"wrote {sub_path} (unconstrained best Δ={best_unconstrained['delta_oof']:+.6f})")

    summary = dict(
        oof_disagreement_rows=int(dis_oof.sum()),
        test_disagreement_rows=int(dis_test.sum()),
        v1_bias=bias_v1.tolist(), raw_bias=bias_raw.tolist(),
        v1_oof_bal=float(balanced_accuracy_score(y, v1_arg_oof)),
        raw_oof_bal=float(balanced_accuracy_score(y, raw_arg_oof)),
        n_v1_wins=n_v1, n_raw_wins=n_raw, n_both_wrong=n_both,
        clean_disagreement_n=int(len(y_dis_clean)),
        raw_wins_rate=float(y_dis_clean.mean()),
        fold_aucs=fold_aucs,
        overall_auc=overall_auc,
        feat_names=feat_names,
        tau_sweep=sweep,
        best_gate_pass=best,
    )
    out_p = ART / "which_model_router_results.json"
    with open(out_p, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
