"""W13 — Train-OOF wrongness predictor XGB.

Reconstruct 4b's argmax on TRAIN OOF, then train a small XGB to predict
"is 4b wrong on this row?" using:
  - 4b's class onehot
  - 14-bank entropy + majority agreement
  - dgp_score, distance features
  - LB-validated submission disagreement counts

Apply to test. Where P(4b wrong) > τ, override 4b with 14-bank majority.

Mechanism-novel: learned per-row wrongness detector specific to 4b's
prediction surface. None of the prior router/which-model variants
trained on 4b directly — they were on v1/B.

Build TRAIN-side 4b proxy:
  - V1 = oof_sklearn_rf_meta_natural.npy (now v2 prob, near-v1 at v1 bias)
  - B = V1 with {raw_oof, tier1b_oof} k=2 unanimous override
  - 4b = B with bagged_v1' + {raw, tier1b} unanimous + 14-bank-majority
    selective override (need TRAIN-side 14-bank for that)

For 14-bank TRAIN: aggregate argmax across the 14 LB-validated components'
TRAIN OOF predictions. We have all the OOF arrays.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, IDX2CLS  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def normed(a): return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def csv_argmax(name: str) -> np.ndarray:
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map({"Low": 0, "Medium": 1, "High": 2}).to_numpy(dtype=np.int8)


def main():
    log("=== W13: train-OOF wrongness predictor for 4b ===")

    # Load TRAIN labels
    train = pd.read_csv(DATA / "train.csv")
    y_train = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).to_numpy()
    n_tr = len(y_train)
    log(f"  n_train = {n_tr}")

    # Build dist features for train
    train_d = add_distance_features(train)
    dist_cols = ["sm_dist", "rf_dist", "tc_dist", "ws_dist",
                 "sm_abs", "rf_abs", "tc_abs", "ws_abs",
                 "min_boundary_dist", "min_axis_abs",
                 "score_dist_low_mid", "score_dist_mid_high",
                 "dgp_score", "rule_pred"]
    X_dist_tr = train_d[dist_cols].to_numpy(dtype=np.float32)

    # 14-bank TRAIN: pick 14 LB-validated component OOFs
    # Use the same 14 components implied by stability_test (we don't know exact list,
    # but pick representative LB-validated ones)
    bank_names = [
        "sklearn_rf_meta_natural",  # v1 RF natural
        "sklearn_rf_meta_natural_a1lgbm",
        "sklearn_rf_meta_natural_r10_with_tier1b",
        "rawashishsin_2600",
        "tier1b_greedy_meta",  # may not exist as oof
        "recipe_full_te",
        "recipe_pseudolabel",
        "recipe_pseudolabel_seed7labeler",
        "realmlp",
        "xgb_nonrule",
        "xgb_metastack",
        "recipe_full_te_catboost_natural",
        "recipe_full_te_catboost",
        "lgbm_meta_natural",
    ]
    bank_oof = []
    for name in bank_names:
        p = ART / f"oof_{name}.npy"
        if p.exists():
            arr = normed(np.load(p).astype(np.float32))
            if arr.shape == (n_tr, 3):
                bank_oof.append(arr.argmax(axis=1).astype(np.int8))
                log(f"  bank: {name}")
    log(f"  loaded {len(bank_oof)} bank components")

    # Compute TRAIN 14-bank majority + agreement
    bank_arr = np.stack(bank_oof, axis=1)  # (n_tr, n_bank)
    counts = np.zeros((n_tr, 3), dtype=np.int32)
    for c in range(3):
        counts[:, c] = (bank_arr == c).sum(axis=1)
    bank_maj_tr = counts.argmax(axis=1)
    bank_agree_tr = counts.max(axis=1) / len(bank_oof)

    log(f"  bank_majority train class counts: L={int((bank_maj_tr==0).sum())} M={int((bank_maj_tr==1).sum())} H={int((bank_maj_tr==2).sum())}")

    # Build "4b proxy" on TRAIN: simulate 4b's mechanism on TRAIN OOF
    # B = V1 + {raw_oof, tier1b_oof} k=2 unanimous override
    # 4b = B + selective bagged_v1' + {raw, t1b} unan + 14-bank-maj filter
    v1_oof = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    raw_oof = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    # tier1b: reconstruct via tier1b_helpers if available
    tier1b_oof = None
    if (ART / "oof_xgb_metastack.npy").exists():
        # tier1b 4-stack = lb3 + xgb_metastack_iso × 0.30
        # Approximate: use xgb_metastack as proxy
        tier1b_oof = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))

    if tier1b_oof is None:
        log("  WARN: tier1b 4-stack OOF not available, using xgb_metastack as proxy")
        return

    V1_BIAS = np.array([0.4324, 0.8689, 3.2008])
    v1_argmax = (np.log(np.clip(v1_oof, 1e-9, 1.0)) + V1_BIAS).argmax(1)
    raw_argmax = (np.log(np.clip(raw_oof, 1e-9, 1.0)) + np.array([1.6324, 1.7689, 3.0008])).argmax(1)
    # tier1b proxy via own bias
    t1b_argmax = (np.log(np.clip(tier1b_oof, 1e-9, 1.0)) + np.array([1.4324, 1.4689, 3.4008])).argmax(1)

    # B on TRAIN: V1 with {raw, t1b} k=2 unanimous
    raw_eq_t1b = (raw_argmax == t1b_argmax)
    diff_v1 = raw_argmax != v1_argmax
    b_override = raw_eq_t1b & diff_v1
    b_train = v1_argmax.copy()
    b_train[b_override] = raw_argmax[b_override]
    log(f"  B on TRAIN: {b_override.sum()} overrides applied to V1")

    # 4b proxy on TRAIN: B + (bagged_v1 ≈ v1 since we don't have all 4 RF on train) + bank-maj filter
    # Approximation: use v1's argmax as bagged proxy + 14-bank majority filter
    fb_override_mask = (
        (b_train != bank_maj_tr) &        # B differs from 14-bank majority
        raw_eq_t1b &                       # {raw, t1b} unanimous
        (raw_argmax == bank_maj_tr) &     # consensus on bank-maj's class
        (b_train != raw_argmax)            # B != consensus class
    )
    fb_train = b_train.copy()
    fb_train[fb_override_mask] = raw_argmax[fb_override_mask]
    log(f"  4b proxy on TRAIN: {fb_override_mask.sum()} additional overrides on top of B")

    # Build target: is 4b wrong on this train row?
    fb_wrong = (fb_train != y_train).astype(np.int8)
    log(f"  4b wrongness rate on TRAIN: {fb_wrong.mean():.4f} ({int(fb_wrong.sum())} of {n_tr})")

    # Build features
    fb_onehot = np.zeros((n_tr, 3), dtype=np.float32)
    fb_onehot[np.arange(n_tr), fb_train] = 1.0
    bank_maj_onehot = np.zeros((n_tr, 3), dtype=np.float32)
    bank_maj_onehot[np.arange(n_tr), bank_maj_tr] = 1.0

    # Disagreement count: 4b vs 14-bank majority
    fb_vs_bank = (fb_train != bank_maj_tr).astype(np.float32)

    # Per-row entropy of 14-bank vote distribution
    eps = 1e-9
    p_bank = counts / np.clip(counts.sum(axis=1, keepdims=True), eps, None)
    bank_entropy = -np.sum(p_bank * np.log(np.clip(p_bank, eps, 1.0)), axis=1).astype(np.float32)

    X_tr = np.concatenate([
        fb_onehot,             # 3
        bank_maj_onehot,        # 3
        bank_agree_tr.reshape(-1, 1).astype(np.float32),  # 1
        bank_entropy.reshape(-1, 1),  # 1
        fb_vs_bank.reshape(-1, 1),  # 1
        X_dist_tr,              # 14
    ], axis=1).astype(np.float32)
    log(f"  X_tr shape: {X_tr.shape}")

    # 5-fold OOF train wrongness predictor
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    p_wrong_oof = np.zeros(n_tr, dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_tr, fb_wrong), 1):
        clf = xgb.XGBClassifier(
            objective="binary:logistic", max_depth=4, learning_rate=0.08,
            n_estimators=300, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=1.0, reg_lambda=1.0, random_state=42,
            tree_method="hist", n_jobs=-1, verbosity=0,
        )
        clf.fit(X_tr[tr_idx], fb_wrong[tr_idx],
                eval_set=[(X_tr[va_idx], fb_wrong[va_idx])], verbose=False)
        p_wrong_oof[va_idx] = clf.predict_proba(X_tr[va_idx])[:, 1]
        log(f"    fold {fold}: best_iter={clf.best_iteration}")

    # AUC of wrongness predictor
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(fb_wrong, p_wrong_oof)
    log(f"  wrongness predictor OOF AUC: {auc:.4f}")

    # Diagnostic: at various thresholds, what's the override precision?
    log("\nThreshold analysis:")
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = p_wrong_oof > tau
        n = mask.sum()
        if n == 0:
            continue
        # Where we'd override 4b with bank_maj, is bank_maj actually the truth?
        override_correct = ((bank_maj_tr[mask] == y_train[mask]) & (fb_wrong[mask] == 1)).sum()
        override_wrong = ((bank_maj_tr[mask] == y_train[mask]) == False).sum() if mask.any() else 0
        # More careful: of the n overrides, how many flips would be net-correct?
        # If 4b is wrong (fb_train != y) AND bank_maj == y → +1 (correct override)
        # If 4b is wrong (fb_train != y) AND bank_maj != y → 0 (still wrong, but different class)
        # If 4b is right (fb_train == y) AND bank_maj != y → -1 (we made it wrong)
        new_pred = fb_train.copy()
        new_pred[mask] = bank_maj_tr[mask]
        correct_before = (fb_train == y_train).sum()
        correct_after = (new_pred == y_train).sum()
        log(f"  τ={tau:.2f}: {n} overrides, train_correct {correct_before}→{correct_after} (Δ {correct_after-correct_before:+d})")

    # Now train final XGB on full TRAIN, apply to TEST
    # Build test features same way
    log("\nBuilding test features and predicting...")
    test = pd.read_csv(DATA / "test.csv")
    test_d = add_distance_features(test)
    X_dist_te = test_d[dist_cols].to_numpy(dtype=np.float32)

    fb_test = csv_argmax("submission_idea4b_selective_override")
    bank_maj_te = np.load(ART / "stability_test_majority.npy")
    bank_agree_te = np.load(ART / "stability_test_agreement.npy")

    # Bank entropy on test: don't have full bank counts, approximate using stability_test_agreement
    # (agree = max_prob of bank vote distribution)
    # entropy ≈ -log(max) but really should compute from full vote dist
    bank_entropy_te = -np.log(np.clip(bank_agree_te, eps, 1.0))

    fb_te_onehot = np.zeros((len(fb_test), 3), dtype=np.float32)
    fb_te_onehot[np.arange(len(fb_test)), fb_test] = 1.0
    bm_te_onehot = np.zeros((len(bank_maj_te), 3), dtype=np.float32)
    bm_te_onehot[np.arange(len(bank_maj_te)), bank_maj_te] = 1.0
    fb_vs_bank_te = (fb_test != bank_maj_te).astype(np.float32)

    X_te = np.concatenate([
        fb_te_onehot,
        bm_te_onehot,
        bank_agree_te.reshape(-1, 1).astype(np.float32),
        bank_entropy_te.reshape(-1, 1).astype(np.float32),
        fb_vs_bank_te.reshape(-1, 1),
        X_dist_te,
    ], axis=1).astype(np.float32)
    log(f"  X_te shape: {X_te.shape}")

    # Train final on full
    clf_final = xgb.XGBClassifier(
        objective="binary:logistic", max_depth=4, learning_rate=0.08,
        n_estimators=300, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=1.0, reg_lambda=1.0, random_state=42,
        tree_method="hist", n_jobs=-1, verbosity=0,
    )
    clf_final.fit(X_tr, fb_wrong, verbose=False)
    p_wrong_test = clf_final.predict_proba(X_te)[:, 1]
    log(f"  test p_wrong percentiles: p50={np.percentile(p_wrong_test, 50):.4f}, p90={np.percentile(p_wrong_test, 90):.4f}, p99={np.percentile(p_wrong_test, 99):.4f}")

    # Apply override at multiple thresholds
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        mask = p_wrong_test > tau
        n = mask.sum()
        if n == 0:
            continue
        new_pred = fb_test.copy()
        new_pred[mask] = bank_maj_te[mask]
        # Direction breakdown
        LMH = ["L", "M", "H"]
        dirs = {}
        for fr in range(3):
            for to in range(3):
                if fr == to: continue
                k = int(((fb_test == fr) & (new_pred == to)).sum())
                if k > 0:
                    dirs[f"{LMH[fr]}->{LMH[to]}"] = k
        h_a = int(((fb_test != 2) & (new_pred == 2)).sum())
        h_r = int(((fb_test == 2) & (new_pred != 2)).sum())
        log(f"  τ={tau:.2f}: {n} overrides, dirs={dirs}, net_H={h_a-h_r:+d}")

        # Save candidate
        test_ids = test["id"].to_numpy()
        sub = pd.DataFrame({
            "id": test_ids,
            "Irrigation_Need": pd.Series(new_pred).map(IDX2CLS),
        })
        out_csv = SUB / f"submission_W13_wrong_pred_tau{int(tau*100)}.csv"
        sub.to_csv(out_csv, index=False)

    out_json = ART / "W13_wrongness_predictor_results.json"
    out_json.write_text(json.dumps({
        "wrongness_oof_auc": float(auc),
        "n_bank_components": len(bank_oof),
        "fb_wrong_rate_train": float(fb_wrong.mean()),
        "fb_train_overrides_proxy": int(fb_override_mask.sum()),
    }, indent=2, default=str))
    log(f"\n=== summary written to {out_json} ===")


if __name__ == "__main__":
    main()
