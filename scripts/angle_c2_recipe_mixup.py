"""Angle C2 — within-cell rule-disagreement mixup on RECIPE base.

Three improvements over Angle C v1:
  - recipe FE pipeline (444 cols) instead of dist features (35)
  - drop Medium↔High pairs (Medium-protection)
  - confidence-gate via primary's max_prob < 0.95 on at least one donor
  - K=1 + β(0.2, 0.2) for sharper labels

Plumbing: append mixup rows to `train` BEFORE recipe_features.py runs.
Combos / digits / num_as_cat / FREQ / ORIG-stats are per-row deterministic
on raw cats + nums — mixup rows pass through transparently. OTE per-fold
is fit on tr_idx ∪ kept_mixup (donors both in tr_idx).

5-fold StratifiedKFold(seed=42) on REAL train rows only.

SMOKE=1 → 20k subsample, 2 folds, capped epochs.
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
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402
from angle_c2_helpers import (  # noqa: E402
    NUM_COLS, CAT_COLS, build_pairs_v2, synthesize_mixup,
)
from recipe_features import (  # noqa: E402
    add_cat_pair_combos, add_digit_features, add_freq_features,
    add_lr_formula_logits, add_num_as_cat, add_orig_mean_std,
    add_threshold_flags,
)
from recipe_ote import OrderedTE  # noqa: E402
from tier1b_helpers import ART, BIAS, build_lbbest_stack, iso_cal, normed  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
SEED = 42
N_FOLDS = 2 if SMOKE else 5
K_MIX = int(os.environ.get("K_MIX", "1"))
BETA_A = float(os.environ.get("BETA_A", "0.2"))
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.95"))
DROP_MH = os.environ.get("DROP_MH", "1") == "1"
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "c2")  # "c2" default; "c3" no-gate variant
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}
SUB = Path("submissions"); SUB.mkdir(exist_ok=True, parents=True)


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def reconstruct_primary_max(y: np.ndarray) -> np.ndarray:
    """LB-best 4-stack max-prob OOF on REAL train rows only."""
    s_o, _ = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
    _meta_o, _ = iso_cal(meta_o, meta_o, y)  # iso on full OOF (matches anchor)
    p = log_blend([s_o, _meta_o], np.array([0.7, 0.3]))
    return p.max(1).astype(np.float32)


def main():
    t0 = time.time()
    log(f"angle C2 recipe-mixup. SMOKE={SMOKE} K={K_MIX} β={BETA_A} conf<{CONF_THRESH}")

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    orig = pd.read_csv("data/archive.zip")
    train[TARGET] = train[TARGET].map(CLS_MAP)
    orig[TARGET] = orig[TARGET].map(CLS_MAP)
    test_ids = test["id"].values
    train.drop(columns=["id"], inplace=True)
    test.drop(columns=["id"], inplace=True)
    if SMOKE:
        keep = np.random.RandomState(SEED).choice(len(train), 20_000, replace=False)
        train = train.iloc[keep].reset_index(drop=True)
        test = test.sample(5_000, random_state=SEED).reset_index(drop=True)
        test_ids = test_ids[:len(test)]

    y_real = train[TARGET].to_numpy()
    n_real = len(train)
    log(f"  train={n_real:,}  test={len(test):,}  orig={len(orig):,}")

    # Reconstruct primary OOF max_prob on real train rows.
    if SMOKE:
        # SMOKE: skip primary reconstruction (would need full 630k OOF).
        pmax = np.full(n_real, 0.5, dtype=np.float32)
    else:
        log("reconstructing LB-best primary max_prob")
        pmax = reconstruct_primary_max(y_real)
    log(f"  primary_max: min={pmax.min():.3f} med={np.median(pmax):.3f} "
        f"<{CONF_THRESH}: {int((pmax<CONF_THRESH).sum()):,} ({100*(pmax<CONF_THRESH).mean():.1f}%)")

    # Build pairs + mixup on RAW train (numerics + cats).
    rng = np.random.default_rng(SEED)
    pi, pj = build_pairs_v2(train, y_real, pmax, rng,
                            conf_thresh=CONF_THRESH, drop_mh=DROP_MH,
                            cap_per_cell=2000 if SMOKE else 4000)
    log(f"  built {len(pi):,} within-cell pairs (drop_mh={DROP_MH}, conf<{CONF_THRESH})")
    mixed, mix_y, mix_w, pi_r, pj_r = synthesize_mixup(
        train, y_real, pi, pj, rng, k=K_MIX, beta_a=BETA_A)
    n_mix = len(mixed)
    log(f"  synthesized {n_mix:,} mixup rows")

    # Append mixup to train BEFORE recipe FE.
    mixed[TARGET] = mix_y
    train_aug = pd.concat([train, mixed], axis=0, ignore_index=True)
    log(f"  train_aug shape: {train_aug.shape}  ({n_real:,} real + {n_mix:,} mixup)")

    # Recipe FE — recipe_features functions are deterministic per row.
    log("recipe FE: thresholds + LR logits + combos + digits + num_as_cat + freq + orig_stats")
    for df in (train_aug, test, orig):
        tres = add_threshold_flags(df)
    for df in (train_aug, test, orig):
        logits = add_lr_formula_logits(df)
    combos = add_cat_pair_combos(train_aug, test, orig, CAT_COLS)
    digits = add_digit_features(train_aug, test, orig, NUM_COLS)
    num_as_cat = add_num_as_cat(train_aug, test, orig, NUM_COLS)
    freq = add_freq_features(train_aug, test, orig, CAT_COLS + combos)
    orig_stats = add_orig_mean_std(train_aug, test, orig, NUM_COLS + CAT_COLS, TARGET)
    for c in CAT_COLS:
        all_v = pd.concat([train_aug[c], test[c], orig[c]]).astype(str)
        codes, _ = pd.factorize(all_v)
        s_a, s_t = len(train_aug), len(test)
        train_aug[c] = codes[:s_a]
        test[c] = codes[s_a:s_a+s_t]
        orig[c] = codes[s_a+s_t:]
    te_cols = CAT_COLS + combos + digits + num_as_cat + tres
    numeric_feats = NUM_COLS + tres + logits + freq + orig_stats
    log(f"  groups: cats={len(CAT_COLS)} combos={len(combos)} digits={len(digits)} "
        f"nac={len(num_as_cat)} freq={len(freq)} orig_stats={len(orig_stats)} "
        f"te_cols={len(te_cols)}")

    # 5-fold split on REAL train rows only.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros((n_real, 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []

    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000, max_depth=4, max_leaves=30,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=2, reg_alpha=5, reg_lambda=5,
        max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", num_class=3, tree_method="hist",
        eval_metric="mlogloss", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )
    real_idx = np.arange(n_real)  # mixup rows occupy n_real..n_real+n_mix-1

    for fold, (tr_idx_real, va_idx) in enumerate(skf.split(real_idx, y_real), 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        tr_set = np.zeros(n_real, dtype=bool); tr_set[tr_idx_real] = True
        keep_mix = (tr_set[pi_r] & tr_set[pj_r]) if n_mix else np.zeros(0, dtype=bool)
        n_keep = int(keep_mix.sum())
        log(f"  +{n_keep:,} mixup rows ({100*n_keep/max(1,n_mix):.1f}% kept)")
        # Indices into augmented frame
        tr_aug_idx = np.concatenate([tr_idx_real, n_real + np.where(keep_mix)[0]])
        X_tr = train_aug.iloc[tr_aug_idx].copy().reset_index(drop=True)
        X_va = train_aug.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)

        # Per-fold OTE on tr ∪ kept_mixup.
        rng2 = np.random.default_rng(SEED + fold)
        perm = rng2.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=te_cols, target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)

        feat_cols = numeric_feats + te.te_col_names()
        y_tr = np.concatenate([y_real[tr_idx_real], mix_y[keep_mix]])
        sw = compute_sample_weight("balanced", y_tr).astype(np.float32)
        sw[len(tr_idx_real):] *= mix_w[keep_mix]  # confidence-attenuate mixup

        log(f"  training XGB on {len(feat_cols)} features, {len(X_tr):,} rows "
            f"({len(tr_idx_real):,} real + {n_keep:,} mix)")
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X_tr[feat_cols], y_tr, sample_weight=sw,
              eval_set=[(X_va[feat_cols], y_real[va_idx])], verbose=500)
        oof[va_idx] = m.predict_proba(X_va[feat_cols]).astype(np.float32)
        test_pred += m.predict_proba(X_te[feat_cols]).astype(np.float32) / N_FOLDS
        bal = balanced_accuracy_score(y_real[va_idx], oof[va_idx].argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} argmax_bal_acc = {bal:.5f}  best_iter={m.best_iteration}")

    overall = balanced_accuracy_score(y_real, oof.argmax(1))
    prior = np.bincount(y_real, minlength=3) / n_real
    bias, tuned = tune_log_bias(oof, y_real, prior)
    log(f"OOF argmax={overall:.5f} tuned={tuned:.5f} bias={bias.round(4).tolist()}")

    np.save(ART / f"oof_angle_{OUT_SUFFIX}_mixup.npy", oof)
    np.save(ART / f"test_angle_{OUT_SUFFIX}_mixup.npy", test_pred)
    eps = 1e-9
    pred_idx = (np.log(np.clip(test_pred, eps, 1)) + bias).argmax(1)
    pd.DataFrame({"id": test_ids, TARGET: [IDX2CLS[i] for i in pred_idx]}).to_csv(
        SUB / f"submission_angle_{OUT_SUFFIX}_mixup.csv", index=False)
    out = dict(
        smoke=SMOKE, n_folds=N_FOLDS, k_mix=K_MIX, beta_a=BETA_A,
        conf_thresh=CONF_THRESH, drop_mh=DROP_MH, suffix=OUT_SUFFIX,
        n_pairs=int(len(pi)), n_mix=int(n_mix),
        fold_scores_argmax=[float(s) for s in fold_scores],
        overall_argmax=float(overall),
        tuned_log_bias_bal_acc=float(tuned),
        log_bias=bias.tolist(),
        wall_min=(time.time() - t0) / 60.0,
    )
    with open(ART / f"angle_{OUT_SUFFIX}_mixup_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote angle_{OUT_SUFFIX}_mixup_results.json wall={out['wall_min']:.1f}min")


if __name__ == "__main__":
    main()
