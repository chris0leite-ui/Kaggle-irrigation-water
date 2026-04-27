"""Experiment A: wide programmatic feature engineering + forward selection.

Cdeotte's NVIDIA cuDF FE pattern (1st place backpack prices):
  1. Generate THOUSANDS of features programmatically via groupby-stat
     combinations, distribution buckets, quantile bin features.
  2. Train one XGB fold to get gain importance.
  3. Forward-select the top ~500-600 features.
  4. Train final 5-fold model on the selected set.

Adapted to our 21GB CPU box (no cuDF):
  - Start from recipe FE base (~440 features).
  - Add ~1700 NEW programmatic features:
      - 7-stat group-by per (cat, num): mean, std, min, max, q25, q50, q75
        (cat ∈ 8, num ∈ 11) = 616 features (we already had mean+std = 176)
      - 8-quantile group-by per (cat, num): [5,10,40,45,55,60,90,95]
        = 704 features
      - extended decimal features: (col * 10) % 1 .round(2) for all 11 nums
        = 11 features
  - 1-fold importance scan (~5 min on full 630k).
  - Select top 600 by gain.
  - 5-fold StratifiedKFold(seed=42) on selected features.
  - Per-fold checkpointing for rehydrate-resilience.

Outputs:
  oof_wide_fe.npy / test_wide_fe.npy
  oof_wide_fe_fold{N}.npy / test_wide_fe_fold{N}.npy (per-fold checkpoints)
  wide_fe_results.json
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402
from recipe_full_te import load_and_engineer, TARGET, CLS_MAP  # noqa: E402
from recipe_ote import OrderedTE  # noqa: E402

SEED = 42
SMOKE = os.environ.get("SMOKE") == "1"
N_FOLDS = 2 if SMOKE else 5
TOP_N = 200 if SMOKE else 600

ART = Path("scripts/artifacts")
SUB = Path("submissions")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def add_wide_groupby_stats(train: pd.DataFrame, test: pd.DataFrame,
                           cats: list[str], nums: list[str]) -> list[str]:
    """For each (cat, num) pair, compute 7 group-by stats fitted on train."""
    new_cols = []
    for c in cats:
        for n in nums:
            grp = train.groupby(c, observed=False)[n]
            stats = grp.agg(["mean", "std", "min", "max"]).reset_index()
            qs = grp.quantile([0.25, 0.5, 0.75]).unstack().reset_index()
            qs.columns = [c, f"WIDE_{c}_{n}_q25", f"WIDE_{c}_{n}_q50", f"WIDE_{c}_{n}_q75"]
            stats.columns = [c, f"WIDE_{c}_{n}_mean", f"WIDE_{c}_{n}_std",
                             f"WIDE_{c}_{n}_min", f"WIDE_{c}_{n}_max"]
            for df in (train, test):
                m = df.merge(stats, on=c, how="left")
                for col in stats.columns[1:]:
                    df[col] = m[col].fillna(0).astype(np.float32).values
                m2 = df.merge(qs, on=c, how="left")
                for col in qs.columns[1:]:
                    df[col] = m2[col].fillna(0).astype(np.float32).values
            new_cols += list(stats.columns[1:]) + list(qs.columns[1:])
    return new_cols


def add_wide_quantile_features(train: pd.DataFrame, test: pd.DataFrame,
                                cats: list[str], nums: list[str]) -> list[str]:
    """For each (cat, num), compute 8 quantiles [5,10,40,45,55,60,90,95]."""
    qs_pct = [0.05, 0.10, 0.40, 0.45, 0.55, 0.60, 0.90, 0.95]
    new_cols = []
    for c in cats:
        for n in nums:
            grp = train.groupby(c, observed=False)[n]
            quants = grp.quantile(qs_pct).unstack().reset_index()
            quants.columns = [c] + [f"WIDEQ_{c}_{n}_q{int(q*100):02d}" for q in qs_pct]
            for df in (train, test):
                m = df.merge(quants, on=c, how="left")
                for col in quants.columns[1:]:
                    df[col] = m[col].fillna(0).astype(np.float32).values
            new_cols += list(quants.columns[1:])
    return new_cols


def add_extended_decimals(df: pd.DataFrame, nums: list[str]) -> list[str]:
    """`(col * 10) % 1 .round(2)` — 1-decimal-shifted decimal features."""
    new = []
    for c in nums:
        if c not in df.columns:
            continue
        name = f"WIDED_{c}"
        df[name] = ((df[c] * 10) % 1).round(2).astype(np.float32)
        new.append(name)
    return new


def main():
    t0 = time.time()
    log(f"config: SMOKE={SMOKE}  N_FOLDS={N_FOLDS}  TOP_N={TOP_N}")
    train, test, info, test_ids = load_and_engineer()
    y = train[TARGET].to_numpy(dtype=np.int32)

    log(f"adding wide group-by 7-stat features (cats={len(info['cats'])} × nums={len(info['nums'])})")
    wide_gby = add_wide_groupby_stats(train, test, info["cats"], info["nums"])
    log(f"  +{len(wide_gby)} cols")
    log(f"adding wide group-by 8-quantile features")
    wide_qs = add_wide_quantile_features(train, test, info["cats"], info["nums"])
    log(f"  +{len(wide_qs)} cols")
    log(f"adding extended decimal features (×10 shift) on all {len(info['nums'])} nums")
    wide_dec = []
    for df in (train, test):
        wide_dec = add_extended_decimals(df, info["nums"])
    log(f"  +{len(wide_dec)} cols")

    base_numeric = (info["nums"] + info["tres"] + info["logits"] +
                    info["freq"] + info["orig_stats"])
    numeric_feats = base_numeric + wide_gby + wide_qs + wide_dec
    te_cols = info["te_cols"]
    log(f"total candidate features: {len(numeric_feats)} numeric + {len(te_cols)} OTE-target cats")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(skf.split(train, y))
    tr_idx, va_idx = folds[0]
    log(f"\n=== 1-fold importance scan (fold 1 of {N_FOLDS}) ===")
    X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
    X_va = train.iloc[va_idx].copy().reset_index(drop=True)

    log(f"  fitting OrderedTE on fold 1 train")
    rng = np.random.default_rng(SEED + 1)
    perm = rng.permutation(len(X_tr))
    X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
    te = OrderedTE(a=1.0)
    X_tr_shuf = te.fit(X_tr_shuf, cat_cols=te_cols, target=TARGET)
    inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
    X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
    X_va = te.transform(X_va)
    feat_cols = numeric_feats + te.te_col_names()
    log(f"  feat_cols total: {len(feat_cols)}  (importance-scan fold 1)")

    sw = compute_sample_weight("balanced", y[tr_idx])
    booster = xgb.XGBClassifier(
        n_estimators=200 if SMOKE else 800,
        max_depth=4, max_leaves=30, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        reg_alpha=5, reg_lambda=5, max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=30 if SMOKE else 100, verbosity=0,
    )
    booster.fit(X_tr[feat_cols], y[tr_idx], sample_weight=sw,
                eval_set=[(X_va[feat_cols], y[va_idx])], verbose=200)
    log(f"  fold 1 best_iter = {booster.best_iteration}")

    importances = booster.get_booster().get_score(importance_type="gain")
    name2gain = dict(importances)
    used = sorted(name2gain.items(), key=lambda kv: -kv[1])
    top_feats = set(n for n, _ in used[:TOP_N])
    log(f"  importance scan: {len(used)} features used, selecting top {TOP_N}")
    log(f"  top-10: {[(n, round(g, 1)) for n, g in used[:10]]}")
    log(f"  bottom-of-selected gain: {round(used[min(TOP_N-1, len(used)-1)][1], 2)}")

    selected_numeric = [c for c in numeric_feats if c in top_feats]
    selected_te_targets = []
    for tc in te_cols:
        if any(f"{tc}_TE_cls{cls}" in top_feats for cls in (0, 1, 2)):
            selected_te_targets.append(tc)
    log(f"  selected: {len(selected_numeric)} numeric + {len(selected_te_targets)} OTE-target cats")

    log(f"\n=== 5-fold full training on selected features ===")
    oof = np.zeros((len(train), 3), dtype=np.float32)
    test_pred = np.zeros((len(test), 3), dtype=np.float32)
    fold_scores = []
    ck_prefix = "wide_fe" + ("_smoke" if SMOKE else "")
    cached: set[int] = set()
    for fold_check in range(1, N_FOLDS + 1):
        ck_o = ART / f"oof_{ck_prefix}_fold{fold_check}.npy"
        ck_t = ART / f"test_{ck_prefix}_fold{fold_check}.npy"
        if ck_o.exists() and ck_t.exists():
            cached.add(fold_check)
    if cached:
        log(f"  resume: {len(cached)} fold(s) cached: {sorted(cached)}")

    xgb_params = dict(
        n_estimators=300 if SMOKE else 3000,
        max_depth=4, max_leaves=30, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
        reg_alpha=5, reg_lambda=5, max_bin=256 if SMOKE else 1024,
        objective="multi:softprob", tree_method="hist",
        eval_metric="mlogloss", n_jobs=-1, random_state=SEED,
        early_stopping_rounds=50 if SMOKE else 200, verbosity=0,
    )
    for fold, (tr_idx, va_idx) in enumerate(folds, 1):
        log(f"=== fold {fold}/{N_FOLDS} ===")
        if fold in cached:
            ck_o = ART / f"oof_{ck_prefix}_fold{fold}.npy"
            ck_t = ART / f"test_{ck_prefix}_fold{fold}.npy"
            vp = np.load(ck_o); tp = np.load(ck_t)
            oof[va_idx] = vp
            test_pred += tp / N_FOLDS
            fold_scores.append(balanced_accuracy_score(y[va_idx], vp.argmax(1)))
            log(f"  fold {fold} CACHED  bal={fold_scores[-1]:.5f}")
            continue
        X_tr = train.iloc[tr_idx].copy().reset_index(drop=True)
        X_va = train.iloc[va_idx].copy().reset_index(drop=True)
        X_te = test.copy().reset_index(drop=True)
        log(f"  fitting OrderedTE on fold {fold}")
        t1 = time.time()
        rng = np.random.default_rng(SEED + fold)
        perm = rng.permutation(len(X_tr))
        X_tr_shuf = X_tr.iloc[perm].reset_index(drop=True)
        te = OrderedTE(a=1.0)
        X_tr_shuf = te.fit(X_tr_shuf, cat_cols=selected_te_targets, target=TARGET)
        inv = np.empty_like(perm); inv[perm] = np.arange(len(perm))
        X_tr = X_tr_shuf.iloc[inv].reset_index(drop=True)
        X_va = te.transform(X_va)
        X_te = te.transform(X_te)
        log(f"    OTE done in {time.time()-t1:.1f}s")
        feat_cols_sel = selected_numeric + te.te_col_names()
        log(f"  training XGB on {len(feat_cols_sel)} features, {len(X_tr):,} rows")
        sw = compute_sample_weight("balanced", y[tr_idx])
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_tr[feat_cols_sel], y[tr_idx], sample_weight=sw,
                  eval_set=[(X_va[feat_cols_sel], y[va_idx])], verbose=500)
        vp = model.predict_proba(X_va[feat_cols_sel]).astype(np.float32)
        tp = model.predict_proba(X_te[feat_cols_sel]).astype(np.float32)
        oof[va_idx] = vp
        test_pred += tp / N_FOLDS
        np.save(ART / f"oof_{ck_prefix}_fold{fold}.npy", vp)
        np.save(ART / f"test_{ck_prefix}_fold{fold}.npy", tp)
        bal = balanced_accuracy_score(y[va_idx], vp.argmax(1))
        fold_scores.append(bal)
        log(f"  fold {fold} bal={bal:.5f} best_iter={model.best_iteration}")

    overall = balanced_accuracy_score(y, oof.argmax(1))
    log(f"=== OOF argmax bal_acc = {overall:.5f}")
    prior = np.bincount(y, minlength=3) / len(y)
    bias, tuned = tune_log_bias(oof, y, prior)
    log(f"tuned log-bias bal_acc = {tuned:.5f}  bias={bias.round(4).tolist()}")

    np.save(ART / f"oof_wide_fe{('_smoke' if SMOKE else '')}.npy", oof)
    np.save(ART / f"test_wide_fe{('_smoke' if SMOKE else '')}.npy", test_pred)
    out = dict(
        config=dict(smoke=SMOKE, n_folds=N_FOLDS, top_n=TOP_N, seed=SEED),
        n_features_total=len(feat_cols),
        n_features_selected=len(selected_numeric) + len(selected_te_targets) * 3,
        top10_features=[(n, float(g)) for n, g in used[:10]],
        fold_scores=fold_scores,
        overall_argmax=float(overall),
        tuned_bal=float(tuned),
        tuned_bias=bias.tolist(),
        elapsed_sec=float(time.time() - t0),
    )
    with open(ART / f"wide_fe{('_smoke' if SMOKE else '')}_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote oof/test_wide_fe.npy + results.json  (elapsed {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
