"""N2 — Per-row gating XGB on v1 <-> rawashishsin disagreement set.

Design (per CLAUDE.md 2026-04-29 next-steps note):
  - v1 (LB 0.98129) and rawashishsin v3 (LB 0.98109) are both
    naturally-calibrated, both LB-positive.
  - They disagree on ~609 test rows (0.226%) and ~2000 OOF rows.
  - Train a small XGB binary classifier on the OOF disagreement
    rows: target = 1 if v1's argmax matches y else 0.
  - Apply at inference on the 609 test disagreement rows:
      P(v1_correct) > 0.55  -> use v1
      P(v1_correct) < 0.45  -> use rawashishsin
      else                   -> default v1
  - Fixed thresholds [0.45, 0.55]. NO grid search (CLAUDE.md
    linear-projection rule).

Each model's argmax is taken at its LB-validated tuned bias.
Bias for v1 = [0.43, 0.87, 3.20]   (RF natural, sklearn tune)
Bias for rawashishsin = its tune_log_bias result on OOF probs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import add_distance_features, fast_bal_acc, tune_log_bias

ART = Path("scripts/artifacts")
SUB = Path("submissions")
SEED = 42
N_FOLDS = 5

# ----- load -----
print("[load] data + OOF/test artefacts")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values.astype(np.int64)
prior = np.bincount(y) / len(y)

v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy")
raw_oof = np.load(ART / "oof_rawashishsin_2600.npy")
v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy")
raw_test = np.load(ART / "test_rawashishsin_2600.npy")

# ----- compute argmaxes at each model's LB-validated bias -----
v1_bias, v1_score = tune_log_bias(v1_oof, y, prior)
raw_bias, raw_score = tune_log_bias(raw_oof, y, prior)
print(f"[bias] v1={v1_bias.round(3).tolist()} score={v1_score:.5f}")
print(f"[bias] raw={raw_bias.round(3).tolist()} score={raw_score:.5f}")

eps = 1e-9
v1_oof_log = np.log(np.clip(v1_oof, eps, 1.0)) + v1_bias
raw_oof_log = np.log(np.clip(raw_oof, eps, 1.0)) + raw_bias
v1_test_log = np.log(np.clip(v1_test, eps, 1.0)) + v1_bias
raw_test_log = np.log(np.clip(raw_test, eps, 1.0)) + raw_bias

v1_arg_oof = v1_oof_log.argmax(1)
raw_arg_oof = raw_oof_log.argmax(1)
v1_arg_test = v1_test_log.argmax(1)
raw_arg_test = raw_test_log.argmax(1)

dis_oof_mask = v1_arg_oof != raw_arg_oof
dis_test_mask = v1_arg_test != raw_arg_test
print(f"[disagree] OOF rows: {dis_oof_mask.sum()} / {len(y)}")
print(f"[disagree] test rows: {dis_test_mask.sum()} / {len(v1_arg_test)}")

# Sanity vs PRIMARY submission CSVs
v1_sub = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")
raw_sub = pd.read_csv(SUB / "submission_rawashishsin_2600_standalone.csv")
v1_sub_lab = v1_sub["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values
raw_sub_lab = raw_sub["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values
sub_disagree = (v1_sub_lab != raw_sub_lab).sum()
print(f"[disagree] LB-validated submissions: {sub_disagree} rows")
v1_match = (v1_arg_test == v1_sub_lab).sum() / len(v1_sub_lab)
raw_match = (raw_arg_test == raw_sub_lab).sum() / len(raw_sub_lab)
print(f"[sanity] v1 argmax matches sub at {v1_match:.4f}; raw matches at {raw_match:.4f}")

# Use submission labels as the "deployed argmax" (they are the LB-validated truth).
# This avoids small bias-tune disagreements between our local recompute and what
# was actually deployed.
v1_arg_test = v1_sub_lab
raw_arg_test = raw_sub_lab
dis_test_mask = v1_arg_test != raw_arg_test
print(f"[disagree] test (after sub-aligned): {dis_test_mask.sum()} / {len(v1_arg_test)}")

# ----- build features for the binary gating XGB -----
print("[features] computing dist features on train+test")
train_feat = add_distance_features(train)
test_feat = add_distance_features(test)

DIST_COLS = ["sm_dist", "rf_dist", "tc_dist", "ws_dist",
             "sm_abs", "rf_abs", "tc_abs", "ws_abs",
             "min_axis_abs", "min_boundary_dist",
             "score_dist_low_mid", "score_dist_mid_high"]
RULE_INT = ["dry", "norain", "hot", "windy", "nomulch", "kc_active",
            "dgp_score", "rule_pred"]
INT_COLS = ["sm_x_rf", "tc_x_ws", "sm_x_kc", "rf_x_kc"]

# Per-row meta features built from BOTH model probs (these encode local
# calibration confidence + which class each model leans toward).
def build_meta_feats(oof_or_test, log_or_test):
    p = np.clip(oof_or_test, eps, 1.0)
    out = {
        "v1_max_prob": p.max(1),
        "v1_p_low": p[:, 0],
        "v1_p_med": p[:, 1],
        "v1_p_high": p[:, 2],
        "v1_arg": log_or_test.argmax(1),
        "v1_argmax_logit": log_or_test.max(1),
        "v1_logit_margin": np.partition(log_or_test, -2, axis=1)[:, -1]
                           - np.partition(log_or_test, -2, axis=1)[:, -2],
    }
    return out

train_meta_v1 = build_meta_feats(v1_oof, v1_oof_log)
train_meta_raw = build_meta_feats(raw_oof, raw_oof_log)
test_meta_v1 = build_meta_feats(v1_test, v1_test_log)
test_meta_raw = build_meta_feats(raw_test, raw_test_log)

def stack_features(df_feat, meta_v1, meta_raw):
    cols = []
    cols.extend([df_feat[c].to_numpy() for c in DIST_COLS])
    cols.extend([df_feat[c].to_numpy().astype(np.float32) for c in RULE_INT])
    cols.extend([df_feat[c].to_numpy() for c in INT_COLS])
    cols.extend([meta_v1[k] for k in ["v1_max_prob", "v1_p_low", "v1_p_med",
                                        "v1_p_high", "v1_argmax_logit", "v1_logit_margin"]])
    cols.extend([meta_raw[k] for k in ["v1_max_prob", "v1_p_low", "v1_p_med",
                                         "v1_p_high", "v1_argmax_logit", "v1_logit_margin"]])
    cols.append(meta_v1["v1_arg"].astype(np.float32))
    cols.append(meta_raw["v1_arg"].astype(np.float32))
    return np.column_stack(cols).astype(np.float32)

X_full_train = stack_features(train_feat, train_meta_v1, train_meta_raw)
X_full_test = stack_features(test_feat, test_meta_v1, test_meta_raw)
print(f"[features] full train: {X_full_train.shape}, test: {X_full_test.shape}")

# Restrict to disagreement rows
X_oof = X_full_train[dis_oof_mask]
y_v1_correct = (v1_arg_oof[dis_oof_mask] == y[dis_oof_mask]).astype(np.int64)
y_raw_correct = (raw_arg_oof[dis_oof_mask] == y[dis_oof_mask]).astype(np.int64)
both_wrong = (~y_v1_correct.astype(bool)) & (~y_raw_correct.astype(bool))
print(f"[oof_dis] n={len(X_oof)}  v1_correct={y_v1_correct.sum()}"
      f"  raw_correct={y_raw_correct.sum()}  both_wrong={both_wrong.sum()}")

# Target for binary gating: "is v1 correct on this disagreement row?"
# Class balance:
print(f"[target] target=1 (v1 correct) frac = {y_v1_correct.mean():.4f}")

# ----- 5-fold CV on the 2009-row disagreement set -----
oof_pgate = np.full(len(X_oof), -1.0, dtype=np.float32)
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

xgb_params = dict(
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=1.0,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    seed=SEED,
)
NROUND = 600
ES = 50

models = []
for fi, (tr, va) in enumerate(skf.split(X_oof, y_v1_correct)):
    dtr = xgb.DMatrix(X_oof[tr], label=y_v1_correct[tr])
    dva = xgb.DMatrix(X_oof[va], label=y_v1_correct[va])
    bst = xgb.train(xgb_params, dtr, num_boost_round=NROUND,
                    evals=[(dva, "val")], early_stopping_rounds=ES, verbose_eval=False)
    p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
    oof_pgate[va] = p
    auc = float(((p > 0.5) == y_v1_correct[va]).mean())
    models.append(bst)
    print(f"[fold {fi+1}] best_iter={bst.best_iteration}  acc@0.5={auc:.4f}")

# ----- diagnostic on OOF disagreement -----
from sklearn.metrics import roc_auc_score, average_precision_score
auc_oof = roc_auc_score(y_v1_correct, oof_pgate)
ap_oof = average_precision_score(y_v1_correct, oof_pgate)
print(f"\n[OOF gating] AUC={auc_oof:.4f}  AP={ap_oof:.4f}  base_rate={y_v1_correct.mean():.4f}")

# Decision rule: P>0.55 use v1, P<0.45 use raw, else default v1.
# Apply to OOF disagreement rows and compute net macro-recall improvement vs v1-only.
gate_pick_v1 = oof_pgate >= 0.45  # band-default to v1
gate_pick_raw = oof_pgate < 0.45
n_pick_raw_oof = gate_pick_raw.sum()
print(f"[gate OOF] band=[0.45, 0.55), n_pick_raw={n_pick_raw_oof}/{len(oof_pgate)}")

# Build gated argmax for OOF (full 630k — only modified on disagree rows)
v1_arg_oof_aligned = v1_arg_oof.copy()
gated_arg_oof = v1_arg_oof_aligned.copy()
dis_idx = np.where(dis_oof_mask)[0]
for j, idx in enumerate(dis_idx):
    if oof_pgate[j] < 0.45:
        gated_arg_oof[idx] = raw_arg_oof[idx]
    # else keep v1's argmax (default)

# Evaluate macro-recall: v1-only vs gated, on full OOF
v1_only_macro = fast_bal_acc(y, v1_arg_oof_aligned)
gated_macro = fast_bal_acc(y, gated_arg_oof)
print(f"\n[macro-recall OOF]")
print(f"  v1 only    : {v1_only_macro:.5f}")
print(f"  gated      : {gated_macro:.5f}  (Δ={gated_macro - v1_only_macro:+.5f})")

# Per-class recall delta
def per_class(y_true, y_pred):
    return [((y_pred == c) & (y_true == c)).sum() / max((y_true == c).sum(), 1)
            for c in range(3)]

v1_pcr = per_class(y, v1_arg_oof_aligned)
gated_pcr = per_class(y, gated_arg_oof)
delta = [g - v for v, g in zip(v1_pcr, gated_pcr)]
print(f"  v1     PCR  : L={v1_pcr[0]:.5f} M={v1_pcr[1]:.5f} H={v1_pcr[2]:.5f}")
print(f"  gated  PCR  : L={gated_pcr[0]:.5f} M={gated_pcr[1]:.5f} H={gated_pcr[2]:.5f}")
print(f"  delta  PCR  : L={delta[0]:+.5f} M={delta[1]:+.5f} H={delta[2]:+.5f}")

# G4: net_high flip and asymmetry
net_h = (gated_arg_oof == 2).sum() - (v1_arg_oof_aligned == 2).sum()
add_h = ((v1_arg_oof_aligned != 2) & (gated_arg_oof == 2)).sum()
rem_h = ((v1_arg_oof_aligned == 2) & (gated_arg_oof != 2)).sum()
churn = add_h + rem_h
g4_ratio = abs(net_h) / max(churn, 1)
print(f"  net_H={net_h}  add_H={add_h}  rem_H={rem_h}  churn={churn}  ratio={g4_ratio:.3f}")

# ----- inference on test -----
print("\n[infer] applying gating to test")
n_dis_test = dis_test_mask.sum()
dis_test_idx = np.where(dis_test_mask)[0]

# Refit on FULL OOF disagreement set (no holdout) for test inference
print("[refit] training on all OOF disagreement rows for test")
dall = xgb.DMatrix(X_oof, label=y_v1_correct)
best_iters = [m.best_iteration for m in models]
final_iter = int(np.mean(best_iters))
print(f"[refit] using {final_iter} rounds (mean of fold best_iters)")
final_bst = xgb.train(xgb_params, dall, num_boost_round=final_iter)

X_test_dis = X_full_test[dis_test_mask]
test_pgate = final_bst.predict(xgb.DMatrix(X_test_dis))

n_to_raw = int((test_pgate < 0.45).sum())
n_to_v1 = int((test_pgate >= 0.45).sum())
print(f"[infer] test gating: {n_to_v1} keep v1, {n_to_raw} -> rawashishsin")

# Build final submission labels
final_labels = v1_arg_test.copy()
for j, idx in enumerate(dis_test_idx):
    if test_pgate[j] < 0.45:
        final_labels[idx] = raw_arg_test[idx]

# Diff vs primary
diff_vs_v1 = (final_labels != v1_arg_test).sum()
print(f"[infer] final differs from v1 PRIMARY on {diff_vs_v1} test rows")

# Class distribution
print(f"[infer] final class dist: L={int((final_labels==0).sum())} "
      f"M={int((final_labels==1).sum())} H={int((final_labels==2).sum())}")
print(f"[infer] v1 PRIMARY dist:  L={int((v1_arg_test==0).sum())} "
      f"M={int((v1_arg_test==1).sum())} H={int((v1_arg_test==2).sum())}")

# Write submission
sub_path = SUB / "submission_n2_v1_raw_gating_b045_055.csv"
inv = {0: "Low", 1: "Medium", 2: "High"}
sub_df = pd.DataFrame({
    "id": test["id"].values,
    "Irrigation_Need": [inv[int(c)] for c in final_labels],
})
sub_df.to_csv(sub_path, index=False)
print(f"[write] {sub_path}")

# Save OOF + test gate probs for diagnostic
np.save(ART / "oof_n2_pgate.npy", oof_pgate)
np.save(ART / "test_n2_pgate.npy", test_pgate)

# ----- 4-gate verdict -----
print("\n[4-gate verdict vs v1 PRIMARY]")
print(f"  G1 (Δ ≥ +0.0002):       Δ={gated_macro - v1_only_macro:+.5f}  "
      f"{'PASS' if gated_macro - v1_only_macro >= 2e-4 else 'FAIL'}")
g2_pass = all(d >= -5e-4 for d in delta)
print(f"  G2 (PCR ≥ -5e-4):       L={delta[0]:+.5f} M={delta[1]:+.5f} H={delta[2]:+.5f}  "
      f"{'PASS' if g2_pass else 'FAIL'}")
print(f"  G3 (linear stability):  not applicable (single-α gating)")
g4_pass = (net_h >= 0) and (g4_ratio >= 0.5)
print(f"  G4 (net_H≥0 + asym≥0.5):net_H={net_h}  ratio={g4_ratio:.3f}  "
      f"{'PASS' if g4_pass else 'FAIL'}")

# Save results JSON
results = {
    "v1_oof_score": float(v1_only_macro),
    "gated_oof_score": float(gated_macro),
    "delta_oof": float(gated_macro - v1_only_macro),
    "v1_pcr": [float(x) for x in v1_pcr],
    "gated_pcr": [float(x) for x in gated_pcr],
    "delta_pcr": [float(x) for x in delta],
    "binary_gate_auc": float(auc_oof),
    "binary_gate_ap": float(ap_oof),
    "binary_target_base_rate": float(y_v1_correct.mean()),
    "n_oof_disagreement": int(dis_oof_mask.sum()),
    "n_test_disagreement": int(dis_test_mask.sum()),
    "n_test_pick_v1": int(n_to_v1),
    "n_test_pick_raw": int(n_to_raw),
    "n_test_diff_from_v1": int(diff_vs_v1),
    "net_h": int(net_h),
    "add_h": int(add_h),
    "rem_h": int(rem_h),
    "g4_ratio": float(g4_ratio),
    "submission_path": str(sub_path),
    "v1_bias": v1_bias.tolist(),
    "raw_bias": raw_bias.tolist(),
}
out_json = ART / "n2_v1_raw_gating_results.json"
out_json.write_text(json.dumps(results, indent=2))
print(f"\n[done] {out_json}")
