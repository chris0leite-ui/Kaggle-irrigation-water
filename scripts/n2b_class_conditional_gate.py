"""N2b — Class-conditional gating: never override v1=H.

After N2 (symmetric gate) hit REMOVE-High Pareto violation (net_H = -391
on OOF, all flips were rem_H), apply the structural fix recommended by
CLAUDE.md's "REMOVE-High direction killed 7+ candidates" rule:

  - If v1=H: ALWAYS keep v1 (rare-class is more careful in v1).
  - If v1=L or v1=M and disagrees with raw: apply original [0.45, 0.55]
    binary-classifier rule (P(v1_correct) < 0.45 -> use raw, else v1).

This is principled (not grid-searched) — motivated by High class's 12x
per-row leverage under macro-recall.

Reuses the N2 binary classifier OOF/test predictions (oof_n2_pgate.npy,
test_n2_pgate.npy) so cost is ~10 sec.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import fast_bal_acc, tune_log_bias

ART = Path("scripts/artifacts")
SUB = Path("submissions")

# Load
print("[load] data + N2 gate probs + LB-validated submission labels")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values
prior = np.bincount(y) / len(y)

v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy")
raw_oof = np.load(ART / "oof_rawashishsin_2600.npy")
v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy")
raw_test = np.load(ART / "test_rawashishsin_2600.npy")

v1_bias, _ = tune_log_bias(v1_oof, y, prior)
raw_bias, _ = tune_log_bias(raw_oof, y, prior)

eps = 1e-9
v1_arg_oof = (np.log(np.clip(v1_oof, eps, 1.0)) + v1_bias).argmax(1)
raw_arg_oof = (np.log(np.clip(raw_oof, eps, 1.0)) + raw_bias).argmax(1)
dis_oof_mask = v1_arg_oof != raw_arg_oof
print(f"[disagree] OOF rows: {dis_oof_mask.sum()}")

# N2 gate probabilities (only computed on disagreement rows)
oof_pgate = np.load(ART / "oof_n2_pgate.npy")
test_pgate = np.load(ART / "test_n2_pgate.npy")

# Sub-aligned test argmaxes
v1_sub = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")
raw_sub = pd.read_csv(SUB / "submission_rawashishsin_2600_standalone.csv")
v1_arg_test = v1_sub["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values
raw_arg_test = raw_sub["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values
dis_test_mask = v1_arg_test != raw_arg_test

# Apply class-conditional gating to OOF
print("\n[OOF gating: never override v1=H]")
gated_arg_oof = v1_arg_oof.copy()
dis_idx = np.where(dis_oof_mask)[0]
n_kept_h = 0
n_overridden = 0
for j, idx in enumerate(dis_idx):
    if v1_arg_oof[idx] == 2:
        # v1 says High - never override
        n_kept_h += 1
        continue
    # v1 != High: apply [0.45, 0.55] rule
    if oof_pgate[j] < 0.45:
        gated_arg_oof[idx] = raw_arg_oof[idx]
        n_overridden += 1

v1_macro = fast_bal_acc(y, v1_arg_oof)
gated_macro = fast_bal_acc(y, gated_arg_oof)
print(f"  v1=H kept on disagree: {n_kept_h} rows  (would-have-been overridden)")
print(f"  overridden (v1!=H): {n_overridden} rows")
print(f"  v1 macro    = {v1_macro:.5f}")
print(f"  gated macro = {gated_macro:.5f}  Δ = {gated_macro - v1_macro:+.5f}")

def per_class(y_true, y_pred):
    return [((y_pred == c) & (y_true == c)).sum() / max((y_true == c).sum(), 1)
            for c in range(3)]

v1_pcr = per_class(y, v1_arg_oof)
g_pcr = per_class(y, gated_arg_oof)
delta = [g - v for v, g in zip(v1_pcr, g_pcr)]
print(f"  v1     PCR: L={v1_pcr[0]:.5f} M={v1_pcr[1]:.5f} H={v1_pcr[2]:.5f}")
print(f"  gated  PCR: L={g_pcr[0]:.5f} M={g_pcr[1]:.5f} H={g_pcr[2]:.5f}")
print(f"  delta  PCR: L={delta[0]:+.5f} M={delta[1]:+.5f} H={delta[2]:+.5f}")

net_h = (gated_arg_oof == 2).sum() - (v1_arg_oof == 2).sum()
add_h = ((v1_arg_oof != 2) & (gated_arg_oof == 2)).sum()
rem_h = ((v1_arg_oof == 2) & (gated_arg_oof != 2)).sum()
churn = add_h + rem_h
ratio = abs(net_h) / max(churn, 1)
print(f"  net_H={net_h}  add_H={add_h}  rem_H={rem_h}  ratio={ratio:.3f}")

g1 = (gated_macro - v1_macro) >= 2e-4
g2 = all(d >= -5e-4 for d in delta)
g4_dir = net_h >= 0
g4_asym = ratio >= 0.5 if churn > 0 else True
print(f"\n  G1 (Δ ≥ +2e-4):                {'PASS' if g1 else 'FAIL'}")
print(f"  G2 (PCR ≥ -5e-4 each):         {'PASS' if g2 else 'FAIL'}")
print(f"  G4 (net_H≥0 + ratio≥0.5):      {'PASS' if g4_dir and g4_asym else 'FAIL'}")
overall = g1 and g2 and g4_dir and g4_asym
print(f"  OVERALL:                       {'PASS — emit candidate' if overall else 'FAIL'}")

# Apply same to test
print("\n[test inference]")
final_labels = v1_arg_test.copy()
dis_test_idx = np.where(dis_test_mask)[0]
n_kept_test = 0
n_overr_test = 0
for j, idx in enumerate(dis_test_idx):
    if v1_arg_test[idx] == 2:
        n_kept_test += 1
        continue
    if test_pgate[j] < 0.45:
        final_labels[idx] = raw_arg_test[idx]
        n_overr_test += 1

print(f"  v1=H kept: {n_kept_test}, overridden: {n_overr_test}")
diff = (final_labels != v1_arg_test).sum()
print(f"  test diff vs v1 PRIMARY: {diff}")
print(f"  final dist: L={(final_labels==0).sum()} M={(final_labels==1).sum()} H={(final_labels==2).sum()}")
print(f"  v1 PRIM dist: L={(v1_arg_test==0).sum()} M={(v1_arg_test==1).sum()} H={(v1_arg_test==2).sum()}")

inv = {0: "Low", 1: "Medium", 2: "High"}
sub_path = SUB / "submission_n2b_classcond_gate.csv"
pd.DataFrame({
    "id": test["id"].values,
    "Irrigation_Need": [inv[int(c)] for c in final_labels],
}).to_csv(sub_path, index=False)
print(f"  wrote {sub_path}")

results = dict(
    n_oof_disagree=int(dis_oof_mask.sum()),
    n_oof_v1H_kept=n_kept_h,
    n_oof_overridden=n_overridden,
    v1_macro=float(v1_macro),
    gated_macro=float(gated_macro),
    delta_macro=float(gated_macro - v1_macro),
    v1_pcr=v1_pcr,
    gated_pcr=g_pcr,
    delta_pcr=delta,
    net_h=int(net_h), add_h=int(add_h), rem_h=int(rem_h),
    g4_ratio=float(ratio),
    g1_pass=bool(g1), g2_pass=bool(g2),
    g4_dir_pass=bool(g4_dir), g4_asym_pass=bool(g4_asym),
    overall_pass=bool(overall),
    n_test_diff_v1=int(diff),
    n_test_v1H_kept=n_kept_test,
    n_test_overridden=n_overr_test,
    submission_path=str(sub_path),
)
out = ART / "n2b_classcond_gate_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"\n[done] {out}")
