"""N3b — 2-way arithmetic mean of RF + BaggingLR (drop ET).

After N3 L3 mean had +0.00267 ADD-High but G2 failed on Medium (-0.00266),
the structural fix is to drop the meta with the highest H-class drift (ET
drift_H = +0.2, vs RF -0.2 and BagLR 0.0). Mean of RF + BagLR has cleanest
drift profile (RF drift_max |0.2|, BagLR |0.1| — both well within
|0.30| gate).

Principled by the drift gate, NOT by OOF grid search. The remaining 2
metas are the lowest-drift bagging architectures from N3.

Mechanism: RF (split-randomized trees) + BagLR (linear-bagged) are
maximally architecturally orthogonal of the original 3, both natural-cal.
Mean preserves natural calibration if both inputs do.
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

print("[load] data + 2 metas (RF + BagLR)")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
y = train["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values
prior = np.bincount(y) / len(y)

rf_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy")
rf_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy")
bl_oof = np.load(ART / "oof_bagginglr_natural.npy")
bl_test = np.load(ART / "test_bagginglr_natural.npy")

# 2-way mean (preserves natural-cal)
def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)

l2_oof = _normed((rf_oof + bl_oof) / 2.0)
l2_test = _normed((rf_test + bl_test) / 2.0)

eps = 1e-9
rf_bias, rf_score = tune_log_bias(rf_oof, y, prior)
bl_bias, bl_score = tune_log_bias(bl_oof, y, prior)
l2_bias, l2_score = tune_log_bias(l2_oof, y, prior)
rf_drift = rf_bias - (-np.log(prior))
bl_drift = bl_bias - (-np.log(prior))
l2_drift = l2_bias - (-np.log(prior))

print(f"\n[standalone metas]")
print(f"  RF      tuned={rf_score:.5f}  bias={rf_bias.round(3).tolist()}  drift={rf_drift.round(3).tolist()}")
print(f"  BagLR   tuned={bl_score:.5f}  bias={bl_bias.round(3).tolist()}  drift={bl_drift.round(3).tolist()}")
print(f"  L2 mean tuned={l2_score:.5f}  bias={l2_bias.round(3).tolist()}  drift={l2_drift.round(3).tolist()}")

drift_max = float(max(abs(d) for d in l2_drift))
print(f"  L2 drift max = {drift_max:.3f}  {'PASS' if drift_max <= 0.30 else 'FAIL'} drift gate")

# 4-gate vs v1 anchor
v1_pred = (np.log(np.clip(rf_oof, eps, 1.0)) + rf_bias).argmax(1)
l2_pred = (np.log(np.clip(l2_oof, eps, 1.0)) + l2_bias).argmax(1)

anchor_bal = fast_bal_acc(y, v1_pred)
l2_bal = fast_bal_acc(y, l2_pred)

def per_class(y, p):
    return [((p == c) & (y == c)).sum() / max((y == c).sum(), 1)
            for c in range(3)]

v1_pcr = per_class(y, v1_pred)
l2_pcr = per_class(y, l2_pred)
delta = [l2_pcr[k] - v1_pcr[k] for k in range(3)]

net_h = int((l2_pred == 2).sum() - (v1_pred == 2).sum())
add_h = int(((v1_pred != 2) & (l2_pred == 2)).sum())
rem_h = int(((v1_pred == 2) & (l2_pred != 2)).sum())
churn = add_h + rem_h
ratio = abs(net_h) / max(churn, 1)

print(f"\n[4-gate vs v1 PRIMARY (LB 0.98129)]")
print(f"  v1     bal={anchor_bal:.5f}  PCR=[L={v1_pcr[0]:.5f} M={v1_pcr[1]:.5f} H={v1_pcr[2]:.5f}]")
print(f"  L2 RF+BL bal={l2_bal:.5f}  PCR=[L={l2_pcr[0]:.5f} M={l2_pcr[1]:.5f} H={l2_pcr[2]:.5f}]")
print(f"  Δ bal     = {l2_bal - anchor_bal:+.5f}")
print(f"  Δ PCR     = L={delta[0]:+.5f}  M={delta[1]:+.5f}  H={delta[2]:+.5f}")
print(f"  net_H={net_h}  add_H={add_h}  rem_H={rem_h}  ratio={ratio:.3f}")

g1 = (l2_bal - anchor_bal) >= 2e-4
g2 = all(d >= -5e-4 for d in delta)
g4 = (net_h >= 0) and (ratio >= 0.5 if churn > 0 else True)
g_drift = drift_max <= 0.30

print(f"\n  G1 (Δ ≥ +2e-4):    {'PASS' if g1 else 'FAIL'}")
print(f"  G2 (PCR ≥ -5e-4):  {'PASS' if g2 else 'FAIL'}")
print(f"  G4 (net_H≥0+ratio≥0.5): {'PASS' if g4 else 'FAIL'}")
print(f"  Drift (≤0.30):     {'PASS' if g_drift else 'FAIL'}")
overall = g1 and g2 and g4 and g_drift
print(f"  OVERALL:           {'PASS — emit candidate' if overall else 'FAIL'}")

# Build submission regardless (for inspection)
inv = {0: "Low", 1: "Medium", 2: "High"}
test_pred = (np.log(np.clip(l2_test, eps, 1.0)) + l2_bias).argmax(1)
sub_path = SUB / "submission_n3b_l2_rf_baglr_mean.csv"
pd.DataFrame({
    "id": test["id"].values,
    "Irrigation_Need": [inv[int(c)] for c in test_pred],
}).to_csv(sub_path, index=False)

# Test diff
v1_sub = pd.read_csv(SUB / "submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")
v1_lab = v1_sub["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values
diff = int((test_pred != v1_lab).sum())
print(f"\n  test diff vs v1 PRIMARY: {diff}")
print(f"  L=={int((test_pred==0).sum())} M={int((test_pred==1).sum())} H={int((test_pred==2).sum())}")
print(f"  v1 L=={int((v1_lab==0).sum())} M={int((v1_lab==1).sum())} H={int((v1_lab==2).sum())}")

np.save(ART / "oof_n3b_l2_rf_baglr.npy", l2_oof)
np.save(ART / "test_n3b_l2_rf_baglr.npy", l2_test)

results = dict(
    rf_score=float(rf_score), rf_drift=rf_drift.tolist(),
    bl_score=float(bl_score), bl_drift=bl_drift.tolist(),
    l2_score=float(l2_score), l2_drift=l2_drift.tolist(), l2_drift_max=drift_max,
    anchor_bal=float(anchor_bal), l2_bal=float(l2_bal),
    delta_bal=float(l2_bal - anchor_bal),
    v1_pcr=v1_pcr, l2_pcr=l2_pcr, delta_pcr=[float(d) for d in delta],
    net_h=net_h, add_h=add_h, rem_h=rem_h, ratio=float(ratio),
    g1_pass=bool(g1), g2_pass=bool(g2), g4_pass=bool(g4), drift_pass=bool(g_drift),
    overall_pass=bool(overall),
    test_diff_vs_v1=diff,
    submission_path=str(sub_path),
)
out = ART / "n3b_l2_rf_baglr_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"\n[done] {out}")
print(f"[wrote] {sub_path}")
