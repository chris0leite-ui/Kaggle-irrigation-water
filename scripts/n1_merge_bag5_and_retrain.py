"""N1 merge + retrain — geomean rawashishsin bag5, drop into v1's bank.

Pulled from Kaggle: oof_rawashishsin_te{seed}.npy + test_... for
seed in {7, 123, 2024, 9999}. Combined with seed=42 already on disk
(oof_rawashishsin_2600.npy) into a 5-seed geomean bag.

Output:
  oof_rawashishsin_bag5.npy
  test_rawashishsin_bag5.npy
  rawashishsin_bag5_results.json

Then re-run RF natural meta-stacker (sklearn_rf_meta_natural.py) with
the bag5 component replacing single-seed rawashishsin_2600 in v1's
exact 7-component bank.

To replace the bank component without modifying the upstream script:
  - Save oof_rawashishsin_2600_orig.npy backup (single-seed v3)
  - Save oof_rawashishsin_bag5.npy as oof_rawashishsin_2600.npy
    (overwrites with the bag, so existing scripts pick it up)
  - After meta retrain, restore the backup

Then 4-gate vs LB-best PRIMARY (LB 0.98129).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ART = Path("scripts/artifacts")
SUFFIX_BAG = "_bag5"
SEED_42 = "rawashishsin_2600"
NEW_SEEDS = [7, 123, 2024, 9999]

print("=== N1 merge: geomean rawashishsin bag5 ===")

# Load all 5 seeds
seed_oof = {}
seed_test = {}

# Seed 42 (already on disk as rawashishsin_2600)
seed_oof[42] = np.load(ART / f"oof_{SEED_42}.npy").astype(np.float64)
seed_test[42] = np.load(ART / f"test_{SEED_42}.npy").astype(np.float64)
print(f"  loaded seed 42 (oof_{SEED_42}.npy): "
      f"oof={seed_oof[42].shape} test={seed_test[42].shape}")

# 4 NEW seeds (pulled from Kaggle bag5 kernel) — handle partial completion
missing = []
for s in NEW_SEEDS:
    op = ART / f"oof_rawashishsin_te{s}.npy"
    tp = ART / f"test_rawashishsin_te{s}.npy"
    if not op.exists() or not tp.exists():
        missing.append(s)
        print(f"  MISSING seed {s}: {op.name} or {tp.name}")
        continue
    o = np.load(op).astype(np.float64)
    # Reject partial-fold OOF (sum<1 on some rows means fold not run for that row)
    if (o.sum(1) < 0.5).any():
        missing.append(s)
        zeros = int((o.sum(1) < 0.5).sum())
        print(f"  PARTIAL seed {s}: {zeros} zero-fold rows in OOF — skip")
        continue
    seed_oof[s] = o
    seed_test[s] = np.load(tp).astype(np.float64)
    print(f"  loaded seed {s}: oof={seed_oof[s].shape} test={seed_test[s].shape}")

n_loaded = len(seed_oof)
if n_loaded < 2:
    print(f"\n[abort] only {n_loaded} seed(s) loaded — bag needs ≥ 2 to be useful")
    print(f"  Pull from Kaggle: kaggle kernels output "
          f"chrisleitescha/irrigation-rawashishsin-bag5 -p {ART}")
    sys.exit(1)
if missing:
    print(f"\n[partial] {len(missing)} seeds missing: {missing}")
    print(f"  Proceeding with bag of {n_loaded} seeds (incl. seed=42)")

# Geomean across all 5 seeds (preserves natural-cal: log-mean of probs)
print(f"\n=== geomean bag of {len(seed_oof)} seeds ===")
eps = 1e-9
log_stack_oof = np.stack([np.log(np.clip(seed_oof[s], eps, 1.0))
                           for s in sorted(seed_oof.keys())], axis=0)
log_stack_test = np.stack([np.log(np.clip(seed_test[s], eps, 1.0))
                            for s in sorted(seed_test.keys())], axis=0)
oof_bag = np.exp(log_stack_oof.mean(axis=0))
test_bag = np.exp(log_stack_test.mean(axis=0))
oof_bag = oof_bag / oof_bag.sum(1, keepdims=True)
test_bag = test_bag / test_bag.sum(1, keepdims=True)

print(f"  bag oof shape={oof_bag.shape}  test={test_bag.shape}")

# Diagnostic: per-seed scores vs bag
sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias
import pandas as pd

y = pd.read_csv("data/train.csv")["Irrigation_Need"].map(
    {"Low": 0, "Medium": 1, "High": 2}).values
prior = np.bincount(y) / len(y)

for s in sorted(seed_oof.keys()):
    bias, score = tune_log_bias(seed_oof[s].astype(np.float32), y, prior)
    print(f"  seed {s:>5}: tuned={score:.5f}  "
          f"drift={(bias - (-np.log(prior))).round(3).tolist()}")

bag_bias, bag_score = tune_log_bias(oof_bag.astype(np.float32), y, prior)
print(f"  BAG    : tuned={bag_score:.5f}  "
      f"drift={(bag_bias - (-np.log(prior))).round(3).tolist()}")

# Save bag artifacts
np.save(ART / f"oof_rawashishsin{SUFFIX_BAG}.npy", oof_bag.astype(np.float32))
np.save(ART / f"test_rawashishsin{SUFFIX_BAG}.npy", test_bag.astype(np.float32))
results = {
    "n_seeds": len(seed_oof),
    "seeds": sorted(list(seed_oof.keys())),
    "bag_oof_tuned": float(bag_score),
    "bag_oof_bias": bag_bias.tolist(),
    "bag_oof_drift": (bag_bias - (-np.log(prior))).tolist(),
}
(ART / f"rawashishsin{SUFFIX_BAG}_results.json").write_text(
    json.dumps(results, indent=2))

print(f"\n[saved] oof_rawashishsin{SUFFIX_BAG}.npy + test_... + results.json")

# === Drop into v1's bank by overwriting rawashishsin_2600 ===
print("\n=== integrate bag into v1's bank: ===")
backup = ART / f"oof_{SEED_42}_singleseed_backup.npy"
if not backup.exists():
    shutil.copy(ART / f"oof_{SEED_42}.npy", backup)
    shutil.copy(ART / f"test_{SEED_42}.npy",
                ART / f"test_{SEED_42}_singleseed_backup.npy")
    print(f"  backup saved: {backup.name}")
else:
    print(f"  backup already exists: {backup.name}")

shutil.copy(ART / f"oof_rawashishsin{SUFFIX_BAG}.npy",
            ART / f"oof_{SEED_42}.npy")
shutil.copy(ART / f"test_rawashishsin{SUFFIX_BAG}.npy",
            ART / f"test_{SEED_42}.npy")
print(f"  oof_{SEED_42}.npy and test_... NOW point to bag5")

# === Retrain RF natural meta with bag5 input on v1's EXACT 7-component bank ===
# Use n1_rf_natural_v1bank.py instead of sklearn_rf_meta_natural.py because
# the latter has the 11-component a1lgbm bank (LB 0.98098 regression),
# not v1's 7-component bank (LB 0.98129).
print("\n=== retrain n1_rf_natural_v1bank.py (META_SUFFIX=_v1bank_bag5) ===")
import os
env = os.environ.copy()
env["META_SUFFIX"] = "_v1bank_bag5"
proc = subprocess.run(
    ["python3", "scripts/n1_rf_natural_v1bank.py"],
    env=env, capture_output=True, text=True,
)
if proc.returncode != 0:
    print("[meta retrain FAILED]")
    print(proc.stdout[-2000:])
    print("STDERR:", proc.stderr[-2000:])
    # Restore backup before aborting
    shutil.copy(backup, ART / f"oof_{SEED_42}.npy")
    shutil.copy(ART / f"test_{SEED_42}_singleseed_backup.npy",
                ART / f"test_{SEED_42}.npy")
    print("  restored single-seed backup")
    sys.exit(2)

print(proc.stdout[-3000:])

# === 4-gate vs v1 PRIMARY ===
print("\n=== 4-gate vs v1 PRIMARY (LB 0.98129) ===")
v1_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1_lb98129.npy")
v1_test = np.load(ART / "test_sklearn_rf_meta_natural_v1_lb98129.npy")
new_oof = np.load(ART / "oof_sklearn_rf_meta_natural_v1bank_bag5.npy")
new_test = np.load(ART / "test_sklearn_rf_meta_natural_v1bank_bag5.npy")

v1_bias, v1_score = tune_log_bias(v1_oof, y, prior)
new_bias, new_score = tune_log_bias(new_oof, y, prior)

print(f"  v1   : tuned={v1_score:.5f}  bias={v1_bias.round(3).tolist()}")
print(f"  bag5 : tuned={new_score:.5f}  bias={new_bias.round(3).tolist()}")

v1_pred = (np.log(np.clip(v1_oof, eps, 1.0)) + v1_bias).argmax(1)
new_pred = (np.log(np.clip(new_oof, eps, 1.0)) + new_bias).argmax(1)


def per_class(y, p):
    return [((p == c) & (y == c)).sum() / max((y == c).sum(), 1)
            for c in range(3)]


pc_v1 = per_class(y, v1_pred)
pc_new = per_class(y, new_pred)
delta = [pc_new[k] - pc_v1[k] for k in range(3)]

net_h = int((new_pred == 2).sum() - (v1_pred == 2).sum())
add_h = int(((v1_pred != 2) & (new_pred == 2)).sum())
rem_h = int(((v1_pred == 2) & (new_pred != 2)).sum())
churn = add_h + rem_h
ratio = abs(net_h) / max(churn, 1)

dbal = float(new_score - v1_score)
print(f"\n  Δ tuned bal = {dbal:+.5f}")
print(f"  Δ PCR    = L={delta[0]:+.5f}  M={delta[1]:+.5f}  H={delta[2]:+.5f}")
print(f"  net_H={net_h}  add_H={add_h}  rem_H={rem_h}  ratio={ratio:.3f}")

g1 = dbal >= 2e-4
g2 = all(d >= -5e-4 for d in delta)
g4 = (net_h >= 0) and (ratio >= 0.5 if churn > 0 else True)
drift_max = float(max(abs(d) for d in (new_bias - (-np.log(prior)))))
g_drift = drift_max <= 0.30

print(f"\n  G1 (Δ ≥ +2e-4):    {'PASS' if g1 else 'FAIL'}")
print(f"  G2 (PCR ≥ -5e-4):  {'PASS' if g2 else 'FAIL'}")
print(f"  G4 (net_H≥0+rat≥0.5): {'PASS' if g4 else 'FAIL'}")
print(f"  Drift (≤0.30):     {'PASS' if g_drift else 'FAIL'}  max={drift_max:.3f}")
overall = g1 and g2 and g4 and g_drift
print(f"  OVERALL:           {'PASS — emit candidate' if overall else 'FAIL'}")

# Test predictions
inv = {0: "Low", 1: "Medium", 2: "High"}
test_pred = (np.log(np.clip(new_test, eps, 1.0)) + new_bias).argmax(1)
sub_path = Path("submissions") / "submission_n1_bag5_rf_natural.csv"
test_ids = pd.read_csv("data/test.csv")["id"].values
sub_df = pd.DataFrame({"id": test_ids,
                       "Irrigation_Need": [inv[int(c)] for c in test_pred]})
sub_df.to_csv(sub_path, index=False)
print(f"\n  wrote {sub_path}")

v1_sub = pd.read_csv("submissions/submission_sklearn_rf_meta_natural_standalone_v1_lb98129.csv")
v1_lab = v1_sub["Irrigation_Need"].map({"Low": 0, "Medium": 1, "High": 2}).values
diff = int((test_pred != v1_lab).sum())
print(f"  test diff vs v1 PRIMARY: {diff}")

# Restore single-seed backup so original scripts unaffected
print("\n=== restoring original single-seed rawashishsin_2600 ===")
shutil.copy(backup, ART / f"oof_{SEED_42}.npy")
shutil.copy(ART / f"test_{SEED_42}_singleseed_backup.npy",
            ART / f"test_{SEED_42}.npy")
print(f"  restored")

results = dict(
    n_seeds=len(seed_oof), seeds=sorted(list(seed_oof.keys())),
    v1_score=float(v1_score), bag5_score=float(new_score),
    delta_bal=dbal, delta_pcr=[float(x) for x in delta],
    net_h=net_h, add_h=add_h, rem_h=rem_h, ratio=float(ratio),
    drift_max=drift_max,
    g1=bool(g1), g2=bool(g2), g4=bool(g4), drift_pass=bool(g_drift),
    overall_pass=bool(overall),
    test_diff_vs_v1=diff,
    submission_path=str(sub_path),
)
(ART / "n1_bag5_rf_natural_results.json").write_text(json.dumps(results, indent=2))
print(f"\n[done] scripts/artifacts/n1_bag5_rf_natural_results.json")
