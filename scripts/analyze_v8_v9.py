"""Post-hoc analysis of v8 (MLP spec {6,7,8}) and v9 (MLP routed).

v8 must be evaluated via override-on-domain — standalone full-OOF
bal_acc is meaningless since v8 only saw score-{6,7,8} rows.
v9 gets a fixed-bias log-blend sweep into greedy + greedy+nonrule.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
ART = Path("scripts/artifacts")
ACTIVE = ("Flowering", "Vegetative")


def dgp_score_vec(df):
    sm = df["Soil_Moisture"].astype(float).values
    rf = df["Rainfall_mm"].astype(float).values
    tc = df["Temperature_C"].astype(float).values
    ws = df["Wind_Speed_kmh"].astype(float).values
    dry = (sm < 25).astype(np.int8); norain = (rf < 300).astype(np.int8)
    hot = (tc > 30).astype(np.int8); windy = (ws > 10).astype(np.int8)
    nomulch = (df["Mulching_Used"].astype(str).values == "No").astype(np.int8)
    kc = np.where(np.isin(df["Crop_Growth_Stage"].astype(str).values, ACTIVE), 2, 0).astype(np.int8)
    return (2*(dry + norain) + (hot + windy + nomulch) + kc).astype(np.int8)


def tune_bias(p, y, prior):
    log_p = np.log(np.clip(p, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_p + bias).argmax(1))
    grid = np.linspace(-2.5, 4.5, 71)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = bias.copy(); scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_p + base).argmax(1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]; best = scores[j]; imp = True
        if not imp: break
    return bias, best


tr = pd.read_csv("data/train.csv")
y = tr[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
prior = np.bincount(y) / len(y)
score = dgp_score_vec(tr)
mask678 = np.isin(score, [6, 7, 8])
print(f"rows with score in {{6,7,8}}: {mask678.sum()}  (total {len(y)})")

greedy = np.load(ART / "oof_greedy_blend.npy")
nonrule = np.load(ART / "oof_xgb_nonrule.npy")
xgb_spec678 = np.load(ART / "oof_xgb_spec_678.npy")
v8 = np.load(ART / "oof_mlp_v8_spec_678.npy")
v9 = np.load(ART / "oof_mlp_v9_routed_inference.npy")

g_bias, g_best = tune_bias(greedy, y, prior)
print(f"\ngreedy tuned OOF: {g_best:.5f}  bias={g_bias.round(4).tolist()}  (LB ref 0.97296)")

base_log = 0.85*np.log(np.clip(greedy, 1e-9, 1)) + 0.15*np.log(np.clip(nonrule, 1e-9, 1))
base_pred = (base_log + g_bias).argmax(1)
base_bal = balanced_accuracy_score(y, base_pred)
print(f"greedy + 0.15*nonrule fixed-bias OOF: {base_bal:.5f}  (current LB best 0.97352)")

# --- v8 as override on {6,7,8} rows --------------------------------------
print("\n=== v8 as override on {6,7,8} rows ===")
def override_on_mask(base_probs, override_probs, mask):
    out = base_probs.copy()
    out[mask] = override_probs[mask]
    return out

# baseline: greedy already has hybrid_v3 (0.45 weight) which already includes
# xgb_spec_678 override. Re-override with v8 and see if it beats.
for name, ovr, alpha_over_spec in [
    ("v8 full override",                  v8, 1.0),
    ("v8 @ 0.5 mixed with xgb_spec_678",  None, None),  # custom
    ("v8 @ 0.3 mixed with xgb_spec_678",  None, None),
]:
    pass

# cleaner comparison sweep: replace greedy on {6,7,8} with convex combo of v8 + xgb_spec_678
print("  on-domain bal_acc (score-{6,7,8} rows only):")
y678 = y[mask678]
for name, p in [("greedy", greedy[mask678]),
                ("xgb_spec_678", xgb_spec678[mask678]),
                ("v8_mlp", v8[mask678])]:
    print(f"    {name:20s}: argmax bal_acc = {balanced_accuracy_score(y678, p.argmax(1)):.5f}")

# full-OOF: override greedy at {6,7,8} with v8, fixed greedy bias
for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
    mix = a*v8 + (1-a)*xgb_spec678
    ovr = override_on_mask(greedy, mix, mask678)
    pred = (np.log(np.clip(ovr, 1e-9, 1)) + g_bias).argmax(1)
    bal = balanced_accuracy_score(y, pred)
    print(f"  override {{6,7,8}} with α*v8 + (1-α)*xgb_spec678   α={a:.2f}  full OOF bal_acc={bal:.5f}  Δ vs greedy {bal-g_best:+.5f}")

# --- v9 as log-blend leg ------------------------------------------------
print("\n=== v9 (routed-inference MLP) as blend leg ===")
log_g = np.log(np.clip(greedy, 1e-9, 1))
log_n = np.log(np.clip(nonrule, 1e-9, 1))
log_9 = np.log(np.clip(v9, 1e-9, 1))

print("  greedy + alpha*v9 (fixed greedy bias):")
for a in np.arange(0.0, 0.31, 0.05):
    pred = ((1-a)*log_g + a*log_9 + g_bias).argmax(1)
    bal = balanced_accuracy_score(y, pred)
    print(f"    alpha={a:.2f}  bal_acc={bal:.5f}  Δ={bal-g_best:+.5f}")

print("\n  greedy + 0.15*nonrule + alpha*v9 (fixed greedy bias):")
base2 = 0.85*log_g + 0.15*log_n
for a in np.arange(0.0, 0.26, 0.05):
    pred = ((1-a)*base2 + a*log_9 + g_bias).argmax(1)
    bal = balanced_accuracy_score(y, pred)
    print(f"    alpha_v9={a:.2f}  bal_acc={bal:.5f}  Δ_vs_base={bal-base_bal:+.5f}  Δ_vs_greedy={bal-g_best:+.5f}")

# error Jaccards
errs_v9 = set(np.where(v9.argmax(1) != y)[0])
errs_g = set(np.where(greedy.argmax(1) != y)[0])
errs_n = set(np.where(nonrule.argmax(1) != y)[0])
def jac(A, B): return len(A & B) / max(1, len(A | B))
print(f"\nfull-OOF Jaccard (v9 vs greedy): {jac(errs_v9, errs_g):.4f}")
print(f"full-OOF Jaccard (v9 vs nonrule): {jac(errs_v9, errs_n):.4f}")
print(f"  |E_v9|={len(errs_v9)} |E_greedy|={len(errs_g)} |E_nonrule|={len(errs_n)}")
