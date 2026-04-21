"""Fixed-bias log-blend of MLP (NumEmb) into the greedy stack.

Two separate tests:
  A) Blend MLP directly with greedy (oof_greedy_blend.npy).
  B) Blend MLP + nonrule into greedy as 2-param sweep, to see if
     the MLP signal survives when nonrule is already in the stack
     (nonrule is our current LB-best lever at +0.00056).

Both use FIXED log-bias from the greedy tuned baseline — no
retune-on-top, which is the rule we added after the binhigh overfit.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
ART = Path("scripts/artifacts")

y = pd.read_csv("data/train.csv")[TARGET].map(CLS2IDX).to_numpy(dtype=np.int64)
prior = np.bincount(y) / len(y)

greedy = np.load(ART / "oof_greedy_blend.npy")          # OOF 0.97375 tuned
mlp    = np.load(ART / "oof_mlp_numemb.npy")
nonrule = np.load(ART / "oof_xgb_nonrule.npy")

# --- reproduce greedy tuned bias via coord-ascent on greedy alone ---
def tune_bias(p, y, prior):
    log_p = np.log(np.clip(p, 1e-9, 1.0))
    bias = -np.log(prior)
    best = balanced_accuracy_score(y, (log_p + bias).argmax(1))
    grid = np.linspace(-2.5, 4.5, 71)
    for _ in range(25):
        imp = False
        for k in range(3):
            base = bias.copy()
            scores = []
            for g in grid:
                base[k] = bias[k] + g
                scores.append(balanced_accuracy_score(y, (log_p + base).argmax(1)))
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-6:
                bias[k] = bias[k] + grid[j]; best = scores[j]; imp = True
        if not imp: break
    return bias, best

g_bias, g_best = tune_bias(greedy, y, prior)
print(f"greedy tuned (reproduction): bal_acc={g_best:.5f}  bias={g_bias.round(4).tolist()}")

# --- A) fixed-bias log-blend: greedy + alpha * mlp ---
log_g = np.log(np.clip(greedy, 1e-9, 1.0))
log_m = np.log(np.clip(mlp,    1e-9, 1.0))
log_n = np.log(np.clip(nonrule,1e-9, 1.0))

print("\n[A] greedy + alpha * mlp  (fixed greedy bias)")
A = []
for a in np.arange(0.0, 0.51, 0.05):
    pred = ((1-a)*log_g + a*log_m + g_bias).argmax(1)
    sc = balanced_accuracy_score(y, pred)
    A.append((float(a), float(sc)))
    print(f"  alpha={a:.2f}  bal_acc={sc:.5f}  delta={sc-g_best:+.5f}")

# --- B) greedy + nonrule @ 0.15 (current LB best) + alpha * mlp ---
print("\n[B] greedy + 0.15*nonrule + alpha * mlp  (fixed greedy bias)")
base_log = 0.85*log_g + 0.15*log_n  # (1-0.15)*greedy + 0.15*nonrule in log space
b0 = balanced_accuracy_score(y, (base_log + g_bias).argmax(1))
print(f"  base (current LB-0.97352 stack reproduction): bal_acc={b0:.5f}")
B = []
for a in np.arange(0.0, 0.41, 0.05):
    pred = ((1-a)*base_log + a*log_m + g_bias).argmax(1)
    sc = balanced_accuracy_score(y, pred)
    B.append((float(a), float(sc)))
    print(f"  alpha_mlp={a:.2f}  bal_acc={sc:.5f}  delta_vs_base={sc-b0:+.5f}  delta_vs_greedy={sc-g_best:+.5f}")

# --- C) error-Jaccard on full OOF (all 630k, not just fold 1) ---
errs_mlp = np.where(mlp.argmax(1) != y)[0]
errs_grd = np.where(greedy.argmax(1) != y)[0]
errs_non = np.where(nonrule.argmax(1) != y)[0]

def jac(a, b):
    A = set(a); B = set(b)
    return len(A & B) / max(1, len(A | B))

print(f"\nerror-Jaccard (MLP  vs greedy)   = {jac(errs_mlp, errs_grd):.4f}  (|E_mlp|={len(errs_mlp)}, |E_greedy|={len(errs_grd)})")
print(f"error-Jaccard (MLP  vs nonrule)  = {jac(errs_mlp, errs_non):.4f}")
print(f"error-Jaccard (greedy vs nonrule)= {jac(errs_grd, errs_non):.4f}")

out = {
    "greedy_tuned_bal_acc": g_best,
    "greedy_bias": g_bias.tolist(),
    "A_sweep_greedy_plus_mlp": A,
    "B_sweep_greedy_nonrule_plus_mlp": B,
    "B_base_bal_acc": float(b0),
    "jaccard_mlp_greedy": jac(errs_mlp, errs_grd),
    "jaccard_mlp_nonrule": jac(errs_mlp, errs_non),
    "jaccard_greedy_nonrule": jac(errs_grd, errs_non),
}
with open(ART / "mlp_blend_test_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote", ART / "mlp_blend_test_results.json")
