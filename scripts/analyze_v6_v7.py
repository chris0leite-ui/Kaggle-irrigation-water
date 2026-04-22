"""Fixed-bias log-blend sweeps for v6 (MLP nonrule) and v7 (MLP top-3).

Tests whether either MLP-on-sliced-features variant lifts greedy OR
greedy+0.15*nonrule (current LB best 0.97352).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score

TARGET = "Irrigation_Need"
CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
ART = Path("scripts/artifacts")


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

greedy = np.load(ART / "oof_greedy_blend.npy")
nonrule = np.load(ART / "oof_xgb_nonrule.npy")
v6 = np.load(ART / "oof_mlp_v6_nonrule.npy")
v7 = np.load(ART / "oof_mlp_v7_top3.npy")

g_bias, g_best = tune_bias(greedy, y, prior)
print(f"greedy tuned OOF: {g_best:.5f}  bias={g_bias.round(4).tolist()}  (LB 0.97296)")
base_log = 0.85*np.log(np.clip(greedy,1e-9,1)) + 0.15*np.log(np.clip(nonrule,1e-9,1))
base_bal = balanced_accuracy_score(y, (base_log + g_bias).argmax(1))
print(f"greedy+0.15*nonrule fixed-bias: {base_bal:.5f}  (LB 0.97352 — current best)")

log_g = np.log(np.clip(greedy, 1e-9, 1))
log_n = np.log(np.clip(nonrule, 1e-9, 1))

def errs(a): return set(np.where(a.argmax(1) != y)[0])
def jac(A, B): return len(A & B) / max(1, len(A | B))

for name, mlp in [("v6_nonrule", v6), ("v7_top3", v7)]:
    print(f"\n=== {name} ===")
    log_m = np.log(np.clip(mlp, 1e-9, 1))
    e_m = errs(mlp)
    print(f"  full-OOF |E_mlp|={len(e_m)}  Jaccard vs greedy={jac(e_m, errs(greedy)):.4f}  "
          f"vs nonrule={jac(e_m, errs(nonrule)):.4f}")

    print(f"  greedy + alpha * {name} (fixed greedy bias):")
    for a in np.arange(0.0, 0.41, 0.05):
        pred = ((1-a)*log_g + a*log_m + g_bias).argmax(1)
        bal = balanced_accuracy_score(y, pred)
        flag = "  ***" if bal > g_best + 1e-5 else ""
        print(f"    alpha={a:.2f}  bal_acc={bal:.5f}  Δ_vs_greedy={bal-g_best:+.5f}{flag}")

    print(f"  greedy + 0.15*nonrule + alpha * {name} (fixed greedy bias):")
    for a in np.arange(0.0, 0.41, 0.05):
        pred = ((1-a)*base_log + a*log_m + g_bias).argmax(1)
        bal = balanced_accuracy_score(y, pred)
        flag = "  ***" if bal > base_bal + 1e-5 else ""
        print(f"    alpha_{name.split('_')[0]}={a:.2f}  bal_acc={bal:.5f}  "
              f"Δ_vs_base={bal-base_bal:+.5f}  Δ_vs_greedy={bal-g_best:+.5f}{flag}")
