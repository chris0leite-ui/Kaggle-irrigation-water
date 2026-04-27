"""Build mlp_metastack standalone at α=0.30 in LB-validated PRIMARY arch.

primary' = 0.7 × LB-best 3-stack + 0.30 × mlp_metastack_iso

The 4-gate sweep showed this is the only candidate to pass G1+G2+G3
(borderline G4 at 0.357). After R5's iso-leak correction (+0.00016 OOF
inflation), the leak-corrected OOF Δ ≈ +0.00017 (sub-G1). LB prediction
at -0.5x small-α carryover: LB ≈ 0.98086 (regression by ~-0.00008).

User-approved confirmation submit to validate the leak-correction +
4-gate filter combined prediction.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed

ART = Path("scripts/artifacts")
SUB = Path("submissions")
BIAS = np.array([1.4324, 1.4689, 3.4008], dtype=np.float32)
LABELS = ["Low", "Medium", "High"]


def log_blend(probs_list, weights):
    s = np.zeros_like(probs_list[0])
    for p, w in zip(probs_list, weights):
        s = s + w * np.log(np.clip(p, 1e-12, 1))
    e = np.exp(s - s.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


y = load_y()
s3_o, s3_t = build_lbbest_stack(y)
mlp_o = normed(np.load(ART / "oof_mlp_metastack.npy"))
mlp_t = normed(np.load(ART / "test_mlp_metastack.npy"))
mlp_iso_o, mlp_iso_t = iso_cal(mlp_o, mlp_t, y)

p_t = log_blend([s3_t, mlp_iso_t], np.array([0.70, 0.30]))
pred_t = (np.log(np.clip(p_t, 1e-12, 1)) + BIAS).argmax(1)

# Diff vs PRIMARY
v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
_, v1_iso_t = iso_cal(v1_o, v1_t, y)
p_v1_t = log_blend([s3_t, v1_iso_t], np.array([0.70, 0.30]))
pred_v1_t = (np.log(np.clip(p_v1_t, 1e-12, 1)) + BIAS).argmax(1)
n_diff = int((pred_t != pred_v1_t).sum())

n_to_h = int(((pred_t == 2) & (pred_v1_t != 2)).sum())
n_from_h = int(((pred_t != 2) & (pred_v1_t == 2)).sum())
churn = n_to_h + n_from_h
ratio = abs(n_to_h - n_from_h) / max(1, churn)

test_df = pd.read_csv("data/test.csv")
sub = pd.DataFrame({"id": test_df["id"].values,
                     "Irrigation_Need": [LABELS[i] for i in pred_t]})
fname = "submission_mlp_metastack_a030.csv"
sub.to_csv(SUB / fname, index=False)
print(f"Saved {fname}")
print(f"  test diff vs PRIMARY: {n_diff}")
print(f"  to-High={n_to_h}  from-High={n_from_h}  net={n_to_h - n_from_h}  ratio={ratio:.3f}")
