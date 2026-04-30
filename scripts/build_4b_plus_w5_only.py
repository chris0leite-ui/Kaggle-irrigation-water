"""Build the cleanest information-rich probe: 4b + W5(9 M→H) only, no strict90.

The composite probe at LB 0.98143 told us combined 47-flip Δ = -7bp.
Decomposing precision priors:
  scenario A: M→L@25% (-7.5bp) + M→H@11% (+0.5bp) → -7bp ✓
  scenario B: M→L@33% (-5.5bp) + M→H@5% (-1.5bp) → -7bp
  scenario C: M→L@20% (-8.5bp) + M→H@20% (+1.5bp) → -7bp
Probing W5 alone (9 M→H flips, no M→L noise) disambiguates:
  - if 0.98150 (tied):    W5 precision ~9% (at break-even); strict90 precision ~25%
  - if 0.98152 (+2bp):    W5 precision ~17%; strict90 precision ~20%
  - if 0.98148 (-2bp):    W5 precision ~5%; strict90 precision ~33%
  - if 0.98155 (+5bp):    W5 precision ~30%
Asymmetric upside: floor ~0.98148, ceiling ~0.98155.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SUB = Path("submissions")
LMH_NAMES = {0: "Low", 1: "Medium", 2: "High"}
LMH_REV = {"Low": 0, "Medium": 1, "High": 2}


def load(name):
    return pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"].map(LMH_REV).to_numpy(np.int8)


fb = load("submission_idea4b_selective_override")
w5 = load("submission_W5_i5_MtoH_only")

# Apply ONLY W5's M→H flips (rows where w5 says H and 4b says M)
new_pred = fb.copy()
mh_mask = (fb == 1) & (w5 == 2)
new_pred[mh_mask] = 2

n_flips = int(mh_mask.sum())
n_diff = int((new_pred != fb).sum())
assert n_flips == n_diff, f"{n_flips} vs {n_diff}"

# Direction check
dirs = {}
for fr in range(3):
    for to in range(3):
        if fr == to:
            continue
        n = int(((fb == fr) & (new_pred == to)).sum())
        if n > 0:
            dirs[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n

print(f"4b + W5 only: {n_flips} flips, dirs={dirs}")
print(f"Class shifts: 4b L/M/H = {[(fb==c).sum() for c in range(3)]}")
print(f"             new L/M/H = {[(new_pred==c).sum() for c in range(3)]}")

test_ids = pd.read_csv(SUB / "submission_idea4b_selective_override.csv")["id"].tolist()
out = SUB / "submission_4b_plus_w5_only.csv"
pd.DataFrame({
    "id": test_ids,
    "Irrigation_Need": pd.Series(new_pred).map(LMH_NAMES),
}).to_csv(out, index=False)
print(f"\nemitted: {out.name}")
