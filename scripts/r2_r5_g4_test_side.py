"""Compute G4 (net-rare-class-flip ratio) for R2/R5 submission CSVs vs PRIMARY.

The 4-gate filter G4 is the key predictor: ratio = |net rare-class flip| /
total churn. Threshold 0.5; below = RESHUFFLE → LB regression predicted.

OOF arrays for these candidates aren't on disk, but we can compute G4
purely from test-side predictions vs PRIMARY's submission.

R2/R5 reported OOF deltas (from build_r2_r5_submissions.py docstring):
  r2_heavy_fulliso_a045:        OOF +0.00039 (raw, NOT leak-corrected)
  r2r5_heavy_perfoldiso_a045:   OOF +0.00029 (leak-corrected per-fold iso)
  r2r5_heavy_perfoldiso_a025:   OOF +0.00014 (leak-corrected, safer α)
  meta_heavy_a500 (internal):   OOF +0.00055 (raw, internal best alpha)
"""
from pathlib import Path
import pandas as pd

SUB = Path("submissions")
PRI = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")

CANDS = [
    ("r2_heavy_fulliso_a045",       "submission_r2_heavy_fulliso_a045.csv",
     "+0.00039 (raw)",       "+0.00026 (leak-corr)"),
    ("r2r5_perfoldiso_a045",        "submission_r2r5_heavy_perfoldiso_a045.csv",
     "+0.00029 (leak-corr)",  "+0.00029"),
    ("r2r5_perfoldiso_a025_safe",   "submission_r2r5_heavy_perfoldiso_a025_safe.csv",
     "+0.00014 (leak-corr)",  "+0.00014"),
    ("meta_heavy_a500_internal",    "submission_tier1b_metastack_meta_heavy_a500.csv",
     "+0.00055 (raw)",        "+0.00042 (leak-corr)"),
]

print(f"{'candidate':35s}  {'OOF Δ (raw)':>20s}  {'leak-corr':>20s}  "
      f"{'to_H':>5s}/{'from_H':>6s}  {'net':>4s}/{'churn':>5s}  "
      f"{'G4 ratio':>9s}  {'G1':>3s}{'G4':>3s}")
for name, fname, oof_raw, oof_corr in CANDS:
    p = SUB / fname
    if not p.exists():
        print(f"{name:35s}  MISSING")
        continue
    sub = pd.read_csv(p)
    assert (sub["id"].values == PRI["id"].values).all()
    cand = sub["Irrigation_Need"].values
    pri = PRI["Irrigation_Need"].values
    n_to_h = int(((cand == "High") & (pri != "High")).sum())
    n_from_h = int(((cand != "High") & (pri == "High")).sum())
    churn = n_to_h + n_from_h
    net = n_to_h - n_from_h
    ratio = abs(net) / max(1, churn)
    # Quick G1 check using leak-corrected OOF Δ string
    leak_corr_val = float(oof_corr.split()[0].replace("+", ""))
    g1 = "Y" if leak_corr_val >= 3e-4 else "·"
    g4 = "Y" if ratio >= 0.5 else "·"
    n_diff = int((cand != pri).sum())
    print(f"{name:35s}  {oof_raw:>20s}  {oof_corr:>20s}  "
          f"{n_to_h:>5d}/{n_from_h:>6d}  {net:+4d}/{churn:>5d}  "
          f"{ratio:>9.3f}  {g1:>3s}{g4:>3s}  ({n_diff} total diff)")

print()
print("Decision: candidates passing both G1+G4 (leak-corrected) are LB-probe-worthy.")
print("RESHUFFLE-class (G4 < 0.5) candidates are predicted to LB-regress at -1.0x to -1.6x ratio.")
