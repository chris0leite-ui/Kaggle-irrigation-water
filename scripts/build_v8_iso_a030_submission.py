#!/usr/bin/env python3
"""Build v8 ISO α=0.30 blend onto LB-best 4-stack as a proper LB-probe candidate.

This matches the architecture that the 4-gate analyzer evaluated:
  primary = log_blend(LB-3-stack, xgb_metastack_v8_iso; 0.7, 0.3)

The auto-emitted submission_tier1b_metastack_meta_v8_a500.csv was at α=0.50
vs LB-3stack (NOT 4-stack, NOT iso) — different blend architecture.

Output: submissions/submission_v8_iso_a030.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal, load_y, normed)
from sklearn.metrics import balanced_accuracy_score

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
INT2LABEL = {0: "Low", 1: "Medium", 2: "High"}

# Load
y = load_y()
v8_o = normed(np.load(ART / "oof_xgb_metastack_v8.npy").astype(np.float32))
v8_t = normed(np.load(ART / "test_xgb_metastack_v8.npy").astype(np.float32))
print(f"v8 oof shape: {v8_o.shape}")

# Iso-cal v8
v8_iso_o, v8_iso_t = iso_cal(v8_o, v8_t, y)

# Build LB-best 3-stack
lb3_o, lb3_t = build_lbbest_stack(y)

# Blend at α=0.30 (matches 4-gate analyzer)
ALPHA = 0.30
w = np.array([1.0 - ALPHA, ALPHA])
blend_o = log_blend([lb3_o, v8_iso_o], w)
blend_t = log_blend([lb3_t, v8_iso_t], w)

# Apply fixed recipe bias
log_p_t = np.log(np.clip(blend_t, 1e-12, 1)) + BIAS
test_pred = np.argmax(log_p_t, axis=1)

# OOF check
log_p_o = np.log(np.clip(blend_o, 1e-12, 1)) + BIAS
oof_pred = np.argmax(log_p_o, axis=1)
bal = balanced_accuracy_score(y, oof_pred)
print(f"OOF tuned bal_acc = {bal:.6f}")
print(f"   (4-gate reported +0.00041 over LB-best 4-stack 0.98084 = 0.98125; reproduced: {bal:.5f})")

# Verify shape
sample = pd.read_csv(DATA / "sample_submission.csv")
assert len(sample) == len(test_pred), f"shape mismatch {len(sample)} vs {len(test_pred)}"

sub = sample.copy()
sub["Irrigation_Need"] = [INT2LABEL[p] for p in test_pred]
out_path = SUB / "submission_v8_iso_a030.csv"
sub.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Class dist: {sub['Irrigation_Need'].value_counts().to_dict()}")

# Compare to LB-best primary submission
primary = pd.read_csv(SUB / "submission_tier1b_greedy_meta.csv")
diff_count = (primary["Irrigation_Need"] != sub["Irrigation_Need"]).sum()
print(f"Disagreement vs LB-best primary: {diff_count} / {len(sub)} rows ({100*diff_count/len(sub):.2f}%)")
