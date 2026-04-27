"""Build classw α=0.40 submission for the carryover-test LB probe."""
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
classw_o = normed(np.load(ART / "oof_xgb_metastack_classw.npy"))
classw_t = normed(np.load(ART / "test_xgb_metastack_classw.npy"))
_, classw_iso_t = iso_cal(classw_o, classw_t, y)

p_t_40 = log_blend([s3_t, classw_iso_t], np.array([0.60, 0.40]))
pred_40 = (np.log(np.clip(p_t_40, 1e-12, 1)) + BIAS).argmax(1)

# Diff vs PRIMARY
v1_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
v1_t = normed(np.load(ART / "test_xgb_metastack.npy"))
_, v1_iso_t = iso_cal(v1_o, v1_t, y)
p_v1_t = log_blend([s3_t, v1_iso_t], np.array([0.70, 0.30]))
pred_v1_t = (np.log(np.clip(p_v1_t, 1e-12, 1)) + BIAS).argmax(1)
n_diff = int((pred_40 != pred_v1_t).sum())

test_df = pd.read_csv("data/test.csv")
sub = pd.DataFrame({"id": test_df["id"].values,
                     "Irrigation_Need": [LABELS[i] for i in pred_40]})
fname = "submission_classw_a040.csv"
sub.to_csv(SUB / fname, index=False)
print(f"Saved {fname}, test_diff vs PRIMARY: {n_diff}")
