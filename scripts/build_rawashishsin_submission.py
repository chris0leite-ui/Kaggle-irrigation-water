#!/usr/bin/env python3
"""Build LB-probe submission from rawashishsin OOF + test (and a blend variant).

Two candidates:
  STANDALONE: rawashishsin alone with its own tuned log-bias
              (matches rawashishsin's own pipeline -> projected LB ~0.9810-0.9813)
  BLEND:      0.70 * LB-best 4-stack + 0.30 * rawashishsin_iso  (fixed recipe bias)
              (potential lift if rawashishsin is structurally diverse)

Reports OOF, errors, per-class recall, Jaccard vs LB-best, projected LB.

Usage:
  python scripts/build_rawashishsin_submission.py [v2|v3]   default v2
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend
from tier1b_helpers import (BIAS, build_lbbest_stack, iso_cal, load_y, normed)
from sklearn.metrics import balanced_accuracy_score

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "v2"
CAND = "rawashishsin" if VARIANT == "v2" else "rawashishsin_2600"

ART = Path("scripts/artifacts")
SUB_DIR = Path("submissions")
DATA = Path("data")

# Map int -> label
INT2LABEL = {0: "Low", 1: "Medium", 2: "High"}

# Load candidate
cand_o = normed(np.load(ART / f"oof_{CAND}.npy").astype(np.float32))
cand_t = normed(np.load(ART / f"test_{CAND}.npy").astype(np.float32))
print(f"Loaded {CAND}: oof={cand_o.shape} test={cand_t.shape}")

y = load_y()
test_ids = pd.read_csv(DATA / "test.csv")["id"].values
sample = pd.read_csv(DATA / "sample_submission.csv")

# === STANDALONE: rawashishsin own tuned bias
def tune_bias(probs, y_true):
    """Coord-ascent log-bias matching rawashishsin's tune_bias func."""
    eps = 1e-15
    log_p = np.log(np.clip(probs, eps, 1.0))
    bias = np.zeros(3, dtype=np.float64)
    best = balanced_accuracy_score(y_true, np.argmax(log_p + bias, axis=1))
    for step in (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002):
        improved = True
        while improved:
            improved = False
            for ci in range(3):
                for d in (-1.0, 1.0):
                    c = bias.copy()
                    c[ci] += d * step
                    s = balanced_accuracy_score(
                        y_true, np.argmax(log_p + c, axis=1))
                    if s > best + 1e-9:
                        bias = c; best = s; improved = True
    return bias, float(best)

own_bias, own_oof = tune_bias(cand_o, y)
own_pred = np.argmax(np.log(np.clip(cand_t, 1e-15, 1.0)) + own_bias, axis=1)  # test predictions
own_oof_pred = np.argmax(np.log(np.clip(cand_o, 1e-15, 1.0)) + own_bias, axis=1)
own_oof_pcr = np.array([(own_oof_pred[y == c] == c).mean() for c in range(3)])
own_errs = (own_oof_pred != y).sum()

print(f"\n=== STANDALONE {CAND} ===")
print(f"  own tuned bias = {own_bias.round(4).tolist()}")
print(f"  OOF tuned      = {own_oof:.6f}")
print(f"  errs (OOF)     = {own_errs}")
print(f"  PCR (OOF) L={own_oof_pcr[0]:.5f} M={own_oof_pcr[1]:.5f} H={own_oof_pcr[2]:.5f}")

# Save standalone submission
sub_standalone = sample.copy()
sub_standalone["Irrigation_Need"] = [INT2LABEL[p] for p in own_pred]
sub_path_standalone = SUB_DIR / f"submission_{CAND}_standalone.csv"
sub_standalone.to_csv(sub_path_standalone, index=False)
print(f"  -> {sub_path_standalone}")
print(f"  test class dist: {sub_standalone['Irrigation_Need'].value_counts().to_dict()}")

# === BLEND: 0.7 × LB-best 4-stack + 0.3 × rawashishsin_iso (fixed recipe bias)
print("\n=== BLEND vs LB-BEST 4-STACK ===")
print(" reconstructing LB-best 4-stack anchor...")
lb3_o, lb3_t = build_lbbest_stack(y)
mv_o = normed(np.load(ART / "oof_xgb_metastack.npy").astype(np.float32))
mv_t = normed(np.load(ART / "test_xgb_metastack.npy").astype(np.float32))
mv_o_iso, mv_t_iso = iso_cal(mv_o, mv_t, y)
lb4_o = log_blend([lb3_o, mv_o_iso], np.array([0.7, 0.3]))
lb4_t = log_blend([lb3_t, mv_t_iso], np.array([0.7, 0.3]))

anchor_pred = np.argmax(np.log(np.clip(lb4_o, 1e-12, 1)) + BIAS, axis=1)
anchor_bal = balanced_accuracy_score(y, anchor_pred)
anchor_errs = (anchor_pred != y).sum()
anchor_pcr = np.array([(anchor_pred[y == c] == c).mean() for c in range(3)])
print(f" LB-best 4-stack OOF = {anchor_bal:.6f}")
print(f"   errs = {anchor_errs}")
print(f"   PCR L={anchor_pcr[0]:.5f} M={anchor_pcr[1]:.5f} H={anchor_pcr[2]:.5f}")

# Iso-cal candidate
cand_o_iso, cand_t_iso = iso_cal(cand_o, cand_t, y)
cand_iso_p = np.argmax(np.log(np.clip(cand_o_iso, 1e-12, 1)) + BIAS, axis=1)
cand_iso_bal = balanced_accuracy_score(y, cand_iso_p)
cand_iso_errs = (cand_iso_p != y).sum()
print(f"\n {CAND} iso @ recipe bias = {cand_iso_bal:.6f} (errs {cand_iso_errs})")

# Jaccard vs anchor
err_anchor = anchor_pred != y
err_cand_iso = cand_iso_p != y
jac = (err_anchor & err_cand_iso).sum() / max(1, (err_anchor | err_cand_iso).sum())
print(f"  Jaccard(cand_iso, lb4) = {jac:.4f}")

# Sweep alpha
print("\n alpha sweep (raw and iso):")
print(f"  {'a':>5}  {'OOF_raw':>8}  {'OOF_iso':>8}  {'errs_iso':>8}  {'recH_iso':>8}")
sweep = {}
for a in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    w = np.array([1.0 - a, a])
    raw_o = log_blend([lb4_o, cand_o], w)
    iso_o = log_blend([lb4_o, cand_o_iso], w)
    raw_t = log_blend([lb4_t, cand_t], w)
    iso_t = log_blend([lb4_t, cand_t_iso], w)
    p_raw = np.argmax(np.log(np.clip(raw_o, 1e-12, 1)) + BIAS, axis=1)
    p_iso = np.argmax(np.log(np.clip(iso_o, 1e-12, 1)) + BIAS, axis=1)
    bal_raw = balanced_accuracy_score(y, p_raw)
    bal_iso = balanced_accuracy_score(y, p_iso)
    errs_iso = (p_iso != y).sum()
    pcr_iso = np.array([(p_iso[y == c] == c).mean() for c in range(3)])
    print(f"  {a:>5.3f}  {bal_raw:.5f}  {bal_iso:.5f}  {errs_iso:>8d}  {pcr_iso[2]:.5f}")
    sweep[a] = {
        "raw_oof": float(bal_raw),
        "iso_oof": float(bal_iso),
        "errs_iso": int(errs_iso),
        "pcr_iso": pcr_iso.tolist(),
    }
    if a == 0.30:
        # Save BLEND submission at alpha=0.30 (LB-validated arch)
        test_pred = np.argmax(np.log(np.clip(iso_t, 1e-12, 1)) + BIAS, axis=1)
        sub_blend = sample.copy()
        sub_blend["Irrigation_Need"] = [INT2LABEL[p] for p in test_pred]
        sub_path_blend = SUB_DIR / f"submission_{CAND}_blend_a030.csv"
        sub_blend.to_csv(sub_path_blend, index=False)

print(f"\n  -> {SUB_DIR / f'submission_{CAND}_blend_a030.csv'}")
print(f"\n=== SUMMARY ===")
print(f"STANDALONE:  OOF tuned = {own_oof:.6f}  -> projected LB ~ {own_oof - 0.0:.4f} +- 0.001")
print(f"            (rawashishsin's own pipeline at LB 0.98132 had gap = -0.00023)")
print(f"BLEND a030:  OOF iso   = {sweep[0.30]['iso_oof']:.6f}")
print(f"            Δ vs LB-best = {sweep[0.30]['iso_oof'] - 0.98084:+.6f}")
print(f"")
print(f"Submissions ready:")
print(f"  {sub_path_standalone}")
print(f"  {SUB_DIR / f'submission_{CAND}_blend_a030.csv'}")
