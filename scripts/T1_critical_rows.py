"""T1-v2 — Stage 1: build critical-rows filter on V1-floor TRAIN OOF.

Filter V1-floor (bank_argmax=M & fb_oof=H, ~2,979 rows) by multi-signal
agreement; the LLM job in subsequent stages is to confirm this prior, not
disambiguate from scratch.

Filter axes:
  - aux_missed_high < 0.20    (low H-flip risk)
  - aux_missed_medium < 0.30  (low M-flip-elsewhere risk)
  - knn_margin >= 0.7         (k=100-NN consensus is unambiguous)
  - bank_max_prob >= 0.80     (the bank itself is fairly confident on M)

Output:
  scripts/artifacts/T1_v2/critical_rows.csv
  scripts/artifacts/T1_v2/critical_rows_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import bank_mean_probs, load_bank  # noqa: E402
from T6_diversity_helpers import load_y_train, normed, tune_log_bias_simple  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
OUT = ART / "T1_v2"
OUT.mkdir(parents=True, exist_ok=True)
CLASS_STR = {0: "Low", 1: "Medium", 2: "High"}


def main():
    print("=== T1-v2 Stage 1: critical-rows filter ===\n")
    y = load_y_train()

    # Build 4b OOF analog
    v1_oof = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    raw_oof = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    t1_oof = normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))
    bv1, _ = tune_log_bias_simple(v1_oof, y)
    bra, _ = tune_log_bias_simple(raw_oof, y)
    bt1, _ = tune_log_bias_simple(t1_oof, y)
    a_v1 = (np.log(np.clip(v1_oof, 1e-9, None)) + bv1).argmax(1).astype(np.int8)
    a_ra = (np.log(np.clip(raw_oof, 1e-9, None)) + bra).argmax(1).astype(np.int8)
    a_t1 = (np.log(np.clip(t1_oof, 1e-9, None)) + bt1).argmax(1).astype(np.int8)
    una = a_ra == a_t1
    fb_oof = a_v1.copy()
    fb_oof[una & (a_v1 != a_ra)] = a_ra[una & (a_v1 != a_ra)]

    # 14-bank stats
    bank = load_bank("oof")
    bank_mean = bank_mean_probs(bank)
    bank_argmax = bank_mean.argmax(axis=1).astype(np.int8)
    bank_max_prob = bank_mean.max(axis=1)

    v1_mask = (bank_argmax == 1) & (fb_oof == 2)
    v1_idx = np.where(v1_mask)[0]
    print(f"V1 floor: n={len(v1_idx)}, true-M frac={(y[v1_idx] == 1).mean():.4f}")

    # Per-row signals
    amh = np.load(ART / "oof_aux_missed_high.npy").astype(np.float32)
    amm = np.load(ART / "oof_aux_missed_medium.npy").astype(np.float32)
    afr = np.load(ART / "oof_aux_flipped_from_rule.npy").astype(np.float32)
    knn_features = np.load(ART / "oof_knn_train.npy").astype(np.float32)
    if knn_features.ndim == 2 and knn_features.shape[1] >= 6:
        knn_margin = knn_features[:, 5]
    else:
        print(f"WARNING: oof_knn_train.npy shape={knn_features.shape}, expected (n,>=6)")
        knn_margin = np.zeros(len(y), dtype=np.float32)

    # Build per-row filter masks (on the V1 indices only)
    sub = {
        "row_idx": v1_idx,
        "true_label": [CLASS_STR[c] for c in y[v1_idx]],
        "true_y": y[v1_idx],
        "amh": amh[v1_idx],
        "amm": amm[v1_idx],
        "afr": afr[v1_idx],
        "knn_margin": knn_margin[v1_idx],
        "bank_max_prob": bank_max_prob[v1_idx],
    }
    df = pd.DataFrame(sub)

    df["c_amh"] = df["amh"] < 0.20
    df["c_amm"] = df["amm"] < 0.30
    df["c_knn"] = df["knn_margin"] >= 0.7
    df["c_bnk"] = df["bank_max_prob"] >= 0.80
    df["n_signals_pass"] = df[["c_amh", "c_amm", "c_knn", "c_bnk"]].sum(axis=1)

    print("\nper-axis pass counts on V1-floor:")
    for k in ["c_amh", "c_amm", "c_knn", "c_bnk"]:
        print(f"  {k}: {df[k].sum()}")
    print("\nn_signals_pass histogram:")
    print(df["n_signals_pass"].value_counts().sort_index())

    # Critical rows: ≥3 signals pass
    df["critical"] = df["n_signals_pass"] >= 3
    crit = df[df["critical"]].copy()
    print(f"\ncritical rows (≥3 signals pass): n={len(crit)}")
    if len(crit) > 0:
        print(f"  baseline true-M: {(crit['true_y'] == 1).mean():.4f}  ({int((crit['true_y'] == 1).sum())}/{len(crit)})")
        print(f"  baseline true-H: {(crit['true_y'] == 2).mean():.4f}  ({int((crit['true_y'] == 2).sum())}/{len(crit)})")

    # Tighter filter: ≥4 signals
    crit4 = df[df["n_signals_pass"] >= 4]
    if len(crit4):
        print(f"\nstrict-4 critical rows (≥4 signals pass): n={len(crit4)}")
        print(f"  baseline true-M: {(crit4['true_y'] == 1).mean():.4f}  ({int((crit4['true_y'] == 1).sum())}/{len(crit4)})")

    # Save the ≥3 critical rows for downstream LLM use
    out_csv = OUT / "critical_rows.csv"
    crit_save = crit[["row_idx", "true_label", "amh", "amm", "afr", "knn_margin", "bank_max_prob", "n_signals_pass"]]
    crit_save.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")

    # Save also a "boundary" set: rows JUST OUTSIDE critical (n_signals_pass == 2)
    # to use as few-shot exemplars showing the failure mode
    boundary = df[df["n_signals_pass"] == 2].copy()
    print(f"\nboundary (exactly 2 signals): n={len(boundary)}, "
          f"true-M={int((boundary['true_y'] == 1).sum())}, "
          f"true-H={int((boundary['true_y'] == 2).sum())}")
    boundary[["row_idx", "true_label", "amh", "amm", "afr", "knn_margin", "bank_max_prob"]].to_csv(
        OUT / "boundary_rows.csv", index=False
    )

    # Decision verdict
    if len(crit) == 0:
        verdict = "EMPTY_FILTER"
    elif (crit["true_y"] == 1).mean() >= 0.92:
        verdict = "FILTER_ALONE_CLEARS"
    elif (crit["true_y"] == 1).mean() >= 0.86:
        verdict = "PROCEED_TO_HAIKU"
    else:
        verdict = "FILTER_TOO_LOOSE"

    summary = {
        "n_v1_floor": int(len(v1_idx)),
        "n_critical": int(len(crit)),
        "n_critical_strict4": int(len(crit4)),
        "n_boundary": int(len(boundary)),
        "critical_baseline_true_M": float((crit["true_y"] == 1).mean()) if len(crit) else None,
        "critical_baseline_true_H": float((crit["true_y"] == 2).mean()) if len(crit) else None,
        "strict4_baseline_true_M": float((crit4["true_y"] == 1).mean()) if len(crit4) else None,
        "verdict": verdict,
    }
    (OUT / "critical_rows_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nverdict: {verdict}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
