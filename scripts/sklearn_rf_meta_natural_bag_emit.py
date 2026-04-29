"""A2 emitter: build candidate from log-mean bag of RF natural seeds.

Loads bag artifacts from sklearn_rf_meta_natural_bag.py, compares against
the LB 0.98129 PRIMARY (single-seed RF natural standalone) and the prior
LB 0.98094 PRIMARY (Tier-1b 4-stack), reports diagnostics, emits candidate
submission CSV.

NO LB submission — per CLAUDE.md, await user approval.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from common import tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
TARGET = "Irrigation_Need"
CLS_MAP = {"Low": 0, "Medium": 1, "High": 2}
IDX2CLS = {v: k for k, v in CLS_MAP.items()}


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def safelog(p, eps=1e-9):
    return np.log(np.clip(p, eps, 1.0))


def per_class_recall(y, pred, n_class=3):
    rec = np.zeros(n_class, dtype=np.float64)
    for k in range(n_class):
        mask = y == k
        if mask.sum() > 0:
            rec[k] = (pred[mask] == k).sum() / mask.sum()
    return rec


def main():
    log("loading bag artifacts")
    bag_oof = np.load(ART / "oof_sklearn_rf_meta_natural_bag.npy").astype(np.float32)
    bag_test = np.load(ART / "test_sklearn_rf_meta_natural_bag.npy").astype(np.float32)

    log("loading train/test for y + ids")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train[TARGET].map(CLS_MAP).to_numpy().astype(np.int32)
    test_ids = test["id"].values

    prior = np.bincount(y, minlength=3) / len(y)
    bag_argmax = balanced_accuracy_score(y, bag_oof.argmax(1))
    bias_bag, tuned_bag = tune_log_bias(bag_oof, y, prior)
    pred_bag = (safelog(bag_oof) + bias_bag).argmax(1)
    pcr_bag = per_class_recall(y, pred_bag)
    errs_bag = (pred_bag != y).sum()
    log(f"bag: argmax={bag_argmax:.5f}  tuned={tuned_bag:.5f}")
    log(f"  bias={bias_bag.round(4).tolist()}  errs={int(errs_bag)}")
    log(f"  PCR=[L={pcr_bag[0]:.4f} M={pcr_bag[1]:.4f} H={pcr_bag[2]:.4f}]")

    # Reference: LB 0.98129 (seed=42 RF natural standalone)
    log("=== vs LB 0.98129 (RF natural seed=42 standalone) ===")
    legacy_oof = np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32)
    legacy_test = np.load(ART / "test_sklearn_rf_meta_natural.npy").astype(np.float32)
    bias_legacy, tuned_legacy = tune_log_bias(legacy_oof, y, prior)
    pred_legacy = (safelog(legacy_oof) + bias_legacy).argmax(1)
    pcr_legacy = per_class_recall(y, pred_legacy)
    errs_legacy = (pred_legacy != y).sum()
    log(f"  legacy: tuned={tuned_legacy:.5f}  bias={bias_legacy.round(4).tolist()}  errs={int(errs_legacy)}")
    log(f"  legacy PCR=[L={pcr_legacy[0]:.4f} M={pcr_legacy[1]:.4f} H={pcr_legacy[2]:.4f}]")
    pcr_delta_legacy = (pcr_bag - pcr_legacy)
    log(f"  Δ tuned = {tuned_bag - tuned_legacy:+.5f}")
    log(f"  Δ PCR  = [L={pcr_delta_legacy[0]:+.5f} M={pcr_delta_legacy[1]:+.5f} H={pcr_delta_legacy[2]:+.5f}]")
    log(f"  Δ errs = {int(errs_bag) - int(errs_legacy):+d}")

    # Test-side prediction at tuned bias for the candidate submission
    test_pred_bag = (safelog(bag_test) + bias_bag).argmax(1)
    test_pred_legacy = (safelog(legacy_test) + bias_legacy).argmax(1)
    rows_diff_legacy = (test_pred_bag != test_pred_legacy).sum()
    net_h_legacy = int(((test_pred_bag == 2) & (test_pred_legacy != 2)).sum() -
                       ((test_pred_legacy == 2) & (test_pred_bag != 2)).sum())
    log(f"  test rows diff vs LB 0.98129: {int(rows_diff_legacy)}/{len(test_ids)}")
    log(f"  net_H flip vs LB 0.98129: {net_h_legacy:+d}")

    # Reference: LB 0.98094 prior PRIMARY (Tier-1b 4-stack) for cross-family check
    legacy_4stack_csv = SUB / "submission_tier1b_greedy_meta.csv"
    if legacy_4stack_csv.exists():
        log("=== vs LB 0.98094 (Tier-1b 4-stack prior PRIMARY) ===")
        sub_4stack = pd.read_csv(legacy_4stack_csv)
        labels_4stack = sub_4stack[TARGET].map(CLS_MAP).to_numpy()
        rows_diff_4stack = (test_pred_bag != labels_4stack).sum()
        net_h_4stack = int(((test_pred_bag == 2) & (labels_4stack != 2)).sum() -
                           ((labels_4stack == 2) & (test_pred_bag != 2)).sum())
        log(f"  test rows diff vs LB 0.98094: {int(rows_diff_4stack)}/{len(test_ids)}")
        log(f"  net_H flip vs LB 0.98094: {net_h_4stack:+d}")
    else:
        log("  WARN: tier1b_greedy_meta.csv not found, skipping")

    # Per-seed diagnostic from the bag's results JSON (if exists)
    summary_p = ART / "sklearn_rf_meta_natural_bag_results.json"
    per_seed_info = None
    if summary_p.exists():
        s = json.loads(summary_p.read_text())
        per_seed_info = s.get("per_seed_tuned", {})
        log(f"=== per-seed standalone tuned ({len(per_seed_info)} seeds) ===")
        for sd, (b, t) in per_seed_info.items():
            log(f"  seed={sd}: tuned={t:.5f}  bias={[round(v, 4) for v in b]}")

    # Emit candidate
    out_path = SUB / "submission_sklearn_rf_meta_natural_bag.csv"
    sub = pd.DataFrame({
        "id": test_ids,
        TARGET: [IDX2CLS[i] for i in test_pred_bag],
    })
    sub.to_csv(out_path, index=False)
    log(f"wrote candidate {out_path}")
    log(f"  class counts: " + str(sub[TARGET].value_counts().to_dict()))

    summary_emit = dict(
        bag_tuned=float(tuned_bag),
        bag_bias=bias_bag.tolist(),
        bag_errs=int(errs_bag),
        bag_PCR=pcr_bag.tolist(),
        legacy_tuned=float(tuned_legacy),
        legacy_bias=bias_legacy.tolist(),
        legacy_errs=int(errs_legacy),
        legacy_PCR=pcr_legacy.tolist(),
        delta_tuned_vs_legacy=float(tuned_bag - tuned_legacy),
        delta_PCR_vs_legacy=pcr_delta_legacy.tolist(),
        delta_errs_vs_legacy=int(errs_bag) - int(errs_legacy),
        test_rows_diff_vs_legacy=int(rows_diff_legacy),
        net_h_vs_legacy=net_h_legacy,
        candidate_csv=str(out_path),
    )
    out_p = ART / "sklearn_rf_meta_natural_bag_emit_results.json"
    out_p.write_text(json.dumps(summary_emit, indent=2, default=float))
    log(f"wrote {out_p}")


if __name__ == "__main__":
    main()
