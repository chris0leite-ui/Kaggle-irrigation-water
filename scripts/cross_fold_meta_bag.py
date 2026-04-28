"""Cross-fold-disjoint meta-stacker bag.

Tests whether the LB-best XGB-meta's lift survives when we average across
DIFFERENT meta CV seeds. Each seed produces (oof_xgb_metastack_fsX_curated.npy,
test_xgb_metastack_fsX_curated.npy). Test predictions are averaged in
probability space → iso-cal'd → substituted into LB-best 4-stack architecture
at α=0.30 (the LB-validated weight that produced LB 0.98094).

Mechanism distinct from prior nulls:
  - Standard meta uses bases (seed=42 OOFs) and meta CV (seed=42) — same
    fold partition. Cross-stack leak: base predictions for meta training
    rows encode info about meta val labels (CLAUDE.md leak-free macrorec
    queue note).
  - Cross-fold-disjoint meta uses bases (seed=42) and meta CV (seed=7 or
    seed=123) — different partitions. Leak is reduced because seed=7's
    val fold is a different row partition than seed=42's training folds.
  - Averaging 3 metas (seed=42 + 7 + 123) dilutes per-seed leak patterns.

For OOF evaluation: each seed's OOF is in its own fold space. We can't
average them directly. Instead, we evaluate:
  (a) standalone tuned bal_acc per seed (each in its own fold space)
  (b) test-side argmax agreement between seeds (correlation diagnostic)
  (c) blend gate of TEST-AVERAGED meta into LB-best 4-stack (deployment
      candidate); we use seed=42's OOF for the gate calculation since that's
      where the LB-best stack lives.

If standalone tuned per-seed are similar AND test argmax agreement is high
(≥99%), the meta is seed-robust → LB-best lift is real. If they differ
materially, partial leak. Either way, the bagged test prediction is the
deployment candidate.
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
from common import log_blend, CLS2IDX  # noqa: E402
from tier1b_xgb_metastack import (  # noqa: E402
    BIAS, build_lbbest_stack, iso_cal, _normed,
)

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
TARGET = "Irrigation_Need"
EPS = 1e-12
SEEDS = [42, 7, 123]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bal(p, y, bias=BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, EPS, 1.0)) + bias).argmax(1))


def per_class_recall(y, pred):
    out = np.zeros(3)
    for c in range(3):
        m = y == c
        out[c] = (pred[m] == c).mean() if m.any() else 0.0
    return out


def main():
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map(CLS2IDX).to_numpy().astype(np.int32)

    log("=" * 60)
    log("CROSS-FOLD-DISJOINT META-BAG ANALYSIS")
    log("=" * 60)

    # Load each seed's meta. seed=42 is the LB-best (no suffix); 7/123 use _fsX_curated.
    seed_paths = {
        42: ("xgb_metastack", "xgb_metastack"),  # original LB-best (full bank)
        7: ("xgb_metastack_fs7_curated", "xgb_metastack_fs7_curated"),
        123: ("xgb_metastack_fs123_curated", "xgb_metastack_fs123_curated"),
    }
    metas = {}
    for s, (oof_name, test_name) in seed_paths.items():
        oof_p = ART / f"oof_{oof_name}.npy"
        test_p = ART / f"test_{test_name}.npy"
        if not oof_p.exists():
            log(f"  WARNING: {oof_p} missing — skipping seed={s}")
            continue
        oof = _normed(np.load(oof_p).astype(np.float32))
        test = _normed(np.load(test_p).astype(np.float32))
        metas[s] = (oof, test)
        argmax_bal = balanced_accuracy_score(y, oof.argmax(1))
        tuned_bal = bal(oof, y)
        log(f"  seed={s}: standalone argmax={argmax_bal:.5f}  "
            f"tuned@recipe-bias={tuned_bal:.5f}")

    if 42 not in metas:
        log("ERROR: seed=42 metastack required as anchor")
        return
    if len(metas) < 2:
        log(f"ERROR: need ≥2 metas; have {len(metas)}")
        return

    log("\n" + "=" * 60)
    log("LB-BEST ANCHOR")
    log("=" * 60)
    lb3_o, lb3_t = build_lbbest_stack(y)
    lb3_bal = bal(lb3_o, y)
    log(f"  LB-best 3-stack OOF tuned bal_acc = {lb3_bal:.5f}")

    # LB-best 4-stack reconstruction: lb_3stack + meta_iso × α=0.30
    iso_42_oof, iso_42_test = iso_cal(metas[42][0], metas[42][1], y)
    lb4_o = log_blend([lb3_o, iso_42_oof], np.array([0.7, 0.3]))
    lb4_t = log_blend([lb3_t, iso_42_test], np.array([0.7, 0.3]))
    lb4_bal = bal(lb4_o, y)
    log(f"  LB-best 4-stack OOF tuned bal_acc = {lb4_bal:.5f}  "
        f"(documented 0.98084)")

    # Cross-meta argmax agreement on test (diagnostic).
    log("\n" + "=" * 60)
    log("CROSS-META TEST ARGMAX AGREEMENT")
    log("=" * 60)
    seeds_present = sorted(metas.keys())
    n_test = metas[42][1].shape[0]
    test_argmaxes = {s: metas[s][1].argmax(1) for s in seeds_present}
    for i, s1 in enumerate(seeds_present):
        for s2 in seeds_present[i + 1:]:
            agree = (test_argmaxes[s1] == test_argmaxes[s2]).mean()
            disagree = int((test_argmaxes[s1] != test_argmaxes[s2]).sum())
            log(f"  seed{s1} vs seed{s2}: agree={agree:.4f}  disagree_rows={disagree}")

    # Build cross-fold-disjoint meta bag: average iso-cal'd test predictions.
    log("\n" + "=" * 60)
    log("CROSS-FOLD META-BAG (test-prediction average)")
    log("=" * 60)
    iso_metas_test = []
    iso_metas_oof = []  # NOTE: each in its own fold space — used for diagnostic only
    for s in seeds_present:
        io, it = iso_cal(metas[s][0], metas[s][1], y)
        iso_metas_oof.append(io)
        iso_metas_test.append(it)
        log(f"  seed={s}: iso-cal'd standalone tuned = {bal(io, y):.5f}")

    # Bag test predictions (probability mean).
    bag_test = np.mean(iso_metas_test, axis=0)
    bag_test = bag_test / bag_test.sum(1, keepdims=True)
    # For OOF blend evaluation, use the seed=42-aligned bag: that means
    # iso_42_oof × N_seeds equivalent. We average all iso OOFs (each seed's
    # OOF for ITS own fold rows is leak-free) — this is a loose approximation
    # since the OOFs aren't directly comparable, but for a diagnostic it's
    # informative.
    bag_oof = np.mean(iso_metas_oof, axis=0)
    bag_oof = bag_oof / bag_oof.sum(1, keepdims=True)
    bag_oof_tuned = bal(bag_oof, y)
    log(f"  bag iso OOF tuned (mean of seed-OOFs) = {bag_oof_tuned:.5f}")

    # Save the bagged meta for downstream blending.
    np.save(ART / "oof_xgb_metastack_xfold_bag.npy", bag_oof.astype(np.float32))
    np.save(ART / "test_xgb_metastack_xfold_bag.npy", bag_test.astype(np.float32))
    log(f"  saved oof/test_xgb_metastack_xfold_bag.npy")

    # Substitute the bag into LB-best 4-stack architecture: lb_3stack +
    # bag_iso × α=0.30. Compare to LB-best 4-stack standalone.
    log("\n" + "=" * 60)
    log("SUBSTITUTE BAG INTO LB-BEST 4-STACK ARCH (α=0.30)")
    log("=" * 60)
    sub_o = log_blend([lb3_o, bag_oof], np.array([0.7, 0.3]))
    sub_t = log_blend([lb3_t, bag_test], np.array([0.7, 0.3]))
    sub_bal = bal(sub_o, y)
    log(f"  bag-substituted 4-stack OOF tuned = {sub_bal:.5f}")
    log(f"  Δ vs LB-best 4-stack (0.98084)    = {sub_bal - lb4_bal:+.5f}")

    sub_pred = (np.log(np.clip(sub_o, EPS, 1.0)) + BIAS).argmax(1)
    pcr_sub = per_class_recall(y, sub_pred)
    lb4_pred = (np.log(np.clip(lb4_o, EPS, 1.0)) + BIAS).argmax(1)
    pcr_lb4 = per_class_recall(y, lb4_pred)
    log(f"  PCR sub: L={pcr_sub[0]:.5f} M={pcr_sub[1]:.5f} H={pcr_sub[2]:.5f}")
    log(f"  PCR lb4: L={pcr_lb4[0]:.5f} M={pcr_lb4[1]:.5f} H={pcr_lb4[2]:.5f}")
    log(f"  Δ PCR:  L={pcr_sub[0]-pcr_lb4[0]:+.5f} "
        f"M={pcr_sub[1]-pcr_lb4[1]:+.5f} H={pcr_sub[2]-pcr_lb4[2]:+.5f}")

    # 4-gate analysis: substitute architecture (NOT additive — it replaces seed=42 meta).
    log("\n" + "=" * 60)
    log("BLEND GATE: stack-on-top onto LB-best 4-stack (α-sweep)")
    log("=" * 60)
    log(f"  {'α':>6} {'OOF':>9} {'Δ vs LB4':>10} {'recH':>8}")
    rows = []
    for a in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        blend_o = log_blend([lb4_o, bag_oof], np.array([1 - a, a]))
        blend_bal = bal(blend_o, y)
        d = blend_bal - lb4_bal
        pred_b = (np.log(np.clip(blend_o, EPS, 1.0)) + BIAS).argmax(1)
        rec_h = ((pred_b == 2) & (y == 2)).sum() / (y == 2).sum()
        rows.append({"alpha": a, "oof": float(blend_bal),
                     "delta": float(d), "recH": float(rec_h)})
        log(f"  {a:>6.3f} {blend_bal:>9.5f} {d:>+10.5f} {rec_h:>8.5f}")

    out = dict(
        seeds=seeds_present,
        per_seed_iso_oof_tuned={
            str(s): float(bal(iso_cal(metas[s][0], metas[s][1], y)[0], y))
            for s in seeds_present
        },
        bag_oof_tuned=float(bag_oof_tuned),
        substitute_4stack_oof=float(sub_bal),
        substitute_delta_vs_lb4=float(sub_bal - lb4_bal),
        substitute_pcr=pcr_sub.tolist(),
        lb4_pcr=pcr_lb4.tolist(),
        cross_seed_test_disagreement={
            f"{s1}_vs_{s2}": int(
                (test_argmaxes[s1] != test_argmaxes[s2]).sum()
            )
            for i, s1 in enumerate(seeds_present)
            for s2 in seeds_present[i + 1:]
        },
        stack_on_top_sweep=rows,
    )
    out_path = ART / "cross_fold_meta_bag_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    log(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
