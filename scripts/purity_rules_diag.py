"""Deterministic-rule purity diagnostic.

Two parts:
  (1) Score-based: count test rows where primary's argmax disagrees with the
      rule on score in {0, 1, 9} (train rule-error rate ~0%).
  (2) Cell-based: compute per-cell purity over the full 128-cell rule cube
      (2^5 stages-pair * 4 stages = 128). Identify cells with 100% (or 99.99%+)
      train purity AND non-trivial test mass; count primary disagreements
      per pure cell. Project per-cell LB delta if every disagreement is flipped
      to the train-majority class.

  No retraining. Reads:
    - data/train.csv, data/test.csv (rule features)
    - scripts/artifacts/oof_recipe_full_te.npy (+ pseudo_s1, pseudo_s7, realmlp,
      xgb_nonrule, xgb_metastack) to reconstruct primary
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend
from tier1b_helpers import (
    ART, BIAS, CLASSES, CLS2IDX, build_lbbest_stack, iso_cal, load_y, log,
    normed,
)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"


def compute_rule(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Returns dict with: dry, norain, hot, windy, nomulch, kc, score, rule_pred,
    cell_id (0..127). cell_id is a packed integer encoding the full 6-bit/4-stage
    rule state for per-cell aggregation.
    """
    dry = (df["Soil_Moisture"] < 25).to_numpy().astype(np.int8)
    norain = (df["Rainfall_mm"] < 300).to_numpy().astype(np.int8)
    hot = (df["Temperature_C"] > 30).to_numpy().astype(np.int8)
    windy = (df["Wind_Speed_kmh"] > 10).to_numpy().astype(np.int8)
    nomulch = (df["Mulching_Used"].to_numpy() == "No").astype(np.int8)
    stage_map = {"Sowing": 0, "Harvesting": 1, "Flowering": 2, "Vegetative": 3}
    stage_idx = df["Crop_Growth_Stage"].map(stage_map).to_numpy().astype(np.int8)
    kc = ((stage_idx == 2) | (stage_idx == 3)).astype(np.int8) * 2

    score = 2 * (dry + norain) + (hot + windy + nomulch) + kc
    score = score.astype(np.int8)

    rule_pred = np.full(score.shape, -1, dtype=np.int8)
    rule_pred[score <= 3] = 0  # Low
    rule_pred[(score >= 4) & (score <= 6)] = 1  # Medium
    rule_pred[score >= 7] = 2  # High

    # cell_id: pack 5 bits (dry,norain,hot,windy,nomulch) + 2 bits stage = 7 bits
    cell_id = (
        (dry.astype(np.int32) << 6)
        | (norain.astype(np.int32) << 5)
        | (hot.astype(np.int32) << 4)
        | (windy.astype(np.int32) << 3)
        | (nomulch.astype(np.int32) << 2)
        | stage_idx.astype(np.int32)
    )

    return {
        "dry": dry, "norain": norain, "hot": hot, "windy": windy,
        "nomulch": nomulch, "kc": kc, "stage_idx": stage_idx,
        "score": score, "rule_pred": rule_pred, "cell_id": cell_id,
    }


def main():
    log("loading data...")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)
    log(f"train shape={train.shape}, test shape={test.shape}")

    log("computing rule features...")
    rt = compute_rule(train)
    re = compute_rule(test)

    log("reconstructing primary on train+test (LB-best 4-stack: lb3 + RealMLP + nonrule_iso + xgb_metastack_iso a=0.30)...")
    s2_o, s2_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    meta_o, meta_t = iso_cal(meta_o, meta_t, y)
    primary_o = log_blend([s2_o, meta_o], np.array([0.7, 0.3]))
    primary_t = log_blend([s2_t, meta_t], np.array([0.7, 0.3]))

    train_argmax = (np.log(np.clip(primary_o, 1e-12, 1)) + BIAS).argmax(1)
    test_argmax = (np.log(np.clip(primary_t, 1e-12, 1)) + BIAS).argmax(1)
    train_acc = (train_argmax == y).mean()
    log(f"primary train accuracy = {train_acc:.5f}")

    # ---- Sanity: confirm test_argmax matches submission ----
    sub = pd.read_csv(ROOT / "submissions/submission_tier1b_greedy_meta.csv")
    sub_pred = sub["Irrigation_Need"].map(CLS2IDX).to_numpy()
    agree = (sub_pred == test_argmax).mean()
    log(f"primary test argmax matches submitted CSV on {agree*100:.4f}% of rows")

    # ============================================================
    # PART 1: score-based diagnostic (score in {0, 1, 9})
    # ============================================================
    log("=" * 64)
    log("PART 1: score-based diagnostic (deterministic score levels)")
    log("=" * 64)
    out_score = []
    target_class_for_score = {0: 0, 1: 0, 9: 2}  # Low, Low, High
    for sc, target in target_class_for_score.items():
        # train side
        tr_mask = rt["score"] == sc
        tr_n = int(tr_mask.sum())
        tr_y = y[tr_mask]
        tr_match_rule = int((tr_y == target).sum())
        tr_rule_acc = tr_match_rule / max(tr_n, 1)

        # test side
        te_mask = re["score"] == sc
        te_n = int(te_mask.sum())
        primary_target_count = int((test_argmax[te_mask] == target).sum())
        primary_disagree = te_n - primary_target_count
        tr_primary_disagree = int((train_argmax[tr_mask] != target).sum())

        log(f"  score={sc} target={CLASSES[target]}: "
            f"train_n={tr_n} rule_acc={tr_rule_acc:.5f} "
            f"primary_disagree(train)={tr_primary_disagree} "
            f"test_n={te_n} primary_disagree(test)={primary_disagree}")
        out_score.append({
            "score": sc, "target_class": CLASSES[target],
            "train_n": tr_n, "train_rule_acc": tr_rule_acc,
            "train_primary_disagree": tr_primary_disagree,
            "test_n": te_n, "test_primary_disagree": primary_disagree,
        })

    # ============================================================
    # PART 2: per-cell purity over the full 128-cell rule cube
    # ============================================================
    log("=" * 64)
    log("PART 2: per-cell purity over 128 rule cells (5 binary axes x 4 stages)")
    log("=" * 64)

    # Aggregate per (cell_id, y) for train; per (cell_id, primary_argmax) for test.
    cells = sorted(set(rt["cell_id"].tolist()) | set(re["cell_id"].tolist()))
    rows = []
    for cid in cells:
        tr_mask = rt["cell_id"] == cid
        te_mask = re["cell_id"] == cid
        tr_n = int(tr_mask.sum())
        te_n = int(te_mask.sum())
        if tr_n == 0:
            continue
        tr_y_in = y[tr_mask]
        cnt = np.bincount(tr_y_in, minlength=3)
        majority = int(np.argmax(cnt))
        purity = cnt[majority] / tr_n

        # Decode cell_id back to bits/stage
        dry_b = (cid >> 6) & 1
        norain_b = (cid >> 5) & 1
        hot_b = (cid >> 4) & 1
        windy_b = (cid >> 3) & 1
        nomulch_b = (cid >> 2) & 1
        stage_b = cid & 0b11
        kc_b = 2 if stage_b in (2, 3) else 0
        score_b = 2 * (dry_b + norain_b) + (hot_b + windy_b + nomulch_b) + kc_b

        # Test-side primary disagreement count
        primary_majority_count = int((test_argmax[te_mask] == majority).sum())
        primary_disagree_te = te_n - primary_majority_count
        primary_disagree_tr = int((train_argmax[tr_mask] != majority).sum())
        # Train side: if primary already nails it and matches majority
        rule_pred_for_cell = int(rt["rule_pred"][tr_mask][0]) if tr_n > 0 else -1

        rows.append({
            "cell_id": cid,
            "dry": dry_b, "norain": norain_b, "hot": hot_b, "windy": windy_b,
            "nomulch": nomulch_b, "stage": stage_b, "kc": kc_b,
            "score": score_b,
            "rule_pred": CLASSES[rule_pred_for_cell] if rule_pred_for_cell >= 0 else "?",
            "majority": CLASSES[majority],
            "majority_eq_rule": int(majority == rule_pred_for_cell),
            "purity": purity, "tr_n": tr_n, "te_n": te_n,
            "tr_primary_disagree_with_maj": primary_disagree_tr,
            "te_primary_disagree_with_maj": primary_disagree_te,
            "majority_count": int(cnt[majority]),
            "Low": int(cnt[0]), "Medium": int(cnt[1]), "High": int(cnt[2]),
        })

    df_cells = pd.DataFrame(rows).sort_values(
        ["purity", "tr_n"], ascending=[False, False]
    ).reset_index(drop=True)

    pure = df_cells[df_cells["purity"] >= 0.99999]
    near_pure = df_cells[(df_cells["purity"] >= 0.999) & (df_cells["purity"] < 0.99999)]
    log(f"  cells with 100% purity:        {len(pure):3d} (covers {pure['tr_n'].sum()} train rows, {pure['te_n'].sum()} test rows)")
    log(f"  cells with 99.9%-99.99% purity: {len(near_pure):3d} (covers {near_pure['tr_n'].sum()} train rows, {near_pure['te_n'].sum()} test rows)")

    log("")
    log("100%-pure cells (sorted by test_n DESC), with primary disagreements:")
    log(f"  {'cid':>4} {'dnhwm':>5} {'st':>2} {'sc':>2} {'rule':>6} {'maj':>6} "
        f"{'pur':>7} {'tr_n':>7} {'te_n':>7} {'tr_dis':>7} {'te_dis':>7}")
    for r in pure.sort_values("te_n", ascending=False).head(20).itertuples():
        bits = f"{r.dry}{r.norain}{r.hot}{r.windy}{r.nomulch}"
        log(f"  {r.cell_id:>4} {bits:>5} {r.stage:>2} {r.score:>2} "
            f"{r.rule_pred:>6} {r.majority:>6} {r.purity:>7.5f} "
            f"{r.tr_n:>7d} {r.te_n:>7d} "
            f"{r.tr_primary_disagree_with_maj:>7d} {r.te_primary_disagree_with_maj:>7d}")

    log("")
    log("99.9-99.99%-pure cells (sorted by test_primary_disagree DESC), top 20:")
    log(f"  {'cid':>4} {'dnhwm':>5} {'st':>2} {'sc':>2} {'rule':>6} {'maj':>6} "
        f"{'pur':>7} {'tr_n':>7} {'te_n':>7} {'tr_dis':>7} {'te_dis':>7}")
    for r in near_pure.sort_values("te_primary_disagree_with_maj", ascending=False).head(20).itertuples():
        bits = f"{r.dry}{r.norain}{r.hot}{r.windy}{r.nomulch}"
        log(f"  {r.cell_id:>4} {bits:>5} {r.stage:>2} {r.score:>2} "
            f"{r.rule_pred:>6} {r.majority:>6} {r.purity:>7.5f} "
            f"{r.tr_n:>7d} {r.te_n:>7d} "
            f"{r.tr_primary_disagree_with_maj:>7d} {r.te_primary_disagree_with_maj:>7d}")

    # ----- Macro-recall delta projection for 100%-pure overrides -----
    # If every primary-disagreement on a 100%-pure cell is overridden to majority,
    # and we ASSUME test purity matches train purity (~100%), then:
    #   gain_macro = (n_correct_flips_per_class / class_n_test) summed per class / 3
    # We can only project under that assumption; AV showed train/test feature
    # distributions are statistically identical so this is a defensible bound.

    # Test class counts (estimated from primary; will be close enough for projection)
    cls_n_test = np.bincount(test_argmax, minlength=3)
    log("")
    log(f"primary test class distribution (for delta scaling): "
        f"L={cls_n_test[0]} M={cls_n_test[1]} H={cls_n_test[2]}")

    overrides_tr = pure["tr_primary_disagree_with_maj"].sum()
    overrides_te = pure["te_primary_disagree_with_maj"].sum()
    by_class = {}
    for r in pure.itertuples():
        cls = CLS2IDX[r.majority]
        by_class.setdefault(cls, [0, 0])
        by_class[cls][0] += r.tr_primary_disagree_with_maj
        by_class[cls][1] += r.te_primary_disagree_with_maj
    log(f"100%-pure cell overrides: train_total={overrides_tr} test_total={overrides_te}")
    for cls, (tr_d, te_d) in sorted(by_class.items()):
        # projected gain = te_d * (1 - 0) / cls_n_test[cls] / 3 (if all flips correct)
        log(f"  class={CLASSES[cls]}: tr_disagree={tr_d} te_disagree={te_d} -> "
            f"max LB gain if all correct = {te_d / max(cls_n_test[cls], 1) / 3:.6f}")

    # Same for 99.9%+ band
    by_class2 = {}
    for r in near_pure.itertuples():
        cls = CLS2IDX[r.majority]
        by_class2.setdefault(cls, [0, 0, 0])
        by_class2[cls][0] += r.tr_primary_disagree_with_maj
        by_class2[cls][1] += r.te_primary_disagree_with_maj
        # impure rows in this cell: tr_n - majority_count (may flip wrong way)
        by_class2[cls][2] += (r.tr_n - r.majority_count)
    log("")
    log("99.9-99.99%-pure cell overrides:")
    for cls, (tr_d, te_d, tr_imp) in sorted(by_class2.items()):
        log(f"  class={CLASSES[cls]}: tr_disagree={tr_d} te_disagree={te_d} "
            f"impure_tr_rows={tr_imp} -> "
            f"projected LB gain = {te_d/max(cls_n_test[cls],1)/3:.6f} "
            f"(but ~{tr_imp/(tr_d+tr_imp+1):.2%} will flip wrong way if extrapolated)")

    # Save artefacts
    out = {
        "score_diag": out_score,
        "primary_train_acc": float(train_acc),
        "submission_match_rate": float(agree),
        "n_pure_cells_100": int(len(pure)),
        "pure_cells_train_n": int(pure["tr_n"].sum()),
        "pure_cells_test_n": int(pure["te_n"].sum()),
        "pure_cells_train_disagree": int(overrides_tr),
        "pure_cells_test_disagree": int(overrides_te),
        "n_near_pure_999": int(len(near_pure)),
    }
    (ART / "purity_rules_diag.json").write_text(json.dumps(out, indent=2))
    df_cells.to_csv(ART / "purity_rules_per_cell.csv", index=False)
    log("=" * 64)
    log("saved scripts/artifacts/purity_rules_diag.json")
    log("saved scripts/artifacts/purity_rules_per_cell.csv (full cell table)")


if __name__ == "__main__":
    main()
