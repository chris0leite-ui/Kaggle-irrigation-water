"""L1 — diagnostic: do focal-loss OOFs carry signal orthogonal to the
14-bank-majority filter that 4b's selective override uses?

Question: 4b's mechanism flips B (LB 0.98140) on 108 rows where
  (a) bagged_v1' disagrees with B
  (b) {raw, tier1b} unanimously agree with bagged_v1'
  (c) 14-bank-majority confirms bagged_v1'
4b's filter does NOT include any focal-loss model. The 4 focal OOFs
(g2h3, g2_aH1, g2_invfreq, effnum) are NULL as standalone blends but
have not been tested as a consensus axis under 4b.

Probes:
  P1  Per-flip-row vote: on 4b's 108 flip rows, do the 4 focal models
      vote with B (the old class) or with B' (the new flip class)?
  P2  Focal-only majority on 108 flip rows: does it agree with the
      14-bank majority? If always agree → no orthogonal signal.
  P3  Extended bank (14 + 4 focal = 18) recompute on test: how many
      of 4b's 108 candidate flips are revoked / how many new flips
      added by the wider bank?
  P4  Global agreement (OOF): Jaccard of focal-only majority vs
      14-bank majority on training data, by class.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
SUB = Path("submissions")
CLS = {"Low": 0, "Medium": 1, "High": 2}

BANK_NAMES = [
    "sklearn_rf_meta_natural", "sklearn_rf_meta_natural_a1lgbm",
    "sklearn_rf_meta_natural_r10_with_tier1b", "rawashishsin_2600",
    "tier1b_greedy_meta", "recipe_full_te", "recipe_pseudolabel",
    "recipe_pseudolabel_seed7labeler", "realmlp", "xgb_nonrule",
    "xgb_metastack", "recipe_full_te_catboost_natural",
    "recipe_full_te_catboost", "lgbm_meta_natural",
]
FOCAL_NAMES = ["recipe_focal_g2h3", "recipe_focal_g2_aH1",
               "recipe_focal_g2_invfreq", "recipe_focal_effnum"]


def _norm(p, eps=1e-9):
    return p / np.clip(p.sum(axis=1, keepdims=True), eps, None)


def _argmax_stack(names, side):
    return np.stack([
        _norm(np.load(ART / f"{side}_{n}.npy").astype(np.float32)).argmax(1)
        for n in names
    ], axis=0)


def _majority(votes):
    n = votes.shape[1]
    out = np.empty(n, dtype=np.int8)
    for r in range(n):
        out[r] = np.bincount(votes[:, r], minlength=3).argmax()
    return out


def _csv_argmax(name):
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map(CLS).to_numpy(dtype=np.int8)


def main():
    b  = _csv_argmax("submission_2other_raw_tier1b_k2")
    bp = _csv_argmax("submission_idea4_foldbag_v1_b_mech")
    bank_maj = np.load(ART / "stability_test_majority.npy")
    flip_mask = (b != bp) & (bank_maj == bp)
    print(f"4b flip mask: {int(flip_mask.sum())} rows "
          f"(should be 108)")

    # ----- P1: focal votes on flip rows -----
    focal_test = _argmax_stack(FOCAL_NAMES, "test")  # (4, 270k)
    p1_per_model = {}
    for i, n in enumerate(FOCAL_NAMES):
        v = focal_test[i, flip_mask]
        b_v = b[flip_mask]; bp_v = bp[flip_mask]
        p1_per_model[n] = {
            "votes_with_B": int((v == b_v).sum()),
            "votes_with_Bprime": int((v == bp_v).sum()),
            "votes_other": int(((v != b_v) & (v != bp_v)).sum()),
        }
    print("\nP1 focal votes on 4b flip rows:")
    for n, d in p1_per_model.items():
        print(f"  {n:30s}  with_B={d['votes_with_B']:4d}  "
              f"with_Bprime={d['votes_with_Bprime']:4d}  "
              f"other={d['votes_other']:3d}")

    # ----- P2: focal-only majority on flip rows -----
    focal_maj = _majority(focal_test)
    p2 = {
        "agree_with_bank_maj_on_flips":
            int((focal_maj[flip_mask] == bp[flip_mask]).sum()),
        "agree_with_B_on_flips":
            int((focal_maj[flip_mask] == b[flip_mask]).sum()),
        "neither_on_flips": int(((focal_maj[flip_mask] != bp[flip_mask]) &
                                  (focal_maj[flip_mask] != b[flip_mask])).sum()),
    }
    print(f"\nP2 focal-only-majority on 108 flip rows: "
          f"agree_w_bank_maj={p2['agree_with_bank_maj_on_flips']}  "
          f"agree_w_B={p2['agree_with_B_on_flips']}  "
          f"neither={p2['neither_on_flips']}")

    # ----- P3: extended-bank (18) majority -----
    bank_test = _argmax_stack(BANK_NAMES, "test")  # (14, 270k)
    ext = np.concatenate([bank_test, focal_test], axis=0)  # (18, 270k)
    ext_maj = _majority(ext)
    new_flip_mask = (b != bp) & (ext_maj == bp)
    p3 = {
        "ext_maj_diff_from_bank_maj": int((ext_maj != bank_maj).sum()),
        "n_flips_under_ext_bank": int(new_flip_mask.sum()),
        "revoked_flips":
            int((flip_mask & ~new_flip_mask).sum()),
        "added_flips":
            int((~flip_mask & new_flip_mask & (b != bp)).sum()),
    }
    print(f"\nP3 ext-bank (18) majority:")
    print(f"  ext_maj differs from 14-bank maj on "
          f"{p3['ext_maj_diff_from_bank_maj']} test rows globally")
    print(f"  flips under ext bank: {p3['n_flips_under_ext_bank']} "
          f"(was 108)")
    print(f"  revoked: {p3['revoked_flips']}, "
          f"added: {p3['added_flips']}")

    # ----- P4: global focal-vs-bank agreement on OOF -----
    bank_oof = _argmax_stack(BANK_NAMES, "oof")
    focal_oof = _argmax_stack(FOCAL_NAMES, "oof")
    bm = _majority(bank_oof); fm = _majority(focal_oof)
    y = pd.read_csv("data/train.csv", usecols=["Irrigation_Need"])[
        "Irrigation_Need"].map(CLS).to_numpy()
    p4 = {
        "agree_rate": float((bm == fm).mean()),
        "disagree_rows": int((bm != fm).sum()),
    }
    by_class = {}
    for c, name in [(0, "L"), (1, "M"), (2, "H")]:
        m = y == c
        by_class[name] = {
            "n": int(m.sum()),
            "bank_acc": float((bm[m] == c).mean()),
            "focal_acc": float((fm[m] == c).mean()),
            "agree_rate": float((bm[m] == fm[m]).mean()),
        }
    print(f"\nP4 OOF global: bank-vs-focal agree_rate="
          f"{p4['agree_rate']:.5f}  disagree_rows={p4['disagree_rows']}")
    for k, v in by_class.items():
        print(f"  class {k}: n={v['n']}  bank_acc={v['bank_acc']:.5f}  "
              f"focal_acc={v['focal_acc']:.5f}  "
              f"agree={v['agree_rate']:.5f}")

    out = {
        "n_flips_4b": int(flip_mask.sum()),
        "P1_per_model": p1_per_model,
        "P2_focal_only_majority_on_flips": p2,
        "P3_extended_bank_majority": p3,
        "P4_oof_global_agreement": {**p4, "by_class": by_class},
    }
    (ART / "L1_focal_in_4b_diagnostic.json").write_text(
        json.dumps(out, indent=2))
    print(f"\n→ wrote scripts/artifacts/L1_focal_in_4b_diagnostic.json")


if __name__ == "__main__":
    main()
