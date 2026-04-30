"""T6 — compare T6 OOF blend vs 4b TRAIN OOF analog directly.

4b TRAIN OOF analog construction:
  Step 1: B_oof = v1_oof argmax with {raw_oof, tier1b_oof} unanimous override
  Step 2: 4b_oof = B_oof with bagged_v1' + {raw, tier1b} unanimous + bank-maj
                    selective override

We don't have an exact bagged_v1' OOF analog, but we can approximate by
using v1's argmax directly (no fold-bag) since the 4b mechanism's load-
bearing axis is the consensus/bank-majority.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mode

sys.path.insert(0, str(Path(__file__).parent))
from T2_conformal_helpers import load_bank  # noqa: E402
from T6_diversity_helpers import (  # noqa: E402
    load_y_train,
    macro_recall,
    normed,
    tune_log_bias_simple,
)
from T6_emit_candidate import PATH, log_blend  # noqa: E402

ART = Path("scripts/artifacts")


def main():
    print("=== T6 vs 4b TRAIN OOF analog comparison ===\n")
    y = load_y_train()

    # ---- Build T6 OOF blend ----
    oof_arrays = []
    for name, alpha in PATH:
        a = normed(np.load(ART / f"oof_{name}.npy").astype(np.float32))
        oof_arrays.append((a, alpha))
    t6_oof = log_blend(oof_arrays)
    bias, t6_score = tune_log_bias_simple(t6_oof, y)
    t6_argmax = (np.log(np.clip(t6_oof, 1e-9, None)) + bias).argmax(1).astype(np.int8)
    print(f"T6 blend OOF tuned macro: {t6_score:.6f}  bias={bias.round(3).tolist()}")
    print(f"T6 OOF argmax dist: {np.bincount(t6_argmax, minlength=3).tolist()}")

    # ---- Build 4b TRAIN OOF analog ----
    # B = v1's argmax with {raw, tier1b} k=2 unanimous override
    v1_oof = normed(np.load(ART / "oof_sklearn_rf_meta_natural.npy").astype(np.float32))
    raw_oof = normed(np.load(ART / "oof_rawashishsin_2600.npy").astype(np.float32))
    tier1b_oof = normed(np.load(ART / "oof_tier1b_greedy_meta.npy").astype(np.float32))

    # Tune log-bias on v1 then apply to predict
    v1_bias, v1_score = tune_log_bias_simple(v1_oof, y)
    v1_argmax = (np.log(np.clip(v1_oof, 1e-9, None)) + v1_bias).argmax(1).astype(np.int8)
    raw_bias, _ = tune_log_bias_simple(raw_oof, y)
    raw_argmax = (np.log(np.clip(raw_oof, 1e-9, None)) + raw_bias).argmax(1).astype(np.int8)
    t1_bias, _ = tune_log_bias_simple(tier1b_oof, y)
    t1_argmax = (np.log(np.clip(tier1b_oof, 1e-9, None)) + t1_bias).argmax(1).astype(np.int8)

    print(f"\nv1 OOF tuned macro: {v1_score:.6f}")
    print(f"v1 argmax dist: {np.bincount(v1_argmax, minlength=3).tolist()}")
    print(f"raw argmax dist: {np.bincount(raw_argmax, minlength=3).tolist()}")
    print(f"tier1b argmax dist: {np.bincount(t1_argmax, minlength=3).tolist()}")

    # B_oof = v1 with {raw, tier1b} unanimous override
    unanimous = (raw_argmax == t1_argmax)
    B_oof_argmax = v1_argmax.copy()
    override_mask_b = unanimous & (v1_argmax != raw_argmax)
    B_oof_argmax[override_mask_b] = raw_argmax[override_mask_b]
    print(f"\nB OOF (v1 + {{raw,tier1b}} unanimous): "
          f"{int(override_mask_b.sum())} overrides")
    print(f"B OOF macro: {macro_recall(y, B_oof_argmax):.6f}")
    print(f"B OOF argmax dist: {np.bincount(B_oof_argmax, minlength=3).tolist()}")

    # 14-bank majority on TRAIN OOF
    oof_bank = load_bank("oof")
    oof_argmax_per = oof_bank.argmax(axis=2)  # (14, 630000)
    bank_maj_oof = mode(oof_argmax_per, axis=0, keepdims=False).mode

    # 4b OOF analog: B with disagree(B, bank_maj) AND raw==tier1b==bank_maj
    disagree = B_oof_argmax != bank_maj_oof
    raw_t1_agree_with_maj = (raw_argmax == bank_maj_oof) & (t1_argmax == bank_maj_oof)
    fb_override_mask = disagree & raw_t1_agree_with_maj
    fb_oof_argmax = B_oof_argmax.copy()
    fb_oof_argmax[fb_override_mask] = bank_maj_oof[fb_override_mask]
    print(f"\n4b OOF analog (B + bank-maj override): "
          f"{int(fb_override_mask.sum())} overrides")
    print(f"4b OOF macro: {macro_recall(y, fb_oof_argmax):.6f}")
    print(f"4b OOF argmax dist: {np.bincount(fb_oof_argmax, minlength=3).tolist()}")

    # T6 vs 4b on TRAIN
    print("\n--- TRAIN OOF: T6 vs 4b argmax comparison ---")
    diff_t6_fb = int((t6_argmax != fb_oof_argmax).sum())
    print(f"T6 argmax differs from 4b OOF analog on: {diff_t6_fb} rows")

    dirs = {}
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            m = (fb_oof_argmax == fr) & (t6_argmax == to)
            if m.sum():
                dirs[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = int(m.sum())
    print(f"T6 directions vs 4b OOF: {dirs}")

    # Per-direction precision: where T6 differs from 4b, who is right?
    print("\n--- Precision by direction (T6 vs 4b on TRAIN OOF):  ---")
    print(f"{'direction':<10} {'n':<8} {'T6 wins':<10} {'4b wins':<10} {'tie':<8}")
    for fr in range(3):
        for to in range(3):
            if fr == to: continue
            m = (fb_oof_argmax == fr) & (t6_argmax == to)
            n = int(m.sum())
            if n == 0: continue
            t6_correct = ((t6_argmax == y) & m).sum()
            fb_correct = ((fb_oof_argmax == y) & m).sum()
            tie = n - t6_correct - fb_correct
            print(f"{['L','M','H'][fr]}->{['L','M','H'][to]:<6} {n:<8} "
                  f"{int(t6_correct):<10} {int(fb_correct):<10} {int(tie):<8}")

    # final T6 vs 4b TRAIN OOF macro
    diff_macro = t6_score - macro_recall(y, fb_oof_argmax)
    print(f"\nT6 OOF macro - 4b OOF macro = {diff_macro:+.6f}")


if __name__ == "__main__":
    main()
