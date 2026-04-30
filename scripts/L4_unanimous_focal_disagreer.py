"""L4 — tighten focal-disagreer to UNANIMOUS consensus and OOF-check precision.

L2 found focal-majority generates 1239 flips at 75.9% H->M precision
(L3) — far below the 90.9% break-even. Hypothesis tested here: at
strict unanimous consensus (all 4 focal AND all 14 bank components
vote for the new class C), precision recovers because unanimous
agreement ≈ what bagged_v1' was implicitly doing via
fold-bag-confidence selection.

Mechanism on TEST:
  flip B where:
    (a) B != C
    (b) all 4 focal_*.argmax == C  (focal-unanimous)
    (c) all 14 bank_*.argmax == C  (bank-unanimous)
    (d) raw == C, tier1b == C  (already implied by (c) since both in bank)

(d) is automatically satisfied if (c) holds, since raw=recipe_full_te
and tier1b=tier1b_greedy_meta are both in the 14-bank.

Sweep strictness levels:
  ALL14 + ALL4   — unanimous on both (strictest)
  ALL14 + 3of4   — bank unanimous, 3 of 4 focal
  13of14 + ALL4  — 1 bank dissent allowed, focal unanimous
  12of14 + ALL4  — 2 bank dissent, focal unanimous
  10of14 + 3of4  — relaxed sweet-spot guess

For each strictness level: count flips on TEST, then check the
analogous OOF precision (no anchor needed — just measure rate at
which the consensus class C == y_true on rows where the consensus
holds AND C != some plausible anchor).

Anchor for OOF check: B's analog. We use rawashishsin_2600 as the
anchor (closest in rank to B's role and was the L3 conservative
upper-bound for n_flips). Precision result is direction-comparable
across strictness levels.
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


def _argmax(name, side):
    return _norm(np.load(ART / f"{side}_{name}.npy").astype(np.float32)).argmax(1)


def _csv_argmax(name):
    s = pd.read_csv(SUB / f"{name}.csv")["Irrigation_Need"]
    return s.map(CLS).to_numpy(dtype=np.int8)


def _stack(names, side):
    return np.stack([_argmax(n, side) for n in names], axis=0)


def _consensus_class(stack, min_votes):
    """Return per-row class C where ≥min_votes of stack agree on C, else -1."""
    n = stack.shape[1]
    out = np.full(n, -1, dtype=np.int8)
    for c in range(3):
        votes = (stack == c).sum(axis=0)
        out = np.where((votes >= min_votes) & (out == -1), c, out)
    return out


def _direction_breakdown(b, new_pred):
    out = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            n = int(((b == fr) & (new_pred == to)).sum())
            if n > 0:
                out[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = n
    return out


def _eval_strictness(label, bank_min, focal_min, bank_test, focal_test,
                     bank_oof, focal_oof, b_test, b_oof_anchor, y):
    """For a given (bank_min, focal_min) strictness level, compute:
      - test-side: candidate flip count + direction breakdown
      - oof-side: precision per direction on the analogous flip set
    """
    # TEST: derive consensus class C
    bC_test = _consensus_class(bank_test, bank_min)
    fC_test = _consensus_class(focal_test, focal_min)
    consensus_test = (bC_test == fC_test) & (bC_test != -1)
    C_test = np.where(consensus_test, bC_test, -1)

    flip_test = (C_test != -1) & (C_test != b_test)
    new_pred = b_test.copy()
    new_pred[flip_test] = C_test[flip_test]
    test_dirs = _direction_breakdown(b_test, new_pred)

    # OOF: same consensus, anchor = b_oof_anchor (rawashishsin_2600 argmax)
    bC_oof = _consensus_class(bank_oof, bank_min)
    fC_oof = _consensus_class(focal_oof, focal_min)
    consensus_oof = (bC_oof == fC_oof) & (bC_oof != -1)
    C_oof = np.where(consensus_oof, bC_oof, -1)
    flip_oof = (C_oof != -1) & (C_oof != b_oof_anchor)

    overall_prec = float((C_oof[flip_oof] == y[flip_oof]).mean()) \
        if flip_oof.sum() else float("nan")

    # Per-direction OOF precision
    dir_prec = {}
    for fr in range(3):
        for to in range(3):
            if fr == to:
                continue
            mm = flip_oof & (b_oof_anchor == fr) & (C_oof == to)
            if mm.sum() < 5:
                continue
            p = (C_oof[mm] == y[mm]).mean()
            dir_prec[f"{['L','M','H'][fr]}->{['L','M','H'][to]}"] = (
                int(mm.sum()), float(p)
            )

    return {
        "label": label,
        "bank_min": bank_min, "focal_min": focal_min,
        "test_n_flips": int(flip_test.sum()),
        "test_directions": test_dirs,
        "oof_n_flips": int(flip_oof.sum()),
        "oof_overall_precision": overall_prec,
        "oof_per_direction_precision": dir_prec,
        "_flip_test_mask": flip_test,
        "_C_test": C_test,
    }


def main():
    print("=== L4: unanimous focal-disagreer w/ OOF-precision check ===\n")

    y = pd.read_csv("data/train.csv", usecols=["Irrigation_Need"])[
        "Irrigation_Need"].map(CLS).to_numpy().astype(np.int8)

    bank_test = _stack(BANK_NAMES, "test")        # (14, 270k)
    bank_oof  = _stack(BANK_NAMES, "oof")         # (14, 630k)
    focal_test = _stack(FOCAL_NAMES, "test")      # (4, 270k)
    focal_oof  = _stack(FOCAL_NAMES, "oof")       # (4, 630k)

    b_test = _csv_argmax("submission_2other_raw_tier1b_k2")  # B (LB 0.98140)
    b_4b   = _csv_argmax("submission_idea4b_selective_override")  # 4b (0.98150)
    # OOF anchor — closest to B's role
    b_oof_anchor = _argmax("rawashishsin_2600", "oof")

    grid = [
        ("ALL14_ALL4",   14, 4),
        ("ALL14_3of4",   14, 3),
        ("13of14_ALL4",  13, 4),
        ("13of14_3of4",  13, 3),
        ("12of14_ALL4",  12, 4),
        ("12of14_3of4",  12, 3),
        ("11of14_ALL4",  11, 4),
        ("10of14_3of4",  10, 3),
    ]

    rows = []
    for label, bm, fm in grid:
        r = _eval_strictness(label, bm, fm, bank_test, focal_test,
                             bank_oof, focal_oof, b_test, b_oof_anchor, y)
        rows.append(r)
        # Compare to 4b's flip mask
        flip_4b_mask = (b_test != b_4b)
        overlap = int((r["_flip_test_mask"] & flip_4b_mask).sum())
        new_vs_4b = int((r["_flip_test_mask"] & ~flip_4b_mask).sum())
        print(f"{label:18s}  test_flips={r['test_n_flips']:5d}  "
              f"dirs={r['test_directions']}")
        print(f"  vs 4b:  overlap={overlap:3d}  new={new_vs_4b:5d}")
        print(f"  OOF: n={r['oof_n_flips']:5d}  "
              f"overall_prec={r['oof_overall_precision']:.4f}")
        for fr_to, (n, p) in r["oof_per_direction_precision"].items():
            be = {"H->M": 0.909, "L->M": 0.612, "M->L": 0.387,
                  "M->H": 0.909, "L->H": 0.952, "H->L": 0.387}.get(fr_to, 0.5)
            mark = "✓" if p >= be else "✗"
            print(f"    {fr_to}: n={n:5d}  prec={p:.4f}  "
                  f"BE={be:.3f}  {mark}")
        print()

    out = []
    for r in rows:
        out.append({k: v for k, v in r.items()
                    if not k.startswith("_")})
    (ART / "L4_unanimous_focal_disagreer.json").write_text(
        json.dumps(out, indent=2))
    print(f"→ wrote scripts/artifacts/L4_unanimous_focal_disagreer.json")


if __name__ == "__main__":
    main()
