"""L3 — OOF precision check on the 4-way consensus mechanism.

L2 found that using focal-majority as the disagreer in 4b's filter
proposes 1239 flips (vs 4b's 108) — 11× more candidates. Project
+0.0019 macro at 95% precision → LB ~0.9834.

Critical question: does precision degrade at 1239 flips vs 108?
Test on OOF: where {raw, tier1b, bank_maj, focal_maj} all agree on
class C != argmax_anchor, what is the empirical precision (rate at
which C == y_true)?

If 4-way-consensus precision on OOF ≈ 95-96%, the mechanism
transfers and the LB candidate is high-EV.

If precision is materially lower (e.g., <90%), the mechanism does
NOT transfer at scale; the 108-flip case may have been precision-
selected by bagged_v1's disagreement structure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ART = Path("scripts/artifacts")
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


def _argmax_oof(name):
    return _norm(np.load(ART / f"oof_{name}.npy").astype(np.float32)).argmax(1)


def _majority(votes):
    n = votes.shape[1]
    out = np.empty(n, dtype=np.int8)
    for r in range(n):
        out[r] = np.bincount(votes[:, r], minlength=3).argmax()
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


def main():
    print("=== L3: OOF precision of {raw, t1b, bank_maj, focal_maj} consensus ===\n")

    y = pd.read_csv("data/train.csv", usecols=["Irrigation_Need"])[
        "Irrigation_Need"].map(CLS).to_numpy().astype(np.int8)

    raw = _argmax_oof("recipe_full_te")
    t1b = _argmax_oof("tier1b_greedy_meta")
    bank_oof = np.stack([_argmax_oof(n) for n in BANK_NAMES], axis=0)
    bank_maj = _majority(bank_oof)
    focal_oof = np.stack([_argmax_oof(n) for n in FOCAL_NAMES], axis=0)
    focal_maj = _majority(focal_oof)

    # B-on-OOF analog: take the LB-best 4-stack as the "anchor" since
    # B is downstream of it. We don't have a perfect B_oof, so we
    # measure consensus precision UNCONDITIONAL on B's argmax — i.e.,
    # on every row where 4 sources agree on a class C, what fraction
    # has y_true == C? This is the upper bound on precision; the
    # B-conditional version would be a subset.
    consensus_class = raw.copy()
    consensus_mask = (raw == t1b) & (raw == bank_maj) & (raw == focal_maj)
    print(f"4-way consensus rate (raw=t1b=bank_maj=focal_maj): "
          f"{int(consensus_mask.sum())}/{len(y)} "
          f"({100*consensus_mask.mean():.3f}%)")
    print(f"unconditional precision when 4-way agrees: "
          f"{(consensus_class[consensus_mask] == y[consensus_mask]).mean():.5f}")

    # Conditional: cases where consensus DISAGREES with raw's argmax
    # — but raw is in the consensus, so raw=consensus by definition.
    # The 4b-style flip is when consensus DISAGREES with B (the
    # anchor we're flipping). Without B_oof we approximate by asking:
    # where does focal_maj DIFFER from {LB-best 4-stack argmax}, but
    # 3 OTHERS (raw, t1b, bank_maj) AGREE with focal_maj?
    # Use lb3+rmlp+nr 4-stack argmax as anchor proxy.

    anchor_candidates = {}
    for anchor_name in ["recipe_full_te", "tier1b_greedy_meta",
                        "lgbm_meta_natural", "rawashishsin_2600"]:
        anchor = _argmax_oof(anchor_name)
        # 4b-style flip: anchor != focal_maj AND raw == t1b == bank_maj == focal_maj
        # (raw and t1b are part of the agreers)
        flip = (anchor != focal_maj) & (raw == focal_maj) \
            & (t1b == focal_maj) & (bank_maj == focal_maj)
        if flip.sum() == 0:
            continue
        prec = (focal_maj[flip] == y[flip]).mean()
        # Direction breakdown
        b = anchor; new = anchor.copy(); new[flip] = focal_maj[flip]
        dirs = _direction_breakdown(b, new)
        # Per-direction precision
        dir_prec = {}
        for fr_to, _n in dirs.items():
            fr_c = CLS[{"L": "Low", "M": "Medium", "H": "High"}[fr_to[0]]]
            to_c = CLS[{"L": "Low", "M": "Medium", "H": "High"}[fr_to[3]]]
            mm = flip & (anchor == fr_c) & (focal_maj == to_c)
            dp = (focal_maj[mm] == y[mm]).mean() if mm.sum() else float("nan")
            dir_prec[fr_to] = (int(mm.sum()), float(dp))
        anchor_candidates[anchor_name] = {
            "n_flips": int(flip.sum()),
            "overall_precision": float(prec),
            "directions": dirs,
            "per_direction_precision": dir_prec,
        }
        print(f"\nanchor={anchor_name}:  n_flips={int(flip.sum())}  "
              f"prec={prec:.5f}  dirs={dirs}")
        for fr_to, (n, p) in dir_prec.items():
            print(f"    {fr_to}: n={n}  precision={p:.4f}")

    # Also report — what fraction of focal_maj's argmax agrees with y?
    fm_acc = (focal_maj == y).mean()
    print(f"\nfocal_maj overall OOF accuracy: {fm_acc:.5f}")
    bm_acc = (bank_maj == y).mean()
    print(f"bank_maj overall OOF accuracy: {bm_acc:.5f}")

    # Headline: when {raw, t1b, bank_maj, focal_maj} all agree on a
    # class C != raw_argmax (vacuous), what does this tell us? The
    # right question is: what's the precision of the JOINT condition
    # ON THE CONDITIONAL flip rows. Since we don't have B_oof, the
    # 4-anchor sweep above is the best proxy.

    out = {
        "consensus_rate_4way": float(consensus_mask.mean()),
        "consensus_unconditional_precision": float(
            (consensus_class[consensus_mask] == y[consensus_mask]).mean()),
        "focal_maj_overall_oof_acc": float(fm_acc),
        "bank_maj_overall_oof_acc": float(bm_acc),
        "by_anchor": anchor_candidates,
    }
    (ART / "L3_oof_precision_check.json").write_text(json.dumps(out, indent=2))
    print(f"\n→ wrote scripts/artifacts/L3_oof_precision_check.json")


if __name__ == "__main__":
    main()
