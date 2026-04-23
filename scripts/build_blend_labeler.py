"""Build a labeler from the stage-1 blend for stage-2 pseudo-label.

Computes the 50/50 log-blend of recipe_full_te × recipe_pseudolabel test
probs (identical to the LB-0.97998 submission's prob surface) and saves
it as a standalone test-probs file with an accompanying bias JSON in the
format recipe_pseudolabel.py consumes via env vars.

The blend uses recipe_full_te's fixed tuned bias [1.43, 1.47, 3.40] — the
same bias that drove LB 0.97998 — so the labeler sees rows argmaxed under
the LB-validated decision rule.

Outputs:
    scripts/artifacts/test_recipe_blend_stage1.npy
    scripts/artifacts/recipe_blend_stage1_results.json   (contains {"bias": [...]})
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ART = Path("scripts/artifacts")


def main():
    recipe_res = json.loads((ART / "recipe_full_te_results.json").read_text())
    bias = np.array(recipe_res["log_bias"])

    p_recipe = np.load(ART / "test_recipe_full_te.npy")
    p_pseudo = np.load(ART / "test_recipe_pseudolabel.npy")

    # 50/50 log-blend (same weights as the LB-0.97998 submission).
    logs = 0.5 * np.log(np.clip(p_recipe, 1e-9, 1.0)) \
        + 0.5 * np.log(np.clip(p_pseudo, 1e-9, 1.0))
    logs -= logs.max(1, keepdims=True)
    e = np.exp(logs)
    blend = e / e.sum(1, keepdims=True)
    blend = blend.astype(np.float32)

    out_probs = ART / "test_recipe_blend_stage1.npy"
    np.save(out_probs, blend)
    print(f"wrote {out_probs}  shape={blend.shape}")

    # Also compute the blend's OOF for logging, using the 50/50 log-blend
    # of saved OOFs. (Stage-2 pseudo-label doesn't need this but it's useful
    # context for the results summary.)
    oof_recipe = np.load(ART / "oof_recipe_full_te.npy")
    oof_pseudo = np.load(ART / "oof_recipe_pseudolabel.npy")
    oof_logs = 0.5 * np.log(np.clip(oof_recipe, 1e-9, 1.0)) \
        + 0.5 * np.log(np.clip(oof_pseudo, 1e-9, 1.0))
    oof_logs -= oof_logs.max(1, keepdims=True)
    oof_e = np.exp(oof_logs)
    oof_blend = (oof_e / oof_e.sum(1, keepdims=True)).astype(np.float32)

    import pandas as pd
    from sklearn.metrics import balanced_accuracy_score

    tr = pd.read_csv("data/train.csv")
    CLS2IDX = {"Low": 0, "Medium": 1, "High": 2}
    y = tr["Irrigation_Need"].map(CLS2IDX).values
    oof_fixed_bias = (np.log(np.clip(oof_blend, 1e-9, 1.0)) + bias).argmax(1)
    oof_ba = float(balanced_accuracy_score(y, oof_fixed_bias))
    print(f"blend OOF @ recipe bias = {oof_ba:.5f}")

    # Persist bias in the JSON format recipe_pseudolabel.py consumes.
    # LABELER_BIAS_JSON reader accepts either "log_bias" or "bias".
    summary = dict(
        bias=bias.tolist(),
        source="recipe_full_te × recipe_pseudolabel 50/50 log-blend",
        weights={"recipe_full_te": 0.5, "recipe_pseudolabel": 0.5},
        oof_tuned_bal_acc=oof_ba,
        lb_score=0.97998,
        oof_to_lb_gap=0.97998 - oof_ba,
    )
    out_json = ART / "recipe_blend_stage1_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
