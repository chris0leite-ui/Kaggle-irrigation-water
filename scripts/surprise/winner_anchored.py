"""Override on 0.98140 winner anchor using helpers OUTSIDE the {raw, tier1b}
set that built the winner. Goal: identify rows where helpers not used in the
winner construction collectively disagree with the winner.

Mechanisms tried:
  W1:  k=2 unan of {lb3, 3way}              — minimal, complement helpers
  W2:  k=2 unan of {lb3, 3way}, H→M ONLY    — direction-masked W1
  W3:  k=3 unan of {lb3, 3way, t4}          — add T4 pseudo as 3rd voter
  W4:  k=3 unan of {v1, lb3, 3way}          — rollback: original anchor + 2 OTHERS
  W5:  k=4 of 5 majority of {lb3,3way,cb,recipe,t4}  — 5 helpers excl. winner-builders
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surprise.loaders import (  # noqa: E402
    all_helpers, load_oof_test, load_test_argmax, load_v1_anchor,
    load_winner_anchor, load_y, oof_argmax_at_bias,
)
from surprise.eval import emit_csv, evaluate, fmt_summary  # noqa: E402
from surprise.options import (  # noqa: E402
    k_majority_mask, k_unanimous_mask, restrict_to_HM,
)


def main():
    y = load_y()
    v1_oof_a, v1_test_a, _, _ = load_v1_anchor()
    winner_test_a = load_winner_anchor()
    helpers = all_helpers(y)

    # winner OOF analog: apply k=2 unanimous {raw, tier1b} on v1's OOF argmax
    raw_oof_a, tier1b_oof_a = helpers["raw"][2], helpers["tier1b"][2]
    raw_test_a, tier1b_test_a = helpers["raw"][3], helpers["tier1b"][3]
    m, vote = k_unanimous_mask([raw_oof_a, tier1b_oof_a], v1_oof_a)
    winner_oof_a = v1_oof_a.copy(); winner_oof_a[m] = vote[m]

    # Sanity: confirm this matches the saved winner CSV exactly
    test_m, test_v = k_unanimous_mask([raw_test_a, tier1b_test_a], v1_test_a)
    rebuilt = v1_test_a.copy(); rebuilt[test_m] = test_v[test_m]
    assert np.array_equal(rebuilt, winner_test_a), "winner test reconstruction mismatch!"
    print(f"winner reconstructs exactly. OOF macro at recipe-bias-tuned: ", end="")
    from sklearn.metrics import balanced_accuracy_score
    print(f"{balanced_accuracy_score(y, winner_oof_a):.5f}")

    # Helper sets for winner-anchored overrides
    lb3_oof, lb3_test = helpers["lb3"][2], helpers["lb3"][3]
    three_oof, three_test = helpers["3way"][2], helpers["3way"][3]
    t4_oof, t4_test = helpers["t4"][2], helpers["t4"][3]

    # Need cb + recipe helpers too
    o_cb, t_cb = load_oof_test("recipe_full_te_catboost")
    from common import tune_log_bias  # noqa
    b_cb, _ = tune_log_bias(o_cb, y, np.bincount(y, minlength=3) / len(y))
    cb_oof_a = oof_argmax_at_bias(o_cb, b_cb)
    cb_test_a = load_test_argmax("submission_recipe_full_te_catboost.csv")

    o_re, t_re = load_oof_test("recipe_full_te")
    b_re, _ = tune_log_bias(o_re, y, np.bincount(y, minlength=3) / len(y))
    re_oof_a = oof_argmax_at_bias(o_re, b_re)
    re_test_a = load_test_argmax("submission_recipe_full_te.csv")

    diags = {}

    def _build(name, helper_oof, helper_test, kind, k=None, hm_only=False):
        if kind == "unan":
            m_t, v_t = k_unanimous_mask(helper_test, winner_test_a)
            m_o, v_o = k_unanimous_mask(helper_oof, winner_oof_a)
        else:
            m_t, v_t = k_majority_mask(helper_test, winner_test_a, k=k)
            m_o, v_o = k_majority_mask(helper_oof, winner_oof_a, k=k)
        out_t = winner_test_a.copy(); out_t[m_t] = v_t[m_t]
        out_o = winner_oof_a.copy(); out_o[m_o] = v_o[m_o]
        if hm_only:
            out_t = restrict_to_HM(out_t, winner_test_a)
            out_o = restrict_to_HM(out_o, winner_oof_a)
        return out_t, out_o, int(m_t.sum()), int(m_o.sum())

    # W1: k=2 unan {lb3, 3way}
    out_t, out_o, n_t, n_o = _build("W1", [lb3_oof, three_oof], [lb3_test, three_test], "unan")
    diags["W1_winner_lb3_3way_k2unan"] = evaluate(out_t, out_o, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["W1_winner_lb3_3way_k2unan"]["overrides_test"] = n_t
    diags["W1_winner_lb3_3way_k2unan"]["overrides_oof"] = n_o
    if n_t > 0:
        emit_csv(out_t, "submission_W1_winner_lb3_3way_k2unan.csv")

    # W2: k=2 unan {lb3, 3way}, H->M only
    out_t, out_o, n_t, _ = _build("W2", [lb3_oof, three_oof], [lb3_test, three_test], "unan", hm_only=True)
    diags["W2_winner_lb3_3way_HMonly"] = evaluate(out_t, out_o, winner_test_a, winner_oof_a, winner_test_a, y)
    if (out_t != winner_test_a).any():
        emit_csv(out_t, "submission_W2_winner_lb3_3way_HMonly.csv")

    # W3: k=3 unan {lb3, 3way, t4}
    out_t, out_o, n_t, n_o = _build("W3", [lb3_oof, three_oof, t4_oof],
                                          [lb3_test, three_test, t4_test], "unan")
    diags["W3_winner_lb3_3way_t4_k3unan"] = evaluate(out_t, out_o, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["W3_winner_lb3_3way_t4_k3unan"]["overrides_test"] = n_t
    if n_t > 0:
        emit_csv(out_t, "submission_W3_winner_lb3_3way_t4_k3unan.csv")

    # W4: k=3 unan {v1, lb3, 3way} — rollback rows where v1 disagrees with k=2 winner override
    out_t, out_o, n_t, n_o = _build("W4", [v1_oof_a, lb3_oof, three_oof],
                                          [v1_test_a, lb3_test, three_test], "unan")
    diags["W4_winner_v1_lb3_3way_k3unan"] = evaluate(out_t, out_o, winner_test_a, winner_oof_a, winner_test_a, y)
    diags["W4_winner_v1_lb3_3way_k3unan"]["overrides_test"] = n_t
    if n_t > 0:
        emit_csv(out_t, "submission_W4_winner_v1_lb3_3way_k3unan.csv")

    # W5: k=4 of 5 majority over {lb3, 3way, cb, recipe, t4} (no winner-builders)
    h5_oof = [lb3_oof, three_oof, cb_oof_a, re_oof_a, t4_oof]
    h5_test = [lb3_test, three_test, cb_test_a, re_test_a, t4_test]
    out_t, out_o, n_t, _ = _build("W5", h5_oof, h5_test, "maj", k=4)
    diags["W5_winner_5nonbuilders_k4of5"] = evaluate(out_t, out_o, winner_test_a, winner_oof_a, winner_test_a, y)
    if (out_t != winner_test_a).any():
        emit_csv(out_t, "submission_W5_winner_5nonbuilders_k4of5.csv")

    Path("scripts/artifacts/surprise_winner_anchored_results.json").write_text(
        json.dumps(diags, indent=2, default=str))

    for k, d in diags.items():
        print(fmt_summary(k, d))
        print()


if __name__ == "__main__":
    main()
