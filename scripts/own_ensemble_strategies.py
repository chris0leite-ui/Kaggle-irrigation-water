"""5 ensemble strategies over 6 LB-validated own submissions.

This is the lever that's truly unlike every prior null: we ensemble at
the SUBMISSION level (treating each LB-validated submission as opaque),
not at the COMPONENT level. Public-CSV blenders use this exact
mechanism on others' submissions; we apply it to our own.

Strategies:
  S1 equal-log:       log_blend with uniform weights
  S2 lb-weighted:     log_blend with weights ∝ softmax((LB - 0.97) * τ)
  S3 hard-vote:       per-row majority of argmax, tie-break to highest LB
  S4 soft-vote:       arithmetic mean of probs (then renormalize)
  S5 greedy-forward:  start from primary, add candidates whose log-blend
                      at α improves fixed-bias OOF (anchor = primary)

For each strategy, report: standalone OOF tuned bal_acc, errs vs primary,
Jaccard, per-class recall delta vs primary, then α-sweep onto primary.
Emit submission if Δ ≥ +5e-4 OOF AND per-class recall guardrail PASSES.
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
from common import log_blend  # noqa: E402
from own_ensemble_helpers import LB_SCORES, reconstruct_lb_validated_set  # noqa: E402
from tier1b_helpers import BIAS, iso_cal, load_y, normed  # noqa: E402


ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
EPS = 1e-12


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def predict(p):
    return (np.log(np.clip(p, EPS, 1.0)) + BIAS).argmax(1)


def bal(p, y):
    return balanced_accuracy_score(y, predict(p))


def errs(p, y):
    return int((y != predict(p)).sum())


def pcr(p, y):
    pp = predict(p)
    return [float((pp[y == c] == c).mean()) for c in range(3)]


def jaccard(p, q, y):
    err_p = predict(p) != y
    err_q = predict(q) != y
    inter = int((err_p & err_q).sum())
    union = int((err_p | err_q).sum())
    return inter / max(union, 1)


def s1_equal_log(oofs):
    w = np.full(len(oofs), 1.0 / len(oofs))
    return log_blend(oofs, w)


def s2_lb_weighted(oofs, lb_scores, tau=200.0):
    """Heavier weight on higher-LB submissions via softmax((lb - 0.97) * τ)."""
    z = np.array([(lb - 0.97) * tau for lb in lb_scores])
    w = np.exp(z - z.max())
    w = w / w.sum()
    return log_blend(oofs, w), w


def s3_hard_vote(oofs, lb_scores):
    """Per-row argmax majority; tie-break to highest-LB submission."""
    preds = np.stack([predict(o) for o in oofs], axis=1)  # (N, K)
    N = preds.shape[0]
    out = np.zeros(N, dtype=np.int64)
    # Sort sub indices by LB descending
    lb_order = np.argsort(-np.asarray(lb_scores))
    for i in range(N):
        row = preds[i]
        cnt = np.bincount(row, minlength=3)
        max_count = cnt.max()
        winners = np.where(cnt == max_count)[0]
        if len(winners) == 1:
            out[i] = winners[0]
        else:
            # Tie: pick the class predicted by the highest-LB sub among ties
            for j in lb_order:
                if row[j] in winners:
                    out[i] = row[j]
                    break
    # Convert to one-hot probs (deterministic, no log-bias retune)
    probs = np.zeros((N, 3), dtype=np.float32)
    probs[np.arange(N), out] = 1.0
    return probs


def s4_soft_vote(oofs):
    return normed(np.mean(np.stack(oofs, axis=0), axis=0))


def s5_greedy_forward(oofs, names, anchor, y, alphas):
    """Start from anchor; greedily add the candidate with best α at fixed bias."""
    base = anchor.copy()
    base_bal = bal(base, y)
    chosen = []
    remaining = set(range(len(oofs)))
    history = [{"step": 0, "base_bal": base_bal, "chosen": list(chosen)}]
    while remaining:
        best = None
        for i in remaining:
            for a in alphas:
                if a == 0.0:
                    continue
                blend = log_blend([base, oofs[i]], np.array([1 - a, a]))
                b = bal(blend, y)
                if best is None or b > best[0]:
                    best = (b, i, a)
        if best is None or best[0] <= base_bal + 1e-5:
            break
        b, i, a = best
        base = log_blend([base, oofs[i]], np.array([1 - a, a]))
        base_bal = b
        chosen.append((names[i], a))
        remaining.discard(i)
        history.append({"step": len(chosen), "base_bal": base_bal,
                        "chosen": [(n, float(aa)) for n, aa in chosen]})
    return base, chosen, history


def main():
    log("Loading y + reconstructing 6 LB-validated submissions")
    y = load_y()
    subs = reconstruct_lb_validated_set(y)
    names = list(subs.keys())
    K = len(names)
    oofs_o = [subs[n][0] for n in names]
    tests = [subs[n][1] for n in names]
    lb_scores = [LB_SCORES[n] for n in names]

    primary_o = subs["primary_lb098094"][0]
    primary_t = subs["primary_lb098094"][1]
    primary_bal = bal(primary_o, y)
    primary_errs = errs(primary_o, y)
    primary_pcr = pcr(primary_o, y)
    log(f"Anchor primary: OOF {primary_bal:.6f}, errs {primary_errs}, PCR {primary_pcr}")

    # OPTIONAL: iso-cal CatBoost so its prob scale aligns with recipe-XGB
    # family. Without this its recipe-bias OOF is 0.9772 (way below LB).
    cb_idx = names.index("catboost_lb097935")
    cb_iso_o, cb_iso_t = iso_cal(oofs_o[cb_idx], tests[cb_idx], y)
    log(f"CatBoost iso-cal'd: OOF {bal(cb_iso_o, y):.6f} (raw {bal(oofs_o[cb_idx], y):.6f})")
    oofs_o[cb_idx] = cb_iso_o
    tests[cb_idx] = cb_iso_t

    log("\n=== Strategies ===")
    candidates = {}

    # S1
    log("S1 equal log-blend (weight 1/6 each)")
    o = s1_equal_log(oofs_o)
    t = s1_equal_log(tests)
    candidates["S1_equal_log"] = (o, t, None)

    # S2
    for tau in (100.0, 200.0, 500.0, 1000.0):
        o, w = s2_lb_weighted(oofs_o, lb_scores, tau=tau)
        t, _ = s2_lb_weighted(tests, lb_scores, tau=tau)
        candidates[f"S2_lb_weighted_tau{int(tau)}"] = (o, t, dict(zip(names, [float(x) for x in w])))

    # S3
    log("S3 hard-vote with LB-tie-break")
    o = s3_hard_vote(oofs_o, lb_scores)
    t = s3_hard_vote(tests, lb_scores)
    candidates["S3_hard_vote"] = (o, t, None)

    # S4
    log("S4 soft-vote (arithmetic mean of probs)")
    o = s4_soft_vote(oofs_o)
    t = s4_soft_vote(tests)
    candidates["S4_soft_vote"] = (o, t, None)

    # S5: greedy forward starting FROM primary, candidates are the OTHER 5
    other_idx = [i for i, n in enumerate(names) if n != "primary_lb098094"]
    other_oofs = [oofs_o[i] for i in other_idx]
    other_names = [names[i] for i in other_idx]
    other_tests = [tests[i] for i in other_idx]
    log("S5 greedy forward from primary, alpha grid 0.05..0.50")
    alphas = [0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    o, chosen, history = s5_greedy_forward(other_oofs, other_names, primary_o, y, alphas)
    # Apply same chain on test
    t = primary_t.copy()
    for nm, a in chosen:
        i = other_names.index(nm)
        t = log_blend([t, other_tests[i]], np.array([1 - a, a]))
    log(f"  greedy chosen: {chosen}")
    candidates["S5_greedy_forward"] = (o, t, {"chosen": chosen, "history": history})

    # Score all
    log("\n=== Standalone OOF results @ recipe bias ===")
    print(f"  {'name':<28} {'bal_acc':>9} {'Δ':>9} {'errs':>6} {'Δerr':>6} {'Jacc':>6} {'PCR_L':>7} {'PCR_M':>7} {'PCR_H':>7}")
    print(f"  {'-'*28} {'-'*9} {'-'*9} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7}")
    print(f"  {'PRIMARY (anchor)':<28} {primary_bal:.6f} {0:+.5f} {primary_errs:>6} {0:>+6} {1.0000:>6.4f} "
          f"{primary_pcr[0]:.4f} {primary_pcr[1]:.4f} {primary_pcr[2]:.4f}")
    summary = {"primary": {"bal": primary_bal, "errs": primary_errs, "pcr": primary_pcr}}
    for name, (o, t, meta) in candidates.items():
        b = bal(o, y)
        e = errs(o, y)
        j = jaccard(o, primary_o, y)
        c = pcr(o, y)
        delta = b - primary_bal
        derr = e - primary_errs
        print(f"  {name:<28} {b:.6f} {delta:+.5f} {e:>6} {derr:>+6} {j:>6.4f} "
              f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}")
        summary[name] = {"bal": b, "errs": e, "delta": delta, "derr": derr,
                         "jaccard_vs_primary": j, "pcr": c, "pcr_delta": [c[k] - primary_pcr[k] for k in range(3)]}
        if meta is not None:
            summary[name]["meta"] = meta

    # Save standalone OOFs/tests for cross-branch reuse
    for name, (o, t, _) in candidates.items():
        np.save(ART / f"oof_own_{name}.npy", o.astype(np.float32))
        np.save(ART / f"test_own_{name}.npy", t.astype(np.float32))

    (ART / "own_ensemble_strategies_results.json").write_text(json.dumps(summary, indent=2))
    log("Saved per-strategy OOF/test arrays + summary JSON")


if __name__ == "__main__":
    main()
