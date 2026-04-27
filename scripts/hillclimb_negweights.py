"""Experiment D: hill-climb ensemble with negative weights (Matt-OP-style).

Mirrors Matt-OP/hillclimbers `climb_hill()` but specialised to our problem:
  - target metric = balanced_accuracy_score at fixed bias [1.43, 1.47, 3.40]
  - blend space = ARITHMETIC sum of per-class probs (NOT log-blend, so
    negative weights are well-defined)
  - precision = 0.005 (Matt-OP default 0.01-0.001; 0.005 balances coverage
    vs wall time on a 70-component pool with 4-direction Δ-scan per step)
  - includes LB-best 4-stack as the seed component (initial weight 1.0)

Algorithm:
  1. Pool = tier1b_helpers.load_pool() + LB-best 4-stack as anchor seed.
  2. weights = zeros; weights[anchor] = 1.0.
  3. blend = Σ w_i * oof_i; metric = bal_acc((log(blend) + bias).argmax(1)).
  4. For each step: scan all components × Δ ∈ {-precision, +precision,
     -10*precision, +10*precision}; pick the (i, Δ) that maximises the
     metric; if no improvement found, stop.
  5. Save final OOF/test/weights JSON; run blend gate.

Output:
  oof_hillclimb_negweights.npy / test_hillclimb_negweights.npy
  hillclimb_negweights_results.json (per-step weights, metric trajectory,
  blend gate vs LB-best 4-stack and 3-stack)

NOT an LB submission emitter — strictly diagnostic OOF + gate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from tier1b_helpers import (  # noqa: E402
    BIAS, build_lbbest_stack, iso_cal, load_pool, load_y, normed,
)
from common import log_blend  # noqa: E402

ART = Path("scripts/artifacts")
EPS = 1e-9
PRECISION = 0.005
DELTAS = np.array([-10 * PRECISION, -PRECISION, PRECISION, 10 * PRECISION])
MAX_STEPS = 400  # safety cap; usually converges in <100


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def normed_clip(p):
    """Clip negatives to 0 + renormalize rows. Ensures log(blend) is valid."""
    p = np.clip(p, EPS, None)
    return p / p.sum(axis=1, keepdims=True)


def metric(blend, y, bias):
    """bal_acc at fixed bias on log(blend)."""
    pred = (np.log(np.clip(blend, EPS, 1.0)) + bias).argmax(1)
    return balanced_accuracy_score(y, pred)


def main():
    log("loading components")
    y = load_y()
    pool = load_pool()
    log(f"  pool size = {len(pool)}")

    # Build LB-best 4-stack as the anchor seed component.
    lb3_oof, lb3_test = build_lbbest_stack(y)  # the LB-best 3-stack
    # 4-stack = 3-stack ⊗ xgb_metastack__iso α=0.30
    mv1_oof = normed(np.load(ART / "oof_xgb_metastack.npy"))
    mv1_test = normed(np.load(ART / "test_xgb_metastack.npy"))
    mv1_oof_iso, mv1_test_iso = iso_cal(mv1_oof, mv1_test, y)
    lb4_oof = log_blend([lb3_oof, mv1_oof_iso], np.array([0.7, 0.3]))
    lb4_test = log_blend([lb3_test, mv1_test_iso], np.array([0.7, 0.3]))
    log(f"  LB-best 4-stack OOF bal@bias = {metric(lb4_oof, y, BIAS):.5f}")

    # Component matrix: anchor first, then sorted pool.
    names = ["lb_best_4stack"] + sorted(pool.keys())
    oofs = [lb4_oof] + [pool[n][0] for n in names[1:]]
    tests = [lb4_test] + [pool[n][1] for n in names[1:]]
    K = len(oofs)
    log(f"  total components = {K} (anchor + {K-1} pool)")

    # Stack into arrays for fast vector ops.
    O = np.stack(oofs, axis=0)  # (K, N, 3)
    T = np.stack(tests, axis=0)  # (K, M, 3)

    weights = np.zeros(K, dtype=np.float64)
    weights[0] = 1.0  # anchor at full weight

    # Resume from checkpoint if present (rehydrate-resilient).
    state_path = ART / "hillclimb_state.npz"
    history: list[dict] = []
    start_step = 1
    if state_path.exists():
        st = np.load(state_path, allow_pickle=True)
        if int(st["K"]) == K and list(st["names"]) == names:
            weights = st["weights"].astype(np.float64)
            history = list(st["history"])
            start_step = int(st["next_step"])
            log(f"  resume: loaded checkpoint @ step {start_step - 1}, "
                f"active={int((weights != 0).sum())}")
        else:
            log(f"  checkpoint dim mismatch (K={int(st['K'])} vs {K}); ignoring")

    # Hill climb loop.
    blend_oof = (weights[:, None, None] * O).sum(0)
    blend_oof = normed_clip(blend_oof)
    best = metric(blend_oof, y, BIAS)
    log(f"start bal_acc = {best:.5f}  (active={int((weights != 0).sum())})")
    if not history:
        history = [{"step": 0, "best": best,
                    "weights_active": int((weights != 0).sum())}]

    t0 = time.time()
    for step in range(start_step, MAX_STEPS + 1):
        # Scan all (i, Δ) candidates; pick the best improvement.
        best_gain = 0.0
        best_pick = None
        # Vectorise across Δ: for each i, compute ΔO contribution to blend.
        for i in range(K):
            for d in DELTAS:
                cand = blend_oof + d * O[i]
                cand = normed_clip(cand)
                m = metric(cand, y, BIAS)
                gain = m - best
                if gain > best_gain + 1e-7:
                    best_gain = gain
                    best_pick = (i, d, m, cand)
        if best_pick is None:
            log(f"step {step}: no improvement, stopping (best={best:.5f})")
            break
        i, d, m, blend_oof = best_pick
        weights[i] += d
        best = m
        history.append({
            "step": step, "i": int(i), "name": names[i], "delta": float(d),
            "weight": float(weights[i]), "best": float(best),
            "weights_active": int((weights != 0).sum()),
        })
        if step % 10 == 0 or step <= 5:
            log(f"step {step}: +Δ={d:+.4f} on '{names[i]}' (w={weights[i]:+.4f})  "
                f"bal={best:.5f}  active={(weights != 0).sum()}/{K}  "
                f"elapsed={time.time()-t0:.0f}s")
        # Checkpoint EVERY step (rehydrate resilience — container kills as fast
        # as ~5 min idle).
        np.savez(state_path,
                 K=np.array(K), names=np.array(names),
                 weights=weights, history=np.array(history, dtype=object),
                 next_step=np.array(step + 1))
    log(f"hill climb done: {step} steps, final bal = {best:.5f}, "
        f"elapsed = {time.time()-t0:.0f}s")

    # Build final test blend with the same weights.
    blend_test = (weights[:, None, None] * T).sum(0)
    blend_test = normed_clip(blend_test)

    # Diagnostics: top-10 active weights.
    order = np.argsort(np.abs(weights))[::-1]
    log("top 10 components by |weight|:")
    for j in order[:10]:
        if weights[j] != 0:
            log(f"  {names[j]:40s}  w = {weights[j]:+.4f}")

    # Blend gate vs LB-best 4-stack (anchor) and 3-stack.
    bal_anchor = metric(lb4_oof, y, BIAS)
    bal_3stack = metric(lb3_oof, y, BIAS)
    bal_final = best
    delta_anchor = bal_final - bal_anchor
    delta_3stack = bal_final - bal_3stack
    log(f"=== blend gate ===")
    log(f"  bal vs LB-best 4-stack: {delta_anchor:+.5f}  (gate ≥ +2e-4 = {'PASS' if delta_anchor >= 2e-4 else 'FAIL'})")
    log(f"  bal vs LB-best 3-stack: {delta_3stack:+.5f}")

    # Per-class recall guardrail vs anchor.
    pred_anchor = (np.log(np.clip(lb4_oof, EPS, 1.0)) + BIAS).argmax(1)
    pred_final = (np.log(np.clip(blend_oof, EPS, 1.0)) + BIAS).argmax(1)
    pcr_anchor = np.array([(pred_anchor[y == k] == k).mean() for k in range(3)])
    pcr_final = np.array([(pred_final[y == k] == k).mean() for k in range(3)])
    pcr_delta = pcr_final - pcr_anchor
    log(f"  per-class recall delta: L={pcr_delta[0]:+.5f} M={pcr_delta[1]:+.5f} H={pcr_delta[2]:+.5f}")
    pcr_pass = bool((pcr_delta >= -5e-4).all())
    log(f"  per-class guardrail (≥ -5e-4 each): {'PASS' if pcr_pass else 'FAIL'}")

    np.save(ART / "oof_hillclimb_negweights.npy", blend_oof.astype(np.float32))
    np.save(ART / "test_hillclimb_negweights.npy", blend_test.astype(np.float32))
    out = dict(
        config=dict(precision=PRECISION, max_steps=MAX_STEPS,
                    deltas=DELTAS.tolist(), bias=BIAS.tolist()),
        n_components=int(K),
        n_active=int((weights != 0).sum()),
        weights={names[i]: float(weights[i]) for i in range(K) if weights[i] != 0},
        bal_anchor=float(bal_anchor),
        bal_3stack=float(bal_3stack),
        bal_final=float(bal_final),
        delta_anchor=float(delta_anchor),
        delta_3stack=float(delta_3stack),
        pcr_anchor=pcr_anchor.tolist(),
        pcr_final=pcr_final.tolist(),
        pcr_delta=pcr_delta.tolist(),
        pcr_pass=pcr_pass,
        gate_pass=bool(delta_anchor >= 2e-4 and pcr_pass),
        history_tail=history[-30:],
        history_steps=len(history),
    )
    with open(ART / "hillclimb_negweights_results.json", "w") as f:
        json.dump(out, f, indent=2)
    log(f"wrote oof_hillclimb_negweights.npy + test + results.json")


if __name__ == "__main__":
    main()
