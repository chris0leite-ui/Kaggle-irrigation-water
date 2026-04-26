"""Per-row gated mixture-of-experts blend (#1 from the speculative menu).

Experts (K=6):
  E0 = LB-best 4-stack          (the strong anchor)
  E1 = realmlp                  (NN orthogonality)
  E2 = xgb_nonrule_iso          (class-rebalanced logits, iso-cal'd)
  E3 = xgb_metastack_iso        (iso-cal'd 63-component meta)
  E4 = leaf_ote_meta_v2         (record-low Jaccard 0.49 — tree-space)
  E5 = xgb_dist_digits          (digit-extraction signal)

Gate is a linear classifier over ~26 low-dim features (4 dist + 4 abs +
agreement scores + per-expert max_prob + per-expert argmax onehot + dgp_score
+ rule_pred + bias). L2 = 1e-3. Per-fold gate fit on tr_idx, applied to
va_idx; test gets full-OOF-fitted gate.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from moe_helpers import fit_gate, forward_blend, gate_features  # noqa: E402
from tier1b_helpers import build_lbbest_stack, iso_cal, load_y, normed, BIAS  # noqa: E402

ART = Path("scripts/artifacts")
DATA = Path("data")
SMOKE = bool(int(__import__("os").environ.get("SMOKE", "0")))
SEED = 42
N_FOLDS = 5


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def L(name):
    return (normed(np.load(ART / f"oof_{name}.npy").astype(np.float32)),
            normed(np.load(ART / f"test_{name}.npy").astype(np.float32)))


def build_lbbest_4stack(y, lb3_o, lb3_t):
    """Reconstruct the LB-best 4-stack used as the meta_iso α=0.30 add."""
    meta_o, meta_t = L("xgb_metastack")
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    w = np.array([0.70, 0.30])
    return log_blend([lb3_o, meta_iso_o], w), log_blend([lb3_t, meta_iso_t], w)


def main():
    log("Loading y + raw frames")
    y = load_y()
    train_df = pd.read_csv(DATA / "train.csv")
    test_df = pd.read_csv(DATA / "test.csv")

    if SMOKE:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(train_df), 20000, replace=False)
        sm_idx = np.sort(idx)
        sm_test_idx = np.arange(min(10000, len(test_df)))
    else:
        sm_idx = None

    log("Loading & building experts")
    lb3_o, lb3_t = build_lbbest_stack(y)
    e0_o, e0_t = build_lbbest_4stack(y, lb3_o, lb3_t)
    realmlp_o, realmlp_t = L("realmlp")
    nr_raw_o, nr_raw_t = L("xgb_nonrule")
    nr_o, nr_t = iso_cal(nr_raw_o, nr_raw_t, y)
    meta_raw_o, meta_raw_t = L("xgb_metastack")
    meta_o, meta_t = iso_cal(meta_raw_o, meta_raw_t, y)
    leaf_o, leaf_t = L("leaf_ote_meta_v2")
    dig_o, dig_t = L("xgb_dist_digits")

    expert_oof_full = [e0_o, realmlp_o, nr_o, meta_o, leaf_o, dig_o]
    expert_test_full = [e0_t, realmlp_t, nr_t, meta_t, leaf_t, dig_t]
    K = len(expert_oof_full)
    log(f"Experts loaded K={K}")

    if SMOKE:
        train_df_use = train_df.iloc[sm_idx].reset_index(drop=True)
        test_df_use = test_df.iloc[sm_test_idx].reset_index(drop=True)
        y_use = y[sm_idx]
        expert_oof_use = [p[sm_idx] for p in expert_oof_full]
        expert_test_use = [p[sm_test_idx] for p in expert_test_full]
    else:
        train_df_use, test_df_use = train_df, test_df
        y_use = y
        expert_oof_use = expert_oof_full
        expert_test_use = expert_test_full

    log("Building gate features")
    Xtr, Xte = gate_features(train_df_use, test_df_use,
                             expert_oof_use, expert_test_use)
    log(f"Gate feature dim={Xtr.shape[1]}, train rows={Xtr.shape[0]}")

    # Reshape experts to (K, N, 3) for gather-style einsum
    expert_oof_arr = np.stack(expert_oof_use, axis=0)
    expert_test_arr = np.stack(expert_test_use, axis=0)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_tr = Xtr.shape[0]
    oof_blend = np.zeros((n_tr, 3), dtype=np.float32)
    gate_records = []
    test_blend_acc = np.zeros((Xte.shape[0], 3), dtype=np.float32)

    n_iter = 60 if SMOKE else 200
    l2 = 1e-3 if not SMOKE else 1e-2
    for fi, (tr, va) in enumerate(skf.split(np.zeros(n_tr), y_use)):
        t0 = time.time()
        Xt = Xtr[tr]
        et = expert_oof_arr[:, tr, :]
        W = fit_gate(Xt, et, y_use[tr], K=K, l2=l2, maxiter=n_iter, seed=SEED + fi)
        # OOF on val
        wv, Pv = forward_blend(W, Xtr[va], expert_oof_arr[:, va, :])
        oof_blend[va] = Pv
        # Test side
        wte, Pte = forward_blend(W, Xte, expert_test_arr)
        test_blend_acc += Pte / N_FOLDS
        gate_records.append({
            "fold": fi + 1,
            "W_norm": float(np.linalg.norm(W)),
            "mean_gate_weights": [float(x) for x in wv.mean(0)],
            "wall_s": round(time.time() - t0, 2),
        })
        log(f"  fold {fi+1}: W_norm={np.linalg.norm(W):.3f} "
            f"mean_w={wv.mean(0).round(3)} wall={time.time()-t0:.1f}s")

    log("Evaluating MoE blend at fixed bias")
    bal = balanced_accuracy_score(
        y_use, (np.log(np.clip(oof_blend, 1e-12, 1)) + BIAS).argmax(1))
    log(f"MoE OOF tuned bal_acc @ recipe-bias = {bal:.6f}")

    suffix = "_smoke" if SMOKE else ""
    np.save(ART / f"oof_moe_gated{suffix}.npy", oof_blend)
    np.save(ART / f"test_moe_gated{suffix}.npy", test_blend_acc)
    out = {
        "K": K,
        "smoke": SMOKE,
        "oof_tuned_bal_acc_at_recipe_bias": bal,
        "experts": ["lb_best_4stack", "realmlp", "xgb_nonrule_iso",
                    "xgb_metastack_iso", "leaf_ote_meta_v2", "xgb_dist_digits"],
        "gate_records": gate_records,
        "n_iter": n_iter,
        "l2": l2,
    }
    (ART / f"moe_gated_results{suffix}.json").write_text(json.dumps(out, indent=2))
    log(f"Saved oof_moe_gated{suffix}.npy + test + results JSON")


if __name__ == "__main__":
    main()
