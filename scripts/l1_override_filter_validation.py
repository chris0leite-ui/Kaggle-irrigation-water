"""Train-side validation of three override filters on Layer-2 train disagreements.

Goal: rescue the 25th-saturation L1 override (LB -0.00032) by filtering the
36 test candidates with side information that distinguishes primary-wrong
disagreements from primary-right (NN-flip-detection) disagreements.

Layer-2 train disagreements (51 rows: 43 primary-wrong + 8 primary-right) are
the only train rows where we can ground-truth-validate "primary was right
to disagree with cell-majority", because Layer-1 cells are 100%-pure on
train by construction (all 46 train disagreements are wrong by definition).

Three filters validated:
  F1  hedge_agrees_with_cell_majority   (consensus discriminator)
  F2  primary_max_prob < threshold      (primary uncertainty)
  F3  primary 2nd-choice == cell_majority  (posterior shape)

Plus a tiny meta-discriminator (LR on 51 examples) using all 3 + extras.

Outputs:
  scripts/artifacts/l1_filter_validation_results.json
  Summary printed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend
from purity_rules_diag import compute_rule
from tier1b_helpers import (
    ART, BIAS, CLASSES, CLS2IDX, build_lbbest_stack, iso_cal, normed,
)

CLS = CLASSES
CLS2I = CLS2IDX
EPS = 1e-12

PURITY_LO = 0.999
PURITY_HI = 1.0  # exclude strict-100% (those are Layer-1)
MIN_TR_N = 50

SUB_CATS = ["Soil_Type", "Crop_Type", "Season",
            "Irrigation_Type", "Water_Source", "Region"]


def main():
    print("=== L1 override filter validation ===\n")
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    y = train["Irrigation_Need"].map(CLS2I).to_numpy().astype(np.int32)

    rt = compute_rule(train); re_ = compute_rule(test)
    cell_tr = rt["cell_id"]; cell_te = re_["cell_id"]

    # ---- Reconstruct primary OOF + test probs (LB-best 4-stack) ----
    s2_o, s2_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    meta_o_iso, meta_t_iso = iso_cal(meta_o, meta_t, y)
    primary_o = log_blend([s2_o, meta_o_iso], np.array([0.7, 0.3]))
    primary_t = log_blend([s2_t, meta_t_iso], np.array([0.7, 0.3]))

    # primary argmax with bias
    primary_logits_tr = np.log(np.clip(primary_o, EPS, 1)) + BIAS
    primary_logits_te = np.log(np.clip(primary_t, EPS, 1)) + BIAS
    primary_argmax_tr = primary_logits_tr.argmax(1)
    primary_argmax_te = primary_logits_te.argmax(1)
    # softmax-of-bias-adjusted-logits = primary's calibrated posterior
    p = np.exp(primary_logits_tr - primary_logits_tr.max(1, keepdims=True))
    primary_post_tr = p / p.sum(1, keepdims=True)
    p = np.exp(primary_logits_te - primary_logits_te.max(1, keepdims=True))
    primary_post_te = p / p.sum(1, keepdims=True)

    # ---- Reconstruct hedge OOF + test probs (3-way recipe + pseudo_s1 + pseudo_s7) ----
    r = (normed(np.load(ART / "oof_recipe_full_te.npy")),
         normed(np.load(ART / "test_recipe_full_te.npy")))
    s1 = (normed(np.load(ART / "oof_recipe_pseudolabel.npy")),
          normed(np.load(ART / "test_recipe_pseudolabel.npy")))
    s7 = (normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy")),
          normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy")))
    hedge_o = log_blend([r[0], s1[0], s7[0]], np.array([0.25, 0.35, 0.40]))
    hedge_t = log_blend([r[1], s1[1], s7[1]], np.array([0.25, 0.35, 0.40]))
    hedge_argmax_tr = (np.log(np.clip(hedge_o, EPS, 1)) + BIAS).argmax(1)
    hedge_argmax_te = (np.log(np.clip(hedge_t, EPS, 1)) + BIAS).argmax(1)

    # Sanity check primary OOF
    pri_oof_macro = (primary_argmax_tr == y).mean()
    print(f"Primary OOF raw acc:  {pri_oof_macro:.5f} "
          f"(reconstruction sanity check)")

    # ---- Per-cell train majority + cell purity ----
    cells = np.unique(cell_tr)
    cell_majority = {}; cell_purity = {}; cell_n = {}
    for cid in cells:
        m = cell_tr == cid
        cnt = np.bincount(y[m], minlength=3)
        cell_majority[int(cid)] = int(np.argmax(cnt))
        cell_n[int(cid)] = int(m.sum())
        cell_purity[int(cid)] = cnt.max() / m.sum()

    # ---- Layer-2 (99.9-99.99% pure) sub-cell rules ----
    print("\nBuilding Layer-2 sub-cell rules...")
    rules = []
    layer2_tr_mask = np.zeros(len(train), dtype=bool)
    layer2_te_mask = np.zeros(len(test), dtype=bool)
    layer2_tr_maj = np.full(len(train), -1, dtype=np.int8)
    layer2_te_maj = np.full(len(test), -1, dtype=np.int8)
    layer2_tr_subN = np.zeros(len(train), dtype=np.int32)
    layer2_te_subN = np.zeros(len(test), dtype=np.int32)
    for cid in cells:
        if cell_purity[int(cid)] >= 1.0: continue
        m_tr = cell_tr == cid; m_te = cell_te == cid
        if m_te.sum() == 0: continue
        maj = cell_majority[int(cid)]
        cell_y = y[m_tr]
        impure_in_cell = cell_y != maj
        for cat in SUB_CATS:
            cat_tr = train.loc[m_tr, cat].to_numpy()
            cat_te = test.loc[m_te, cat].to_numpy()
            for val in pd.unique(cat_tr):
                m_tr_val = cat_tr == val; m_te_val = cat_te == val
                tr_n = int(m_tr_val.sum())
                if tr_n < MIN_TR_N: continue
                n_imp = int(impure_in_cell[m_tr_val].sum())
                pur = (tr_n - n_imp) / tr_n
                if not (PURITY_LO <= pur < PURITY_HI): continue
                idx_tr_cell = np.where(m_tr)[0]
                idx_te_cell = np.where(m_te)[0]
                idx_tr_sub = idx_tr_cell[m_tr_val]
                idx_te_sub = idx_te_cell[m_te_val]
                layer2_tr_mask[idx_tr_sub] = True
                layer2_te_mask[idx_te_sub] = True
                layer2_tr_maj[idx_tr_sub] = maj
                layer2_te_maj[idx_te_sub] = maj
                layer2_tr_subN[idx_tr_sub] = tr_n
                layer2_te_subN[idx_te_sub] = tr_n
                rules.append((cid, cat, val, maj, tr_n, n_imp))
    print(f"  Layer-2 rules: {len(rules)}, train coverage {layer2_tr_mask.sum():,}, test coverage {layer2_te_mask.sum():,}")

    # ---- Layer-2 train disagreements: primary disagrees with cell-majority ----
    l2_disagree_tr = layer2_tr_mask & (primary_argmax_tr != layer2_tr_maj) & (layer2_tr_maj >= 0)
    n_l2_dis = int(l2_disagree_tr.sum())
    primary_was_wrong = l2_disagree_tr & (primary_argmax_tr != y)
    primary_was_right = l2_disagree_tr & (primary_argmax_tr == y)
    n_pw = int(primary_was_wrong.sum())
    n_pr = int(primary_was_right.sum())
    print(f"\nLayer-2 train disagreements: {n_l2_dis} (primary_wrong={n_pw}, primary_right={n_pr})")

    # ---- Build per-row features for the 51 disagreement rows ----
    idx = np.where(l2_disagree_tr)[0]
    df = pd.DataFrame({
        "row_idx": idx,
        "primary_pred": primary_argmax_tr[idx],
        "y_true": y[idx],
        "cell_majority": layer2_tr_maj[idx],
        "primary_was_wrong": primary_was_wrong[idx].astype(int),
        "sub_cell_N": layer2_tr_subN[idx],
        # Primary posterior (post-bias)
        "p_low": primary_post_tr[idx, 0],
        "p_med": primary_post_tr[idx, 1],
        "p_high": primary_post_tr[idx, 2],
        "primary_max_prob": primary_post_tr[idx].max(1),
        "primary_argmax_prob": primary_post_tr[idx, primary_argmax_tr[idx]],
        "primary_majority_prob": primary_post_tr[idx, layer2_tr_maj[idx]],
        "primary_2nd_choice": np.argsort(-primary_post_tr[idx], 1)[:, 1],
        # Hedge prediction
        "hedge_pred": hedge_argmax_tr[idx],
        "hedge_agrees_with_majority": (hedge_argmax_tr[idx] == layer2_tr_maj[idx]).astype(int),
    })
    df["margin"] = df["primary_argmax_prob"] - df["primary_majority_prob"]
    df["second_is_majority"] = (df["primary_2nd_choice"] == df["cell_majority"]).astype(int)
    df["primary_class"] = df["primary_pred"].map({0:"L",1:"M",2:"H"})
    df["maj_class"] = df["cell_majority"].map({0:"L",1:"M",2:"H"})
    df["truth"] = df["y_true"].map({0:"L",1:"M",2:"H"})

    print("\n=== Layer-2 disagreement table (truncated) ===")
    cols = ["primary_class","maj_class","truth","primary_max_prob","margin","second_is_majority","hedge_agrees_with_majority","sub_cell_N","primary_was_wrong"]
    print(df[cols].head(20).to_string(index=False))

    # ---- Filter analyses ----
    print("\n=== FILTER 1: hedge_agrees_with_majority ===")
    for v in [0, 1]:
        s = df[df["hedge_agrees_with_majority"] == v]
        if len(s) == 0: continue
        prec = s["primary_was_wrong"].mean()
        print(f"  hedge_agrees={v}: n={len(s):3d}, primary_wrong_rate={prec:.3f}")

    print("\n=== FILTER 2: primary_max_prob threshold ===")
    for tau in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        s = df[df["primary_max_prob"] < tau]
        if len(s) == 0:
            print(f"  max_prob<{tau}: n=0")
            continue
        prec = s["primary_was_wrong"].mean()
        print(f"  max_prob<{tau}: n={len(s):3d}, primary_wrong_rate={prec:.3f}")
    # also high tail
    for tau in [0.85, 0.90, 0.95]:
        s = df[df["primary_max_prob"] >= tau]
        prec = s["primary_was_wrong"].mean() if len(s)>0 else float("nan")
        print(f"  max_prob>={tau}: n={len(s):3d}, primary_wrong_rate={prec:.3f}")

    print("\n=== FILTER 3: second_is_majority ===")
    for v in [0, 1]:
        s = df[df["second_is_majority"] == v]
        if len(s) == 0: continue
        prec = s["primary_was_wrong"].mean()
        print(f"  second_is_majority={v}: n={len(s):3d}, primary_wrong_rate={prec:.3f}")

    print("\n=== Combined: hedge AND (second==majority OR low_max_prob) ===")
    for τ in [0.85, 0.90, 0.95]:
        s = df[(df["hedge_agrees_with_majority"] == 1) &
               ((df["second_is_majority"] == 1) | (df["primary_max_prob"] < τ))]
        if len(s) == 0: continue
        prec = s["primary_was_wrong"].mean()
        print(f"  hedge=AGREE & (2nd=maj | maxp<{τ}): n={len(s):3d}, prec={prec:.3f}")

    # ---- Tiny meta-discriminator (LR with class_weight='balanced' on 51 rows) ----
    print("\n=== META-DISCRIMINATOR (LR, leave-one-out CV) ===")
    feats = ["hedge_agrees_with_majority", "primary_max_prob", "margin",
             "second_is_majority", "sub_cell_N"]
    X = df[feats].to_numpy(dtype=float)
    Y = df["primary_was_wrong"].to_numpy()
    # LOO CV (51-fold) for AUC since N=51
    preds = np.zeros(len(X))
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool); mask[i] = False
        try:
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(X[mask], Y[mask])
            preds[i] = lr.predict_proba(X[i:i+1])[0, 1]
        except Exception:
            preds[i] = Y.mean()  # majority class fallback
    auc = roc_auc_score(Y, preds)
    print(f"  LOO AUC: {auc:.4f}  (N={len(X)})")
    # Calibration: bucketed precision
    df["meta_prob_loo"] = preds
    print("  Bucketed primary_wrong_rate by meta_prob:")
    for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        s = df[(df["meta_prob_loo"] >= lo) & (df["meta_prob_loo"] < hi)]
        if len(s) == 0: continue
        prec = s["primary_was_wrong"].mean()
        print(f"    P(primary_wrong)∈[{lo:.2f}, {hi:.2f}): n={len(s):3d}, actual_rate={prec:.3f}")

    # ---- Now fit on ALL 51 train, apply to 36 test candidates ----
    print("\n=== Apply filters to 36 test candidates ===")
    drop_te = np.load(ART / "drop_mask_test.npy").astype(bool)
    test_maj = np.load(ART / "test_cell_majority.npy").astype(np.int64)
    primary_pred_csv = pd.read_csv("submissions/submission_tier1b_greedy_meta_l1override.csv")
    sub_orig = pd.read_csv("submissions/submission_tier1b_greedy_meta.csv")
    primary_te_argmax = sub_orig["Irrigation_Need"].map(CLS2I).to_numpy()

    test_dis = drop_te & (test_maj != primary_te_argmax) & (test_maj >= 0)
    test_idx = np.where(test_dis)[0]
    n_te = len(test_idx)
    print(f"  Test candidates: {n_te}")

    # Build features for test candidates (need a sub_cell_N estimate per row)
    # For each test row, find the matching Layer-2 sub-cell (or Layer-1 cube cell)
    # and lookup its train sample size.
    # Layer-1 cube cells (cid=0, cid=8): use cell_n[cid]
    # Layer-1 sub-cell rules: train rows are in `drop_mask_train` (set by purity_subcells.py).
    # Don't have per-row sub_cell_N for layer-1 in scope; use layer-2 mask if available, else cell_n.
    test_subN = np.zeros(len(test), dtype=np.int32)
    # default: use cube cell's total N
    for i in range(len(test)):
        test_subN[i] = cell_n.get(int(cell_te[i]), 0)
    # overlay layer-2 sub-cell N where applicable
    for i in range(len(test)):
        if layer2_te_subN[i] > 0:
            test_subN[i] = layer2_te_subN[i]

    test_df = pd.DataFrame({
        "row_idx": test_idx,
        "primary_pred": primary_te_argmax[test_idx],
        "cell_majority": test_maj[test_idx],
        "p_low": primary_post_te[test_idx, 0],
        "p_med": primary_post_te[test_idx, 1],
        "p_high": primary_post_te[test_idx, 2],
        "primary_max_prob": primary_post_te[test_idx].max(1),
        "primary_argmax_prob": primary_post_te[test_idx, primary_te_argmax[test_idx]],
        "primary_majority_prob": primary_post_te[test_idx, test_maj[test_idx]],
        "primary_2nd_choice": np.argsort(-primary_post_te[test_idx], 1)[:, 1],
        "hedge_pred": hedge_argmax_te[test_idx],
        "hedge_agrees_with_majority": (hedge_argmax_te[test_idx] == test_maj[test_idx]).astype(int),
        "sub_cell_N": test_subN[test_idx],
    })
    test_df["margin"] = test_df["primary_argmax_prob"] - test_df["primary_majority_prob"]
    test_df["second_is_majority"] = (test_df["primary_2nd_choice"] == test_df["cell_majority"]).astype(int)

    # Filter F1
    f1_set = test_df[test_df["hedge_agrees_with_majority"] == 1]
    print(f"  F1 (hedge agrees): {len(f1_set)} of {n_te}")
    # Filter F2 at τ=0.85
    f2_set = test_df[test_df["primary_max_prob"] < 0.85]
    print(f"  F2 (max_prob < 0.85): {len(f2_set)} of {n_te}")
    # Filter F3
    f3_set = test_df[test_df["second_is_majority"] == 1]
    print(f"  F3 (second is majority): {len(f3_set)} of {n_te}")

    # Combined filter (F1 AND F3)
    fcomb = test_df[(test_df["hedge_agrees_with_majority"] == 1) &
                    (test_df["second_is_majority"] == 1)]
    print(f"  F1+F3 (hedge AGREE AND second=maj): {len(fcomb)}")

    # Meta-discriminator on test
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(X, Y)
    Xt = test_df[feats].to_numpy(dtype=float)
    test_df["meta_prob"] = lr.predict_proba(Xt)[:, 1]
    print("\n  Meta-discriminator on 36 test candidates:")
    print("    distribution of P(primary_wrong):")
    for lo, hi in [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]:
        s = test_df[(test_df["meta_prob"] >= lo) & (test_df["meta_prob"] < hi)]
        print(f"      [{lo:.2f}, {hi:.2f}): n={len(s):3d}")

    # Detailed look at which rows are in each filter set
    print("\n=== Detailed test candidate table (sorted by meta_prob desc) ===")
    cols = ["row_idx","primary_pred","cell_majority","primary_max_prob","margin","second_is_majority","hedge_agrees_with_majority","sub_cell_N","meta_prob"]
    test_df_sorted = test_df.sort_values("meta_prob", ascending=False)
    print(test_df_sorted[cols].to_string(index=False))

    # ---- Save artefacts ----
    summary = {
        "n_l2_disagreements": n_l2_dis,
        "n_primary_wrong_train": n_pw,
        "n_primary_right_train": n_pr,
        "n_test_candidates": n_te,
        "filter_results_train": {
            "hedge_agrees_n_total": int(df["hedge_agrees_with_majority"].sum()),
            "hedge_agrees_primary_wrong_rate": float(df[df["hedge_agrees_with_majority"]==1]["primary_was_wrong"].mean()) if (df["hedge_agrees_with_majority"]==1).sum()>0 else None,
            "hedge_disagrees_primary_wrong_rate": float(df[df["hedge_agrees_with_majority"]==0]["primary_was_wrong"].mean()) if (df["hedge_agrees_with_majority"]==0).sum()>0 else None,
            "second_is_maj_n_total": int(df["second_is_majority"].sum()),
            "second_is_maj_primary_wrong_rate": float(df[df["second_is_majority"]==1]["primary_was_wrong"].mean()) if (df["second_is_majority"]==1).sum()>0 else None,
            "loo_auc_meta": float(auc),
        },
        "test_filter_n": {
            "F1_hedge_agrees": int(len(f1_set)),
            "F2_max_prob_lt_085": int(len(f2_set)),
            "F3_second_is_maj": int(len(f3_set)),
            "F1_AND_F3": int(len(fcomb)),
        },
    }
    out = ART / "l1_filter_validation_results.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {out}")

    # save test_df for downstream use
    test_df.to_csv(ART / "l1_filter_test_candidates.csv", index=False)
    df.to_csv(ART / "l1_filter_train_disagreements.csv", index=False)
    print(f"Saved {ART / 'l1_filter_test_candidates.csv'}")
    print(f"Saved {ART / 'l1_filter_train_disagreements.csv'}")


if __name__ == "__main__":
    main()
