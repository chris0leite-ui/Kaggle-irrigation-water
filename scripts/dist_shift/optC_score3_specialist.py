"""Option C — conformal score=3 specialist with AV-score input.

Train binary XGB on score=3 ∩ teacher_argmax=Low rows, target = (y == Medium).
Inputs: 35 dist features + 7 non-rule continuous features + 5 teacher
meta-features (recipe + LB-best 3-stack probs + margin) + 1 AV score.

Mondrian split-conformal: sweep θ s.t. Wilson 90% lower CI on precision
≥ 39.3% (= L/(L+M) macro-recall break-even for L↔M overrides).

Per the diagnostic AUC of P(orig) at score=3 = 0.522 (full train), this
is structurally a long-shot. EV ~5%. But cheap and informative — even
a null result tells us if AV-score adds even 1-2 pp of precision over
the prior 2026-04-26 spec_lm_v3 attempt.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.dist_shift.flip_manifold import _bits, _rule, _score
from scripts.dist_shift.loader import ARTI, NUMS, load
from scripts.tier1b_helpers import build_lbbest_stack, load_y

warnings.filterwarnings("ignore")

NON_RULE_NUMS = ["Soil_pH", "Humidity", "Previous_Irrigation_mm",
                 "Electrical_Conductivity", "Organic_Carbon",
                 "Sunlight_Hours", "Field_Area_hectare"]


def _wilson_lower(k, n, z=1.282):
    """One-sided Wilson 90% lower bound on binomial p (z=1.282 = 90% one-sided)."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / denom


def main() -> None:
    train, test, _ = load()
    y = load_y()
    rng = np.random.RandomState(42)

    # rule + score on train
    b = _bits(train)
    s = _score(b).to_numpy()
    rule = _rule(s)

    # AV scores
    av_train = np.load(ARTI / "oof_av_p_synth_train.npy").astype(np.float32)
    av_test = np.load(ARTI / "test_av_p_synth.npy").astype(np.float32)

    # LB-best 3-stack OOF/test as teacher
    lb3_oof, lb3_test = build_lbbest_stack(y)
    rec_oof = np.load("scripts/artifacts/oof_recipe_full_te.npy")
    rec_test = np.load("scripts/artifacts/test_recipe_full_te.npy")

    # Restrict to score=3 ∩ teacher_argmax=Low
    teacher_arg = lb3_oof.argmax(axis=1)  # 0=L, 1=M, 2=H
    train_mask = (s == 3) & (teacher_arg == 0)
    n_dom = int(train_mask.sum())
    print(f"Train domain (score=3 ∩ teacher=Low): n={n_dom}")
    y_dom = y[train_mask]
    targ_dom = (y_dom == 1).astype(int)  # is Medium
    print(f"  target dist: Low={int((y_dom==0).sum())} Medium={int((y_dom==1).sum())} High={int((y_dom==2).sum())}")
    print(f"  target prevalence (Medium): {targ_dom.mean()*100:.3f}%")

    # Build train features (48 dim)
    def feats(df_full, av_full, rec_full, lb3_full, idx):
        sub = df_full.iloc[idx].reset_index(drop=True)
        out = pd.DataFrame(index=range(len(sub)))
        # raw nums
        for c in NUMS:
            out[c] = sub[c].astype(np.float32).values
        # cats factorized
        for c in ["Soil_Type","Crop_Type","Crop_Growth_Stage","Season",
                  "Irrigation_Type","Water_Source","Mulching_Used","Region"]:
            out[c+"_f"] = pd.factorize(sub[c])[0].astype(np.int32)
        # rule indicators (always 1 for dry+norain, etc., at score=3 — but useful per-cell)
        out["dry"] = (sub["Soil_Moisture"] < 25).astype(np.int8).values
        out["norain"] = (sub["Rainfall_mm"] < 300).astype(np.int8).values
        out["hot"] = (sub["Temperature_C"] > 30).astype(np.int8).values
        out["windy"] = (sub["Wind_Speed_kmh"] > 10).astype(np.int8).values
        out["nomulch"] = (sub["Mulching_Used"] == "No").astype(np.int8).values
        # signed / abs distances to rule thresholds
        out["sm_dist"] = (sub["Soil_Moisture"] - 25).astype(np.float32).values
        out["rf_dist"] = (sub["Rainfall_mm"] - 300).astype(np.float32).values
        out["tc_dist"] = (sub["Temperature_C"] - 30).astype(np.float32).values
        out["ws_dist"] = (sub["Wind_Speed_kmh"] - 10).astype(np.float32).values
        # av-score (the new feature)
        out["av_p_synth"] = av_full[idx]
        out["av_p_orig"] = 1.0 - av_full[idx]
        # teacher meta-features
        out["rec_pL"] = rec_full[idx, 0]
        out["rec_pM"] = rec_full[idx, 1]
        out["rec_pH"] = rec_full[idx, 2]
        out["lb3_pL"] = lb3_full[idx, 0]
        out["lb3_pM"] = lb3_full[idx, 1]
        out["lb3_pH"] = lb3_full[idx, 2]
        out["lb3_margin_LM"] = lb3_full[idx, 1] - lb3_full[idx, 0]
        return out

    # Build train rows (only domain-belonging) — but features still use full df idx
    train_idx_dom = np.where(train_mask)[0]
    X_dom = feats(train, av_train, rec_oof, lb3_oof, train_idx_dom)
    print(f"Feature dim: {X_dom.shape[1]}")

    # 5-fold StratifiedKFold within the domain (stratified on Medium=target)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_p = np.zeros(n_dom, dtype=np.float32)
    fold_aucs = []
    for f, (tr, va) in enumerate(cv.split(X_dom, targ_dom), 1):
        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=1.0,
            eval_metric="logloss", early_stopping_rounds=50,
            tree_method="hist", n_jobs=4, random_state=42,
        )
        clf.fit(X_dom.iloc[tr], targ_dom[tr],
                eval_set=[(X_dom.iloc[va], targ_dom[va])], verbose=False)
        oof_p[va] = clf.predict_proba(X_dom.iloc[va])[:, 1]
        auc = roc_auc_score(targ_dom[va], oof_p[va])
        fold_aucs.append(float(auc))
        print(f"  fold {f}: AUC={auc:.4f}  best_iter={clf.best_iteration}")

    overall_auc = roc_auc_score(targ_dom, oof_p)
    print(f"\nOverall OOF AUC (target=Medium on score=3∩teacher=Low) = {overall_auc:.4f}")
    print(f"prior 2026-04-26 spec_lm_v3 (no AV-score) AUC was 0.827; we have AUC {overall_auc:.4f}")

    # Top-K precision under macro-recall L↔M break-even = M/(M+L)*100% = 39.3%
    # On training y_dom: that's targ_dom.mean()/(1 - targ_dom.mean())  but
    # we use the macro-recall-derived 39.3% from CLAUDE.md.
    BREAK_EVEN = 0.393
    print("\nTop-K precision (rank by P(Medium) desc):")
    order = np.argsort(-oof_p)
    targ_sorted = targ_dom[order]
    pk_rows = []
    for k in [50, 100, 200, 500, 1000, 2000]:
        if k > n_dom:
            continue
        nk = int(targ_sorted[:k].sum())
        prec = nk / k
        wlow = _wilson_lower(nk, k)
        meets = wlow >= BREAK_EVEN
        pk_rows.append({"K": k, "n_correct": nk, "precision": round(prec, 4),
                        "wilson_lower_90": round(wlow, 4),
                        "meets_break_even": meets})
    print(pd.DataFrame(pk_rows).to_string(index=False))

    # Conformal-style: pick smallest K (largest θ) where wilson_lower ≥ break-even
    feasible = [r for r in pk_rows if r["meets_break_even"]]
    print(f"\nConformal-feasible operating points (Wilson 90% lower CI ≥ {BREAK_EVEN}):")
    if feasible:
        for r in feasible:
            print(f"  K={r['K']}: precision={r['precision']:.4f} wilson_lower={r['wilson_lower_90']:.4f}")
    else:
        print("  NONE — top-K precision never clears the macro-recall break-even floor.")

    # Compute macro-recall delta of the largest feasible K
    if feasible:
        # largest K = most overrides
        best = feasible[-1]
        K = best["K"]
        n_M_total = int((y == 1).sum())
        n_L_total = int((y == 0).sum())
        gain = best["n_correct"] / n_M_total
        loss = (K - best["n_correct"]) / n_L_total
        macro_delta = (gain - loss) / 3
        print(f"\nIf deployed at K={K}: macro-recall Δ on OOF = {macro_delta:+.6f}")
        print(f"  M-recall gain  = +{gain:.6f}")
        print(f"  L-recall loss  = -{loss:.6f}")
    else:
        print("\nNo deployable operating point — Option C closes here.")

    out = {
        "n_domain": n_dom,
        "auc_overall": float(overall_auc),
        "fold_aucs": fold_aucs,
        "target_prevalence": float(targ_dom.mean()),
        "top_k_precision": pk_rows,
        "break_even_floor": BREAK_EVEN,
        "feasible_K": [r["K"] for r in feasible],
    }
    (ARTI / "optC_score3_specialist_results.json").write_text(json.dumps(out, indent=2, default=str))
    np.save(ARTI / "oof_optC_score3_pmedium.npy", oof_p)
    print(f"\nWrote {ARTI/'optC_score3_specialist_results.json'}")


if __name__ == "__main__":
    main()
