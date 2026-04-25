"""N3 5-shuffle blend gate diagnostic.

Run after pulling N3 outputs from Kaggle:
  kaggle kernels output chrisleitescha/irrigation-n3-5shuffle \
    -p scripts/artifacts/n3_kaggle_output/

This script then:
  1. Loads oof_recipe_5shuffle.npy + test_recipe_5shuffle.npy
  2. Computes standalone tuned bal_acc + bias
  3. Computes Jaccard vs:
     - recipe_full_te (the natural baseline — N3 swaps for it)
     - LB-best 3-stack (recipe + RealMLP + nonrule_iso)
     - LB-best 4-stack / current primary (3-stack + xgb_metastack_iso)
  4. Computes error count at recipe bias for all comparisons
  5. Two blend strategies tested:
     a) ADD: blend N3 onto current primary at fixed recipe bias.
        Sweep α in {0.025, 0.05, ..., 0.5}.
     b) SUBSTITUTE: replace recipe_full_te in the 3-way with N3,
        rebuild the full primary chain, compute new sub-primary OOF.
  6. Apply gates:
     G1: standalone tuned > recipe (0.97967)
     G2: errs ≤ recipe (10114)
     G3: Jaccard vs primary < 0.85
     G4: ADD-blend peak Δ ≥ +2e-4 (LB-transfer threshold)
     G5: SUB-primary OOF > current primary 0.98084
     G6: per-class recall floor (no class drops > 0.0010 vs primary)
  7. Emit a candidate submission only if (G4 AND G3) OR (G5 AND G6).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend, tune_log_bias  # noqa: E402

ART = Path("scripts/artifacts")
SUB = Path("submissions")
DATA = Path("data")
CLASSES = ["Low", "Medium", "High"]
TARGET = "Irrigation_Need"
RECIPE_BIAS = np.array([1.4324, 1.4689, 3.4008])
PRIMARY_OOF_REF = 0.98084   # LB-best 4-stack OOF
PRIMARY_LB_REF = 0.98094    # LB-best 4-stack LB


def _normed(a):
    return a / np.clip(a.sum(1, keepdims=True), 1e-9, None)


def iso_cal(oof, test, y):
    oo = np.zeros_like(oof, dtype=np.float32)
    tt = np.zeros_like(test, dtype=np.float32)
    for c in range(3):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        ir.fit(oof[:, c], (y == c).astype(np.float32))
        oo[:, c] = ir.predict(oof[:, c])
        tt[:, c] = ir.predict(test[:, c])
    return _normed(oo), _normed(tt)


def bal_at_bias(p, y, bias=RECIPE_BIAS):
    return balanced_accuracy_score(
        y, (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1))


def errs_at_bias(p, y, bias=RECIPE_BIAS):
    pred = (np.log(np.clip(p, 1e-12, 1)) + bias).argmax(1)
    return int((pred != y).sum()), pred


def jaccard_err(p1, p2, y, bias=RECIPE_BIAS):
    e1 = (np.log(np.clip(p1, 1e-12, 1)) + bias).argmax(1) != y
    e2 = (np.log(np.clip(p2, 1e-12, 1)) + bias).argmax(1) != y
    inter = (e1 & e2).sum()
    union = (e1 | e2).sum()
    return float(inter / max(union, 1))


def main():
    train = pd.read_csv(DATA / "train.csv")
    y = train[TARGET].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    test = pd.read_csv(DATA / "test.csv")
    test_ids = test["id"].values

    # Load N3 outputs (try Kaggle pull dir first, then artifacts root)
    n3_oof_paths = [
        ART / "n3_kaggle_output" / "oof_recipe_5shuffle.npy",
        ART / "oof_recipe_5shuffle.npy",
    ]
    n3_test_paths = [
        ART / "n3_kaggle_output" / "test_recipe_5shuffle.npy",
        ART / "test_recipe_5shuffle.npy",
    ]
    n3_oof_path = next((p for p in n3_oof_paths if p.exists()), None)
    n3_test_path = next((p for p in n3_test_paths if p.exists()), None)
    if n3_oof_path is None or n3_test_path is None:
        sys.exit(
            f"N3 outputs not found. Searched:\n  "
            + "\n  ".join(str(p) for p in n3_oof_paths + n3_test_paths)
            + "\nFirst pull from Kaggle:\n  "
            "kaggle kernels output chrisleitescha/irrigation-n3-5shuffle "
            f"-p {ART / 'n3_kaggle_output'}/")
    print(f"loading N3 OOF: {n3_oof_path}")
    print(f"loading N3 test: {n3_test_path}")
    n3_o = _normed(np.load(n3_oof_path).astype(np.float32))
    n3_t = _normed(np.load(n3_test_path).astype(np.float32))
    assert n3_o.shape == (len(y), 3), n3_o.shape
    assert n3_t.shape == (len(test), 3), n3_t.shape

    # Load LB-best stack components
    r = _normed(np.load(ART / "oof_recipe_full_te.npy"))
    rt = _normed(np.load(ART / "test_recipe_full_te.npy"))
    s1 = _normed(np.load(ART / "oof_recipe_pseudolabel.npy"))
    s1t = _normed(np.load(ART / "test_recipe_pseudolabel.npy"))
    s7 = _normed(np.load(ART / "oof_recipe_pseudolabel_seed7labeler.npy"))
    s7t = _normed(np.load(ART / "test_recipe_pseudolabel_seed7labeler.npy"))
    rm = _normed(np.load(ART / "oof_realmlp.npy"))
    rmt = _normed(np.load(ART / "test_realmlp.npy"))
    nr = _normed(np.load(ART / "oof_xgb_nonrule.npy"))
    nrt = _normed(np.load(ART / "test_xgb_nonrule.npy"))
    meta_o = _normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = _normed(np.load(ART / "test_xgb_metastack.npy"))

    # Build LB-best primary (4-stack with iso-meta)
    nr_iso_o, nr_iso_t = iso_cal(nr, nrt, y)
    meta_iso_o, meta_iso_t = iso_cal(meta_o, meta_t, y)
    w3 = np.array([0.25, 0.35, 0.40])
    lb3_o = log_blend([r, s1, s7], w3)
    lb3_t = log_blend([rt, s1t, s7t], w3)
    st1_o = log_blend([lb3_o, rm], np.array([0.8, 0.2]))
    st1_t = log_blend([lb3_t, rmt], np.array([0.8, 0.2]))
    st2_o = log_blend([st1_o, nr_iso_o], np.array([0.925, 0.075]))
    st2_t = log_blend([st1_t, nr_iso_t], np.array([0.925, 0.075]))
    primary_o = log_blend([st2_o, meta_iso_o], np.array([0.7, 0.3]))
    primary_t = log_blend([st2_t, meta_iso_t], np.array([0.7, 0.3]))
    primary_bal = bal_at_bias(primary_o, y)
    primary_errs, primary_pred = errs_at_bias(primary_o, y)
    primary_recL = recall_score(y, primary_pred, labels=[0], average=None)[0]
    primary_recM = recall_score(y, primary_pred, labels=[1], average=None)[0]
    primary_recH = recall_score(y, primary_pred, labels=[2], average=None)[0]
    print(f"\ncurrent primary OOF (recipe bias) = {primary_bal:.5f}  errs = {primary_errs}")
    print(f"  per-class recall: L={primary_recL:.4f} M={primary_recM:.4f} H={primary_recH:.4f}")

    # === N3 standalone diagnostics ===
    n3_argmax = balanced_accuracy_score(y, n3_o.argmax(1))
    n3_at_recipe_bias = bal_at_bias(n3_o, y)
    n3_errs_recipe_bias, _ = errs_at_bias(n3_o, y)
    prior = np.bincount(y, minlength=3) / len(y)
    n3_bias, n3_tuned = tune_log_bias(n3_o, y, prior)
    j_recipe = jaccard_err(n3_o, r, y)
    j_lb3 = jaccard_err(n3_o, lb3_o, y)
    j_lbbest = jaccard_err(n3_o, primary_o, y)
    print(f"\n=== N3 standalone ===")
    print(f"  argmax bal_acc      = {n3_argmax:.5f}")
    print(f"  at recipe bias      = {n3_at_recipe_bias:.5f}  errs={n3_errs_recipe_bias}")
    print(f"  tuned (own bias)    = {n3_tuned:.5f}  bias={n3_bias.round(4).tolist()}")
    print(f"  Jaccard vs recipe   = {j_recipe:.4f}")
    print(f"  Jaccard vs LB-3way  = {j_lb3:.4f}")
    print(f"  Jaccard vs primary  = {j_lbbest:.4f}")

    # === Strategy A: ADD N3 onto primary (fixed bias) ===
    print(f"\n=== Strategy A: ADD N3 onto primary at fixed bias ===")
    add_sweep = []
    for a in [0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        blend = log_blend([primary_o, n3_o], np.array([1 - a, a]))
        b = bal_at_bias(blend, y)
        d = b - primary_bal
        add_sweep.append({"alpha": a, "oof": float(b), "delta": float(d)})
        print(f"  α={a:.3f}  OOF={b:.5f}  Δ={d:+.5f}")
    add_best = max(add_sweep, key=lambda r: r["delta"])

    # === Strategy B: SUBSTITUTE recipe in 3-way with N3, rebuild ===
    sub_lb3_o = log_blend([n3_o, s1, s7], w3)
    sub_lb3_t = log_blend([n3_t, s1t, s7t], w3)
    sub_st1_o = log_blend([sub_lb3_o, rm], np.array([0.8, 0.2]))
    sub_st1_t = log_blend([sub_lb3_t, rmt], np.array([0.8, 0.2]))
    sub_st2_o = log_blend([sub_st1_o, nr_iso_o], np.array([0.925, 0.075]))
    sub_st2_t = log_blend([sub_st1_t, nr_iso_t], np.array([0.925, 0.075]))
    sub_primary_o = log_blend([sub_st2_o, meta_iso_o], np.array([0.7, 0.3]))
    sub_primary_t = log_blend([sub_st2_t, meta_iso_t], np.array([0.7, 0.3]))
    sub_bal = bal_at_bias(sub_primary_o, y)
    sub_errs, sub_pred = errs_at_bias(sub_primary_o, y)
    sub_recL = recall_score(y, sub_pred, labels=[0], average=None)[0]
    sub_recM = recall_score(y, sub_pred, labels=[1], average=None)[0]
    sub_recH = recall_score(y, sub_pred, labels=[2], average=None)[0]
    print(f"\n=== Strategy B: SUBSTITUTE N3 for recipe in 3-way ===")
    print(f"  sub-primary OOF = {sub_bal:.5f}  Δ vs primary = {sub_bal-primary_bal:+.5f}")
    print(f"  sub-primary errs = {sub_errs}  Δ = {sub_errs-primary_errs:+}")
    print(f"  recall L={sub_recL:.4f} M={sub_recM:.4f} H={sub_recH:.4f}")
    print(f"  recall Δ: L={sub_recL-primary_recL:+.4f} M={sub_recM-primary_recM:+.4f} H={sub_recH-primary_recH:+.4f}")

    # === Gates ===
    recipe_bal = bal_at_bias(r, y)
    recipe_errs, _ = errs_at_bias(r, y)
    G1 = n3_tuned > recipe_bal                              # standalone > recipe baseline
    G2 = n3_errs_recipe_bias <= recipe_errs                 # errs ≤ recipe
    G3 = j_lbbest < 0.85                                    # Jaccard vs primary
    G4 = add_best["delta"] >= 2e-4                          # ADD blend lift transferable
    G5 = sub_bal > primary_bal                              # SUB-primary > current
    G6 = (sub_recL >= primary_recL - 0.0010 and             # per-class floor
          sub_recM >= primary_recM - 0.0010 and
          sub_recH >= primary_recH - 0.0010)
    add_gate = G1 and G3 and G4
    sub_gate = G5 and G6
    print(f"\n=== Gates ===")
    print(f"  G1 (n3_tuned > recipe {recipe_bal:.5f})        = {G1}")
    print(f"  G2 (n3_errs {n3_errs_recipe_bias} ≤ recipe {recipe_errs}) = {G2}")
    print(f"  G3 (Jaccard vs primary {j_lbbest:.3f} < 0.85)   = {G3}")
    print(f"  G4 (ADD peak Δ {add_best['delta']:+.5f} ≥ +2e-4)  = {G4}")
    print(f"  G5 (SUB-primary > current)                  = {G5}")
    print(f"  G6 (per-class recall floor)                 = {G6}")
    print(f"  ADD strategy passes: G1 ∧ G3 ∧ G4 = {add_gate}")
    print(f"  SUB strategy passes: G5 ∧ G6 = {sub_gate}")

    # Emit submission candidates
    emitted = []
    if add_gate:
        a = add_best["alpha"]
        blend_t = log_blend([primary_t, n3_t], np.array([1 - a, a]))
        pred = (np.log(np.clip(blend_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
        sub = pd.DataFrame({"id": test_ids, TARGET: [CLASSES[i] for i in pred]})
        path = SUB / f"submission_n3_add_a{int(a*1000):03d}.csv"
        sub.to_csv(path, index=False)
        print(f"\nEMIT: ADD strategy → {path}")
        emitted.append(str(path))
    if sub_gate:
        pred = (np.log(np.clip(sub_primary_t, 1e-12, 1)) + RECIPE_BIAS).argmax(1)
        sub = pd.DataFrame({"id": test_ids, TARGET: [CLASSES[i] for i in pred]})
        path = SUB / "submission_n3_sub_primary.csv"
        sub.to_csv(path, index=False)
        print(f"\nEMIT: SUB strategy → {path}")
        emitted.append(str(path))
    if not emitted:
        print(f"\nNo emit. Both gates failed.")

    # Save full diagnostic JSON
    out = {
        "n3_standalone": {
            "argmax": float(n3_argmax),
            "at_recipe_bias": float(n3_at_recipe_bias),
            "errs_at_recipe_bias": n3_errs_recipe_bias,
            "tuned": float(n3_tuned),
            "tuned_bias": n3_bias.tolist(),
            "jaccard_vs_recipe": float(j_recipe),
            "jaccard_vs_lb3way": float(j_lb3),
            "jaccard_vs_primary": float(j_lbbest),
        },
        "current_primary": {
            "oof": float(primary_bal),
            "lb_ref": PRIMARY_LB_REF,
            "errs": primary_errs,
            "recall": [float(primary_recL), float(primary_recM), float(primary_recH)],
        },
        "strategy_add": {
            "sweep": add_sweep,
            "best": add_best,
        },
        "strategy_sub": {
            "oof": float(sub_bal),
            "delta_vs_primary": float(sub_bal - primary_bal),
            "errs": sub_errs,
            "recall": [float(sub_recL), float(sub_recM), float(sub_recH)],
        },
        "gates": {"G1": bool(G1), "G2": bool(G2), "G3": bool(G3),
                  "G4": bool(G4), "G5": bool(G5), "G6": bool(G6),
                  "add_gate": bool(add_gate), "sub_gate": bool(sub_gate)},
        "emitted_submissions": emitted,
    }
    out_path = ART / "n3_blend_gate_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
