"""Deploy missed-High detector as a hard override on LB-best teacher.

Rule: for each row, if teacher_pred ∈ {Low, Medium} AND P_missed_high > θ,
flip prediction to High. Never override teacher when it already predicts High.
θ-sweep on OOF, report per-class recall, total errors, and bal_acc delta.

Also evaluates the score-band-restricted variant: override only for rows
with dgp_score ∈ {5, 6, 7, 8} (where 95% of missed-High live per diagnostic).

Emit gate:
    Δ bal_acc ≥ +0.00020 AND
    High recall rises AND
    Low+Medium recall does not drop more than the High gain
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from common import fast_bal_acc
from meta_common import ART, EPS, build_teacher, load_y_and_features, recipe_bias

OUT_JSON = ART / "missed_high_deploy_results.json"

CSV_PATH = Path("submissions")

THETAS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SCORE_BANDS = {
    "all_scores": None,
    "score_5_6_7_8": {5, 6, 7, 8},
    "score_6_only": {6},
}


def recalls(pred: np.ndarray, y: np.ndarray) -> dict:
    return {
        "Low": float((pred[y == 0] == 0).mean()),
        "Med": float((pred[y == 1] == 1).mean()),
        "High": float((pred[y == 2] == 2).mean()),
    }


def main() -> None:
    y, tr_score, *_ = load_y_and_features()
    oof_t, test_t = build_teacher()
    bias = recipe_bias()
    log_t_oof = np.log(np.clip(oof_t, EPS, 1.0))
    log_t_test = np.log(np.clip(test_t, EPS, 1.0))
    t_pred_oof = (log_t_oof + bias).argmax(1)
    t_pred_test = (log_t_test + bias).argmax(1)

    p_miss_oof = np.load(ART / "oof_missed_high.npy")
    p_miss_test = np.load(ART / "test_missed_high.npy")

    t_ba = fast_bal_acc(y, t_pred_oof)
    t_rec = recalls(t_pred_oof, y)
    t_errs = int((t_pred_oof != y).sum())
    print(f"Teacher OOF bal_acc = {t_ba:.5f}  errs={t_errs}")
    print(f"Teacher per-class recall: Low={t_rec['Low']:.4f} Med={t_rec['Med']:.4f} "
          f"High={t_rec['High']:.4f}")
    print()

    import pandas as pd
    te = pd.read_csv("data/test.csv")
    te_ids = te["id"].to_numpy()
    CLS = ["Low", "Medium", "High"]

    sweep_results = {}
    best = {"bal_acc": t_ba, "label": "teacher", "theta": None, "band": None}
    for band_name, allowed_scores in SCORE_BANDS.items():
        if allowed_scores is None:
            score_mask = np.ones_like(tr_score, dtype=bool)
        else:
            score_mask = np.isin(tr_score, list(allowed_scores))
        band_results = {}
        for theta in THETAS:
            gate = (t_pred_oof != 2) & (p_miss_oof > theta) & score_mask
            pred = np.where(gate, 2, t_pred_oof)
            ba = fast_bal_acc(y, pred)
            rec = recalls(pred, y)
            n_over = int(gate.sum())
            correct_overrides = int((gate & (y == 2)).sum())
            false_overrides = int((gate & (y != 2)).sum())
            band_results[f"{theta:.2f}"] = dict(
                bal_acc=float(ba),
                delta=float(ba - t_ba),
                n_overridden=n_over,
                correct_overrides=correct_overrides,
                false_overrides=false_overrides,
                precision=correct_overrides / max(n_over, 1),
                recall_Low=rec["Low"],
                recall_Med=rec["Med"],
                recall_High=rec["High"],
            )
            if ba > best["bal_acc"]:
                best = dict(bal_acc=float(ba), label=band_name,
                            theta=theta, band=band_name)
        sweep_results[band_name] = band_results
        print(f"[{band_name}] band rows={score_mask.sum()}")
        print(f"  {'θ':>5s}  {'n_ovr':>6s}  {'correct':>7s}  {'false':>5s}  "
              f"{'prec':>5s}  {'bal_acc':>9s}  {'delta':>9s}  {'recall H':>9s}")
        for theta_key, d in band_results.items():
            print(f"  {theta_key:>5s}  {d['n_overridden']:>6d}  "
                  f"{d['correct_overrides']:>7d}  {d['false_overrides']:>5d}  "
                  f"{d['precision']:>5.2f}  {d['bal_acc']:.5f}  "
                  f"{d['delta']:+.5f}  {d['recall_High']:>9.4f}")
        print()

    print(f"BEST: {best}")
    summary = dict(
        teacher_bal_acc=float(t_ba),
        teacher_recall=t_rec,
        sweep=sweep_results,
        best=best,
        thetas=THETAS,
        score_bands=list(SCORE_BANDS.keys()),
    )
    OUT_JSON.write_text(json.dumps(summary, indent=2))

    # Emit a diagnostic submission at the best config (regardless of gate)
    if best["theta"] is not None and best["bal_acc"] > t_ba + 1e-6:
        band = best["band"]
        allowed = SCORE_BANDS[band]
        te_score_np = np.array([
            -1 for _ in range(len(te_ids))  # placeholder; fill below
        ])
        import pandas as pd
        te_full = pd.read_csv("data/test.csv")
        from common import add_distance_features
        te_d = add_distance_features(te_full)
        te_score_np = te_d["dgp_score"].to_numpy()
        te_score_mask = (np.ones_like(te_score_np, dtype=bool)
                         if allowed is None else np.isin(te_score_np, list(allowed)))
        gate_test = ((t_pred_test != 2) & (p_miss_test > best["theta"])
                     & te_score_mask)
        pred_test = np.where(gate_test, 2, t_pred_test)
        labels = np.array([CLS[c] for c in pred_test])
        out = pd.DataFrame({"id": te_ids, "Irrigation_Need": labels})
        name = f"submission_missed_high_override_{band}_theta{int(best['theta']*100)}.csv"
        out.to_csv(CSV_PATH / name, index=False)
        print(f"wrote {CSV_PATH / name}")


if __name__ == "__main__":
    main()
