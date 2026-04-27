"""Extended purity search: find 100%-pure sub-cells within the 128-cell rule
cube, conditioned on each of the 6 untouched categorical features (Soil_Type,
Crop_Type, Season, Irrigation_Type, Water_Source, Region).

For each cell × cat × value tuple, count train rows and mismatch rate vs the
cell-majority class. Sub-cells with 0 errors are 100%-deterministic and can
be dropped from training without losing learnable signal.

Output:
  - scripts/artifacts/purity_subcells.json  (summary stats + drop_set sizes)
  - scripts/artifacts/purity_subcell_rules.csv (full rule table)
  - scripts/artifacts/drop_mask_train.npy  (boolean drop mask for tr_idx, length 630k)
  - scripts/artifacts/drop_mask_test.npy   (boolean drop mask for te_idx, length 270k)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import log_blend  # noqa: E402
from purity_rules_diag import compute_rule  # noqa: E402
from tier1b_helpers import (  # noqa: E402
    ART, BIAS, CLASSES, CLS2IDX, build_lbbest_stack, iso_cal, log, normed,
)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

# 6 cats not yet partitioned by cell_id (Crop_Growth_Stage and Mulching_Used
# are already in the cell). These are the candidate axes for sub-cell rules.
SUB_CATS = [
    "Soil_Type", "Crop_Type", "Season",
    "Irrigation_Type", "Water_Source", "Region",
]


def main():
    log("loading data...")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = train["Irrigation_Need"].map(CLS2IDX).to_numpy().astype(np.int32)

    rt = compute_rule(train)
    re = compute_rule(test)
    cell_tr = rt["cell_id"]
    cell_te = re["cell_id"]

    # ---- Per-cell train majority class ----
    log("computing per-cell majority and impure-row mask...")
    cells = np.unique(cell_tr)
    cell_majority = {}
    cell_n = {}
    cell_n_err = {}
    for cid in cells:
        m = cell_tr == cid
        n = int(m.sum())
        cnt = np.bincount(y[m], minlength=3)
        maj = int(np.argmax(cnt))
        cell_majority[int(cid)] = maj
        cell_n[int(cid)] = n
        cell_n_err[int(cid)] = int(n - cnt[maj])

    # ---- Cell-level pure-row mask (100%-pure cells) ----
    pure_cell_ids = sorted(cid for cid in cell_majority if cell_n_err[cid] == 0)
    log(f"  cell-level 100%-pure cells: {len(pure_cell_ids)} "
        f"covering train_n={sum(cell_n[c] for c in pure_cell_ids)}, "
        f"test_n={int(np.isin(cell_te, pure_cell_ids).sum())}")

    cell_drop_tr = np.isin(cell_tr, pure_cell_ids)
    cell_drop_te = np.isin(cell_te, pure_cell_ids)

    # ---- Sub-cell search: cell × cat × value → 100%-pure tuple ----
    log("")
    log("sub-cell purity search across 6 untouched cats...")
    rules: list[dict] = []
    sub_drop_tr = np.zeros(len(train), dtype=bool)
    sub_drop_te = np.zeros(len(test), dtype=bool)

    for cid in cells:
        if cell_n_err[int(cid)] == 0:
            continue  # already entirely pure; sub-cell search is moot
        m_tr_cell = cell_tr == cid
        m_te_cell = cell_te == cid
        if m_te_cell.sum() == 0:
            continue
        majority = cell_majority[int(cid)]
        # Within cell: which rows are impure?
        cell_y = y[m_tr_cell]
        impure_in_cell = cell_y != majority

        for cat in SUB_CATS:
            cat_tr = train.loc[m_tr_cell, cat].to_numpy()
            cat_te = test.loc[m_te_cell, cat].to_numpy()
            if len(cat_te) == 0:
                continue
            for val in pd.unique(cat_tr):
                m_tr_val = cat_tr == val
                # Test side requires the value to exist in test
                m_te_val = cat_te == val
                tr_n = int(m_tr_val.sum())
                te_n = int(m_te_val.sum())
                if tr_n < 30:  # min sample to declare "100% pure" with confidence
                    continue
                impure_in_subcell = int(impure_in_cell[m_tr_val].sum())
                if impure_in_subcell != 0:
                    continue
                # 100%-pure sub-cell. Add its train+test rows to drop sets.
                # Need indices into the full train/test arrays
                idx_tr_cell = np.where(m_tr_cell)[0]
                idx_te_cell = np.where(m_te_cell)[0]
                idx_tr_sub = idx_tr_cell[m_tr_val]
                idx_te_sub = idx_te_cell[m_te_val]
                # Don't double-mark rows already covered by cell-level rule
                already_tr = cell_drop_tr[idx_tr_sub].sum()
                already_te = cell_drop_te[idx_te_sub].sum()
                new_tr = tr_n - int(already_tr)
                new_te = te_n - int(already_te)
                # Mark
                sub_drop_tr[idx_tr_sub] = True
                sub_drop_te[idx_te_sub] = True
                rules.append({
                    "cell_id": int(cid),
                    "cell_score": int(rt["score"][m_tr_cell][0]),
                    "cell_majority": CLASSES[majority],
                    "cell_purity": (cell_n[int(cid)] - cell_n_err[int(cid)]) / cell_n[int(cid)],
                    "cell_tr_n": cell_n[int(cid)],
                    "cell_n_err": cell_n_err[int(cid)],
                    "subcat": cat,
                    "subval": str(val),
                    "tr_n": tr_n, "te_n": te_n,
                    "new_tr": new_tr, "new_te": new_te,
                })

    log(f"  found {len(rules)} sub-cell rules with 100%-pure tr_n>=30")

    # Deduplicate: a row may be covered by multiple sub-cell rules; keep just
    # the union (sub_drop_tr / sub_drop_te already reflect this).
    total_drop_tr = cell_drop_tr | sub_drop_tr
    total_drop_te = cell_drop_te | sub_drop_te
    log(f"  total deterministic train coverage = {int(total_drop_tr.sum())} "
        f"({total_drop_tr.mean()*100:.2f}% of train)")
    log(f"  total deterministic test coverage  = {int(total_drop_te.sum())} "
        f"({total_drop_te.mean()*100:.2f}% of test)")

    # ---- Class breakdown of dropped rows ----
    log("")
    log("dropped-rows class composition:")
    if total_drop_tr.sum() > 0:
        cls_dropped = np.bincount(y[total_drop_tr], minlength=3)
        cls_full = np.bincount(y, minlength=3)
        log(f"  train labels in drop set: L={cls_dropped[0]} M={cls_dropped[1]} H={cls_dropped[2]}")
        for c in range(3):
            pct = cls_dropped[c] / cls_full[c] * 100 if cls_full[c] > 0 else 0
            log(f"    {CLASSES[c]}: dropped {cls_dropped[c]} of {cls_full[c]} ({pct:.1f}%)")

        retained_y = y[~total_drop_tr]
        cls_retained = np.bincount(retained_y, minlength=3)
        log(f"  retained train labels: L={cls_retained[0]} M={cls_retained[1]} H={cls_retained[2]}")
        log(f"  retained class shares: "
            f"L={cls_retained[0]/len(retained_y):.4f} "
            f"M={cls_retained[1]/len(retained_y):.4f} "
            f"H={cls_retained[2]/len(retained_y):.4f} "
            f"(vs full L={cls_full[0]/len(y):.4f} "
            f"M={cls_full[1]/len(y):.4f} "
            f"H={cls_full[2]/len(y):.4f})")

    # ---- Sanity: confirm primary already nails dropped TEST rows ----
    log("")
    log("sanity: confirm primary already nails dropped TEST rows...")
    s2_o, s2_t = build_lbbest_stack(y)
    meta_o = normed(np.load(ART / "oof_xgb_metastack.npy"))
    meta_t = normed(np.load(ART / "test_xgb_metastack.npy"))
    meta_o, meta_t = iso_cal(meta_o, meta_t, y)
    primary_t = log_blend([s2_t, meta_t], np.array([0.7, 0.3]))
    test_argmax = (np.log(np.clip(primary_t, 1e-12, 1)) + BIAS).argmax(1)
    test_majority = np.zeros(len(test), dtype=np.int8)
    for i, cid in enumerate(cell_te):
        test_majority[i] = cell_majority[int(cid)]
    primary_disagree_te = int((test_argmax[total_drop_te] != test_majority[total_drop_te]).sum())
    log(f"  primary disagreements with cell-majority on dropped TEST rows: {primary_disagree_te}")

    # ---- Save ----
    out = {
        "n_pure_cells": len(pure_cell_ids),
        "pure_cell_ids": pure_cell_ids,
        "n_subcell_rules": len(rules),
        "n_drop_train": int(total_drop_tr.sum()),
        "n_drop_test": int(total_drop_te.sum()),
        "frac_drop_train": float(total_drop_tr.mean()),
        "frac_drop_test": float(total_drop_te.mean()),
        "primary_disagree_on_dropped_test": primary_disagree_te,
        "drop_class_breakdown_train": [
            int(np.bincount(y[total_drop_tr], minlength=3)[c]) for c in range(3)
        ] if total_drop_tr.sum() > 0 else [0, 0, 0],
        "retained_class_shares": (
            [float(np.bincount(y[~total_drop_tr], minlength=3)[c]
                   / max((~total_drop_tr).sum(), 1)) for c in range(3)]
            if (~total_drop_tr).sum() > 0 else [0.0, 0.0, 0.0]
        ),
        "full_class_shares": [float(np.bincount(y, minlength=3)[c] / len(y)) for c in range(3)],
    }
    (ART / "purity_subcells.json").write_text(json.dumps(out, indent=2))
    log(f"saved {ART / 'purity_subcells.json'}")

    pd.DataFrame(rules).to_csv(ART / "purity_subcell_rules.csv", index=False)
    log(f"saved {ART / 'purity_subcell_rules.csv'} ({len(rules)} rules)")

    np.save(ART / "drop_mask_train.npy", total_drop_tr)
    np.save(ART / "drop_mask_test.npy", total_drop_te)
    log(f"saved drop_mask_train.npy + drop_mask_test.npy")


if __name__ == "__main__":
    main()
