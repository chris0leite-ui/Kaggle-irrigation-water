# Legacy index

Pruned 2026-04-22 after the multi-branch consolidation merge. Everything
here is recoverable via `git mv legacy/null/foo.py scripts/foo.py` if a
hypothesis becomes live again.

## `legacy/null/` — superseded or null scripts

96 scripts moved out of `scripts/` because their findings are documented
as null in `CLAUDE.md` / `NEXT_STEPS.md` / `LEARNINGS.md`, OR they were
superseded by a later version (e.g., `xgb_dist_routed.py` →
`xgb_dist_routed_v3.py`). The full per-script readout lives in those
docs; broad categories:

- **Routed-XGB iterations** (`xgb_dist_routed{.py,_v2,_v4,_v5,_v6,_v7,_v3_seed7}`): only `_v3` is canonical.
- **Specialist iterations** (`xgb_specialist_{3,46,6,678_aug,678_seed7}`, `xgb_per_class_specialists`): only `xgb_specialist_678.py` survived the 20–80% class-ambiguity heuristic.
- **NN attempts** (`mlp_*.py`, `nn_orig_features.py`, `xgb_dist_with_nn_feats`): all plateaued at ~0.965 OOF and blend-null.
- **Tree-FE attempts** (`benchmark_fe`, `benchmark_xgb_dist_fe`, `seed_bag_dist_fe`): trees at 127 leaves already find the interactions; engineered products redundant.
- **Empirical-Bayes / per-cell** (`empirical_bayes_cell`, `per_cell_lr*`): plateau at 0.963 because they only see the 6 rule features.
- **Rank/Borda/hard-vote blends** (`rank_blend`, `blend_*_vote*`, `blend_borda*`, `blend_pairwise_veto`, `blend_rule_deferred`, `blend_high_supermajority`): rank-space loses the calibration log-bias needs.
- **Pseudo-label / self-distill** (`pseudo_label_*`, `self_distill_xgb`): boundary-error compounding.
- **Gated/noise-inversion / GCE / score-experts**: see CLAUDE.md 2026-04-21 entries.
- **TE-encode variants** (`benchmark_te_oof`, `te_targets*`, `te_xgb_regression`, `blend_te_reg`): all null.
- **Hyperopt — LGBM** (`hyperopt_lgbm`, `finalize_lgbm`, `lgbm_competitor_baseline`): plateau at the same OOF.
- **Hyperopt — XGB Optuna 80-trial** (`hp_common`, `hp_dist_routed`, `hp_nonrule`, `hp_spec_678`, `refit_best_hp`, `blend_tuned_greedy`): inner-val and outer-CV reward shallow + heavily regularized HPs that DON'T transfer to LB. Both production and peak-α candidates regressed −0.00016 / −0.00021 vs baseline LB despite +0.00034–0.00040 OOF gain. New rule: require blend-level lift ≥ +0.001 before LB-probing HP changes.
- **Ordinal decomposition** (`ordinal_corn`, `ordinal_tabpfn`): CORN trades High recall for Medium (blend monotone-neg). TabPFN +16.7 % errors vs greedy → blend monotone-neg. Both null.
- **CatBoost Optuna** (`catboost_optuna`, `catboost_jaccard_blend`): OOF 0.97179 < LGBM-dist 0.97266; blend peak α=0.05 → +0.00005 (non-signal).
- **Binary High head** (`binary_high_head`): selection-overfit on hybrid blend (+0.00036 OOF / −0.00084 LB), monotone-negative on greedy with fixed bias.
- **DGP-archaeology variants** (`benchmark_dgp_fe2`, `benchmark_oracle`, `dgp_archaeology`, `archaeology_id_mod`, `band_routed_lgbm`, `knn_six_features`, `weighted_lgbm_dgp`): superseded by the canonical 6-feature rule in `scripts/dgp_formula.py`.

## `legacy/submissions/` — stale candidate CSVs

67 submission CSVs whose OOF or LB lost to the current best. Kept for
git history but moved out of `submissions/` so the active set is
scannable. The 9 retained at top level cover:

| File | LB | Role |
|---|---|---|
| `submission_greedy_nonrule_blend.csv` | **0.97352** | **Primary final-selection candidate.** |
| `submission_xgb_hybrid_v3_routed012_spec678.csv` | 0.97271 | **Safe fallback for final selection.** |
| `submission_blend_greedy_w045_040_015.csv` | 0.97296 | Greedy 3-way log-blend (anchor for the LB-best). |
| `submission_blend_lgbm_xgb_dist.csv` | 0.97170 | Calibration-ladder waypoint. |
| `submission_xgb_hybrid_routed_spec.csv` | 0.97224 | Calibration-ladder waypoint. |
| `submission_lgbm_dgp_tuned.csv` | 0.97137 | Calibration-ladder waypoint. |
| `submission_baseline_lgbm_tuned.csv` | 0.96972 | Calibration-ladder anchor (first sub). |
| `submission_dgp_formula.csv` | 0.95835 | Pure rule, ladder anchor. |
| `submission_hybrid_lgbmxgb_blend.csv` | not LB-probed | Reproducibility reference for `oof_hybrid_lgbmxgb_blend.npy`. |

## `legacy/playbook-patch/` — pre-existing
Methodology snapshot, untouched.
