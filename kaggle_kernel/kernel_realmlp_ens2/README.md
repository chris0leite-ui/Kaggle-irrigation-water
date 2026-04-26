# RealMLP n_ens=2 @ n_epochs=40 — diagnostic kernel

Mechanism diagnostic between `n_ens=1` (LB +0.00003 in 3-stack, the only
NN family that ever transferred) and `n_ens=4` (NULL — under-converged at
n_epochs=25 due to 1h cap).

`n_ens=2 @ n_epochs=40` disambiguates:
- **Beats n_ens=1 standalone** → under-convergence was the issue;
  schedule overnight `n_ens=4 @ n_epochs=40` on a non-Kaggle box.
- **Plateaus at n_ens=1** → variance floor structural; lever closed.

## Usage

```bash
# 1) SMOKE first (IS_SMOKE = True at top of script).
cd kaggle_kernel/kernel_realmlp_ens2
kaggle kernels push -p .

# Watch it on https://www.kaggle.com/code/chrisleitescha/irrigation-realmlp-ens2
# Smoke is 2-fold × 20k rows × 3 epochs; ~5 min wall.

# 2) After smoke completes (any non-zero OOF tuned), flip:
#      IS_SMOKE = False
sed -i 's/^IS_SMOKE = True .*/IS_SMOKE = False  # production/' \
    realmlp_pytabkit_ens2.py
kaggle kernels push -p .

# Production: 5-fold × 504k × 40 epochs × n_ens=2; ETA ~50-55 min.
# Hard kill at t+55min if wall budget exceeded.

# 3) Pull artifacts:
mkdir -p ../output_realmlp_ens2
kaggle kernels output chrisleitescha/irrigation-realmlp-ens2 \
    -p ../output_realmlp_ens2
cp ../output_realmlp_ens2/oof_realmlp_ens2.npy ../../scripts/artifacts/
cp ../output_realmlp_ens2/test_realmlp_ens2.npy ../../scripts/artifacts/
cp ../output_realmlp_ens2/realmlp_ens2_results.json ../../scripts/artifacts/

# 4) Run the blend gate:
python ../../scripts/blend_realmlp_ens2.py
```

## Decision rule (post-pull)

| signal | action |
|---|---|
| `tuned OOF >= 0.97700` AND `Jaccard < 0.62` AND `errs <= 1.04 * anchor` | LB-probe: replace realmlp in 3-stack, sweep alpha {0.15, 0.20, 0.25} |
| `tuned OOF >= n_ens=1's 0.97636 + 0.0005` | Plan n_ens=4 @ n_epochs=40 overnight (3h on local 16-core box if available) |
| `tuned OOF` plateaus at n_ens=1 | Variance floor structural; close the RealMLP lever |
| Magnitude trap (`errs > 1.05 * anchor`) | NULL; mirrors n_ens=4 — variance floor structural |

## Wall-time risk

Two heads at full epochs ≈ 1.6× n_ens=1's ~38min wall = ~60min projected.
The 55-min hard kill may catch the run mid-fold-5. Partial outputs
(3-4 fold OOF + test_pred rescaled) save cleanly per the existing kill
guards. Even a 4-fold result is informative for the diagnostic.
