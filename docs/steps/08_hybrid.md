# Step 08: Hybrid, ML + residual kriging (regression kriging)

**Status:** **done** 2026-10-18 23:25–23:34 (9 min: out-of-fold fits 9 min,
everything else seconds). Terms in *italics* are explained in [glossary.md](glossary.md).

> **Re-run on the Mac, 2026-10-19 (step 09 uses these numbers):** Re-run with the IDW tie fix: lgbm_rk RMSE 46.00 → **45.94** (R² 0.715 → 0.716), lgbm_idw 44.98 → **45.00**; out-of-fold R² 0.664 → 0.663. **Step 09's CV check (27 blocks) shows the gain is much smaller over many places (R² 0.66 → 0.68) than on these 6 test blocks; see [09_evaluation.md](09_evaluation.md).**

## Purpose

Step 07 found that ML (LightGBM) and interpolation (kriging) are good at
**different things**:

- **ML** learns how PM2.5 depends on the predictors, so it works even far from
  any training cell (it beats interpolation beyond 100 km) and where conditions
  change sharply (northwest, Himalaya).
- **Interpolation** copies what neighbouring cells actually show, so it catches
  local events the predictors don't describe (northeast fire plumes) and wins
  close to training cells.

*Regression kriging* combines them, and gives RQ6 its three-way comparison:
**interpolation only vs ML only vs ML + interpolation**.

## The idea, in plain words

1. ML predicts PM2.5 at every cell from the predictors.
2. At **training** cells we know the true value, so we know ML's error there
   (the *residual* = observed − predicted).
3. ML's errors are spatially correlated: if it is too low at a cell, it is
   usually too low next door as well (e.g. a fire plume it doesn't know about).
4. So at a **test** cell, interpolate the residuals of the surrounding training
   cells (with kriging) and add that correction to the ML prediction.

Prediction at test cell = ML prediction + kriged residuals of nearby training cells.

Far from training cells, kriging weights spread out and the correction fades to
the average residual, so the hybrid falls back to ML. Close to training cells it
behaves like interpolation.

## How it works

Code: `src/pm25/hybrid.py`, `scripts/run_hybrid.py`.

1. **Honest residuals at training cells (out-of-fold).** If we used the final
   step 07 model, which was trained *on* the training cells, its residuals there
   would be much smaller than its real errors at new places, and the correction
   would be too weak. So for each of the 5 CV folds (whole blocks, step 05),
   LightGBM (same tuned setting as step 07) is fitted on the other 4 folds and
   predicts the held-out fold. Every training cell thus gets a prediction from a
   model that never saw it. 5 fits on ~4.7 M rows each; the result is cached in
   `oof_train_lgbm.nc`.
2. **Variogram of the residuals**, per season, on training cells only (same
   recipe as steps 04/06: each time step standardised, 1,000 random time steps per
   season, 75 km bins, exponential model fitted up to 1,000 km).
3. **Kriging weights** (ordinary kriging, one weight matrix per season, as in
   step 06) from training cells to test cells. Correction = weights × residuals
   at each time step, added to the step 07 LightGBM test prediction. Clipped at 0.
4. **Check with IDW:** same, but residuals interpolated with inverse-distance
   weights (8 nearest, power 2) instead of kriging.
5. Score on test cells with the shared metrics, next to interpolation-only
   baselines (nearest, IDW, kriging of PM2.5 itself).

The test-cell ML prediction comes from the step 07 model (fitted on all training
cells), the residuals from the out-of-fold models. This is the standard way of
doing regression kriging with cross-validated residuals.

## Outputs

`results/08_hybrid/`:

| File | Content |
|---|---|
| `metrics_overall.csv`, `metrics_by_region.csv`, `metrics_by_season.csv`, `metrics_by_region_season.csv`, `metrics_by_distance.csv` | test metrics: lgbm, lgbm_rk (+ kriged residuals), lgbm_idw (+ IDW residuals), nearest, idw, kriging |
| `predictions_test.nc` | lgbm, lgbm_rk, lgbm_idw and observed at test cells (not in git) |
| `oof_train_lgbm.nc` | out-of-fold predictions and observed at all training cells (not in git; reused by later steps) |
| `residual_variogram_bins.csv` | residual variograms per season |
| `config.json` | arguments, model setting, out-of-fold scores, fitted variograms |
| `rmse_by_region.png` | test RMSE by region: kriging vs LightGBM vs hybrid |

Log: `results/logs/run_hybrid.log`. Tests: `tests/test_hybrid.py` (each fold
predicted only by a model that didn't see it; correction added and clipped).

## How to run

```bash
conda run -n pm25 python scripts/run_hybrid.py     # reuses oof_train_lgbm.nc if present
```

## Decisions

| Decision | Reason |
|---|---|
| Residuals out-of-fold, not in-sample | In-sample residuals underestimate errors at new places; the correction would be too weak. |
| Same CV folds as tuning | Whole blocks held out, so residuals look like errors at unseen blocks, like the test set. |
| LightGBM only | Best ML model in step 07. |
| Ordinary kriging per season, plus IDW as a check | Same machinery as step 06; IDW shows whether the variogram fit matters. |
| Krige raw residuals, standardised variogram | Residuals are roughly symmetric (unlike PM2.5), so no log needed; weights only depend on the variogram's shape. |

## Results (2020–2024, 5 × 5 split)

### Out-of-fold residuals at training cells

LightGBM predicting training cells it didn't see (block-wise folds):
RMSE 41.9, MAE 19.4, R² 0.66, bias −6.3 (similar to its test-cell scores, as
intended: these residuals look like real errors at new places).

**Residual variograms** (standardised, per season): this time they **do reach a
sill**, unlike the raw PM2.5 variogram (step 04/06), because the model has
already explained the large-scale patterns. Range ≈ **480–530 km** in every
season; nugget ≈ 0 except pre-monsoon (0.23 of the sill: fire plumes vary from
cell to cell). Winter example: γ = 0.41 at 75 km, 0.65 at 180 km, 0.88 at 340 km,
0.95 (sill) at ~490 km. So ML errors are correlated up to ~500 km, which is what
makes the correction work.

### Test cells (1,388,520 values)

| method | what it uses | RMSE | MAE | R² | bias |
|---|---|---|---|---|---|
| **lgbm_rk** | ML + kriged residuals | 46.0 | **17.4** | 0.715 | **−1.4** |
| lgbm_idw | ML + IDW residuals | **45.0** | 18.7 | **0.727** | −1.7 |
| lgbm | ML only (step 07) | 55.0 | 19.1 | 0.593 | −7.9 |
| nearest | interpolation only | 53.7 | 24.0 | 0.611 | −7.9 |
| idw | interpolation only | 55.3 | 23.9 | 0.588 | −7.2 |
| kriging | interpolation only | 55.6 | 22.6 | 0.583 | −6.7 |

R² by region:

| method | igp | northwest | central | south | northeast | himalaya |
|---|---|---|---|---|---|---|
| lgbm_rk | 0.74 | **0.68** | **0.87** | **0.69** | 0.53 | **0.86** |
| lgbm_idw | **0.75** | 0.55 | 0.81 | 0.59 | **0.59** | **0.86** |
| lgbm | 0.59 | 0.67 | 0.79 | 0.67 | 0.33 | 0.80 |
| kriging | 0.50 | 0.44 | 0.80 | 0.63 | 0.42 | 0.70 |
| nearest | 0.57 | 0.01 | 0.71 | 0.60 | 0.58 | −0.01 |

RMSE by region: lgbm_rk igp 58.7 (lgbm 73.9), northwest 23.2 (23.8), central 12.9
(16.3), south 21.5 (22.1), northeast 123.7 (147.1), himalaya 21.2 (24.9).

R² by season (lgbm_rk / lgbm / kriging): winter 0.78 / 0.72 / 0.61, pre-monsoon
**0.59 / 0.39 / 0.47**, monsoon 0.81 / 0.72 / 0.67, post-monsoon 0.80 / 0.74 / 0.65.

RMSE by distance to the nearest training cell:

| method | ≤ 100 km | 100–200 km | 200–300 km |
|---|---|---|---|
| lgbm_rk | 42.1 | **46.2** | 62.8 |
| lgbm_idw | **40.7** | 46.7 | **58.1** |
| lgbm | 54.8 | 51.5 | 68.1 |
| nearest | 44.8 | 58.3 | 74.7 |
| kriging | 49.3 | 57.9 | 74.8 |

### What this means (RQ6)

- **ML + interpolation beats both parts on their own.** R² rises from 0.59 (ML
  only) or 0.58–0.61 (interpolation only) to **0.72–0.73**; RMSE drops by
  ~9–10 µg m⁻³ (−16%). This is the main answer to RQ6: ML-based enhancement
  works best **combined** with interpolation, not instead of it.
- **It fixes the low bias**: −7.9 → −1.4 µg m⁻³. The ML model systematically
  under-predicts high-pollution areas; the neighbouring residuals tell it by how much.
- **It wins at every distance**, and the gain is largest close to training cells
  (≤ 100 km: 54.8 → 42.1) where the neighbours' residuals are most relevant; far
  away (200–300 km) the correction is smaller but still helps (68.1 → 62.8).
- **Biggest gains** where the predictors miss local sources: **IGP** (R² 0.59 →
  0.74), **pre-monsoon / northeast fires** (pre-monsoon R² 0.39 → 0.59), central.
- **Little gain in the northwest** (0.67 → 0.68): residuals at training cells
  (Thar desert) don't say much about the test block (Kutch coast).
- **Kriging vs IDW of residuals:** almost tied. Kriging has the lower MAE (17.4 vs
  18.7) and is better in the northwest and south; IDW has slightly lower RMSE and
  is better in the northeast. Kriging is the principled choice and the default
  for later steps; IDW shows the result doesn't hinge on the variogram fit.
- **Northeast caveat:** kriging the residuals lowers RMSE there (147 → 124) but
  raises MAE (33.3 → 38.5): it catches more of the fire peaks, but spreads peak
  corrections onto ordinary days at nearby cells.

## Open issues

- The residual variogram fit puts the nugget at 0 in three seasons although the
  first bin (75 km) is already at ~40% of the sill; the exponential shape absorbs
  it. Fine for prediction, worth a note in the write-up.
- Out-of-fold models were fitted on 4/5 of training cells, the test-cell ML model
  on all of them; slightly stronger model at test cells than the residuals assume.
  Standard practice; mention as a limitation.
- Step 09 should test whether the hybrid's gain over each part is larger than
  the uncertainty from having only 6 test blocks (bootstrap over blocks).
