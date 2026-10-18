# Step 06: Baselines

**Status:** done on the **full 2020–2024 data** (14,616 time steps; run
2026-10-18 17:57, 21 s, 3.6 GB). A test run on 2020–2021 came first (17:16);
its results were within ~4 µg m⁻³ RMSE except the log model (see below). Split:
5 × 5 blocks (step 05), 399 training cells, 95 test cells.

## Purpose

Every ML model must beat simple methods to be worth anything (CLAUDE.md rule).
Two kinds of baseline answer two different questions:

- **Spatial interpolation** (RQ6): fill in PM2.5 at a test cell from **PM2.5 at
  training cells at the same time step**. This is the standard way to "enhance"
  a map and the method ML has to beat in RQ6.
- **Linear regression** (RQ1, RQ2): predict PM2.5 at a cell from the **predictors
  at that same cell** (no PM2.5 from neighbours), exactly the task the ML models
  in step 07 do. The simplest version of that task.

Note: interpolation gets to see the target variable (PM2.5) at nearby cells;
regression does not. They use different information, so they are not directly
competing; the hybrid in step 08 combines both.

## How it works

Code: `src/pm25/baselines.py` (interpolation weights), `src/pm25/metrics.py`
(metrics shared by all later steps), `scripts/run_baselines.py`.

**Key trick:** training and test cells are in the same place at every time step,
so each interpolation method is a fixed weight matrix W (95 test × 399 train),
computed once. Predictions for all 14,616 time steps = one matrix multiplication.
Kriging over all 3-hourly steps therefore takes **< 1 s** instead of hours.

| Method | How a test cell is predicted |
|---|---|
| `train_mean` | Average of all 399 training cells at that time. "Do-nothing" reference. |
| `nearest` | Value of the closest training cell (great-circle distance). |
| `idw` | Inverse-distance weighted mean of the **8** nearest training cells, weight = 1/d² (k and power fixed, not tuned). |
| `linear_delaunay` | Piecewise-linear interpolation on a Delaunay triangulation of the training cells (in km, equirectangular projection). This is the **scattered-point version of bilinear**: true bilinear needs a full regular grid, which no longer exists once test blocks are removed. Test cells outside all triangles (**16 of 95**, coastal/border) use `nearest`. |
| `kriging` | **Ordinary kriging** (weights sum to 1, unknown constant mean) with an exponential variogram fitted **on training cells only**, one per season, using the step 04 recipe (log1p, standardised per time step, 1,000 random time steps per season, fit up to 1,000 km). Applied to raw PM2.5. Checked against `pykrige` in the tests: identical predictions. |
| `linreg` | Ordinary least squares on 12 predictors + lat + lon, trained on all training rows (399 cells × 14,616 times = 5.83 M rows). |
| `linreg_log` | Same, but on log(1 + PM2.5); predictions back-transformed with exp(x) − 1. |

**Metrics** (`pm25.metrics`), on test cells only, all time steps:
RMSE, MAE, R² = 1 − SSE/SST (within the group being scored), bias = mean(pred − obs),
so **negative bias = predicts too low**. Reported overall, by region, by season,
region × season and by distance from the test cell to the nearest training cell
(0–100, 100–200, 200–300 km).

## Outputs

`results/06_baselines/` (full data; the 2020–2021 test run is in `results/06_baselines_2020_2021/`):

| File | Content |
|---|---|
| `metrics_overall.csv`, `metrics_by_region.csv`, `metrics_by_season.csv`, `metrics_by_region_season.csv`, `metrics_by_distance.csv` | all metrics, long format (method × group) |
| `predictions_test.nc` | obs + every method's prediction, dims (time, cell), with lat, lon, region, distance coords. Step 09 reuses these. |
| `linreg_coefficients.csv` | regression coefficients (raw and log) |
| `kriging_variogram_bins.csv` | empirical variograms (train cells) per season |
| `config.json` | all arguments, cell counts, fitted kriging variograms |
| `rmse_overall.png`, `rmse_by_region.png` | figures |

Log: `results/logs/run_baselines.log`. Tests: `tests/test_baselines.py`
(nearest, IDW weights, Delaunay reproduces a linear field exactly, fallback outside
the hull, kriging weights sum to 1 and match pykrige), `tests/test_metrics.py`.

## How to run

```bash
conda run -n pm25 python scripts/run_baselines.py                    # full data (final run)
conda run -n pm25 python scripts/run_baselines.py \
    --data data/processed/eac4_india_3h_2020_2021.nc --out-dir results/06_baselines_2020_2021
```

Full data: **21 s, peak memory 3.6 GB** (2020–2021: 13 s, 1.8 GB). Fine on the local machine.

## Decisions

| Decision | Reason |
|---|---|
| Interpolate from **India training cells only** (not from cells outside India) | Same information as the ML models get; neighbouring countries/sea are not part of training. |
| Delaunay-linear instead of grid bilinear | Bilinear is undefined on a grid with holes; Delaunay-linear is its direct equivalent for scattered points. |
| No tuning of IDW (k = 8, power 2) or kriging | Tuning on test cells would be cheating; standard defaults are fine for baselines. Could tune with the CV folds later if needed. |
| Kriging variogram from training cells, per season | Test cells must not influence any choice. Seasons have different ranges (step 04). |
| Kriging on raw PM2.5, variogram shape from log1p | Weights depend only on the variogram's shape (nugget/sill ratio, range), not its scale. Kriging in log space and back-transforming would bias predictions low. |

## Results (2020–2024, 5 × 5 split)

Overall on 95 test cells × 14,616 times (1,388,520 values), µg m⁻³:

| method | RMSE | MAE | R² | bias | (2020–21 test RMSE) |
|---|---|---|---|---|---|
| nearest | **53.7** | 24.0 | **0.61** | −7.9 | 57.5 |
| idw | 55.3 | 23.9 | 0.59 | −7.2 | 58.7 |
| kriging | 55.6 | **22.6** | 0.58 | −6.7 | 58.6 |
| linear_delaunay | 59.8 | 25.2 | 0.52 | −10.3 | 63.5 |
| linreg | 72.0 | 36.8 | 0.30 | −6.8 | 75.7 |
| train_mean | 78.8 | 40.0 | 0.16 | −0.5 | 82.4 |
| linreg_log | 119.6 | 33.6 | **−0.93** | −9.5 | 85.5 |

RMSE by region (R² in brackets for the best methods):

| method | igp | northwest | central | south | northeast | himalaya |
|---|---|---|---|---|---|---|
| nearest | **76** (0.57) | 41 | 19 | 24 | **117** (0.58) | 57 |
| idw | 78 | 37 | 22 | 24 | 132 | 40 |
| kriging | 82 (0.50) | **31** (0.44) | **16** (0.80) | 23 (0.63) | 138 (0.42) | **31** (0.70) |
| linear_delaunay | 95 | 40 | 19 | **23** | 133 | 41 |
| linreg | 101 | 44 | 30 | 44 | 170 | 37 |
| train_mean | 127 | 45 | 28 | 34 | 176 | 50 |

RMSE by season (kriging): winter 53, pre-monsoon 84, monsoon 24, post-monsoon 52.
Monsoon is easiest; pre-monsoon hardest (northeast fire season).

RMSE by distance from the test cell to the nearest training cell rises for all
interpolators: kriging 49 (≤ 100 km) → 58 (100–200) → 75 (200–300 km);
nearest 45 → 58 → 75.

**What this means**

- **Interpolation from neighbouring cells explains ~58–61% of the variance** at
  held-out blocks. This is the bar for RQ6.
- Nearest, IDW and kriging are close overall. **Kriging has the lowest MAE and the
  best RMSE in 4 of 6 regions** (central R² 0.80, Himalaya 0.70); nearest wins on
  overall RMSE mainly because of the northeast and IGP, where it copies sharp
  local peaks that smoother methods average away. Delaunay-linear is worse,
  partly because 16 test cells fall back to nearest.
- **Linear regression on the predictors is weak (R² 0.30)** and even negative R²
  in the south (−0.31) and northwest (−0.16): it predicts worse than the region's
  own mean there. It also predicts **negative PM2.5** (down to −86). The relationship
  is non-linear and season-dependent (step 04). This is what step 07 must improve.
- **The log model blows up on the full data** (R² −0.93, IGP RMSE 246): exp() of a
  linear extrapolation gives a few absurd predictions, **max 55,041 µg m⁻³** (largest
  observed: 5,608), 217 values above 2,000. On 2020–2021 this was milder (RMSE 85.5).
  Its MAE (33.6) is still better than `linreg` (36.8), i.e. it is better on typical
  values and catastrophic on a few. **Lesson for step 07:** if a log target is used,
  clip predictions to the training range and check the back-transform; tree models
  can't extrapolate this way, linear models and MLPs can.
- **The northeast dominates the errors.** The 8 northeast test cells are in
  Mizoram/Tripura (22.5–24 °N, 91.5–93.75 °E) and one in Arunachal, where the spring
  fire season gives CAMS values up to ~5,600 µg m⁻³. One bad fire day costs more
  squared error than a month of normal days elsewhere. All methods are negatively
  biased mainly because of these peaks.

## Open issues

- **Kriging variogram has no sill on training cells**: γ rises almost linearly to
  1,500 km (e.g. winter 0.16 at 75 km → 1.45 at 1,460 km), so the fitted range hits
  the fitting bound (9,361 km = 10 × max lag) in winter and post-monsoon; monsoon
  8,256 km, pre-monsoon 3,856 km (2020–2024 run).
  The resulting model is effectively linear, which is still valid for ordinary
  kriging, and predictions are reasonable. But it is not a clean fit. Options:
  fit a linear/power model explicitly, or krige residuals after removing a trend
  (that is step 08, regression kriging). ~67–70% of kriging weights are slightly
  negative (normal for ordinary kriging with many points, due to screening).
- Northeast extremes: consider also reporting metrics without the top 1% of
  values, or on log scale, so one region's fire peaks don't decide the ranking.
  Decide in step 09.
- ~~Re-run on 2020–2024~~ done 17:57.
- Raw `linreg` predicts negative PM2.5; ML models should clip at 0.
