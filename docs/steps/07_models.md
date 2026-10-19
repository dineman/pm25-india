# Step 07: ML models

**Status:** main run **done** (tuning 19:53–20:29; corrected final fits 22:44–22:51
after a bug fix, see "Problems met"); feature-set variants done 22:52–23:07. Terms in *italics* are explained in
[glossary.md](glossary.md).

> **Re-run on the Mac, 2026-10-19 (step 09 uses these numbers):** Final fits re-run with `--cv-from` (same settings). Ridge and RF identical; LightGBM RMSE 54.97 → **54.95**; **MLP R² 0.545 → 0.528** (RMSE 58.1 → 59.2): neural-network training isn't bit-reproducible across machines. Rankings unchanged. The `brief`/`all_no_latlon` variants were not re-run.

## Purpose

Answer RQ1 ("how accurately can PM2.5 be estimated from CAMS atmospheric and
aerosol variables?"), RQ2 ("how do different ML models compare?") and RQ5 ("how
well does it work at grid locations not used in training?").

The task: for a grid cell s and 3-hourly time t, predict PM2.5(s, t) **only from
the predictors at the same s and t** (AOD, wind, temperature, dewpoint, pressure,
derived humidity and time features, lat/lon). The model never sees PM2.5 at
neighbouring cells; that is what interpolation does (step 06), and the two are
combined in step 08.

Step 06 showed that a straight-line model (linear regression) does this badly
(R² 0.30): the effects are non-linear and change with season (e.g. humidity only
matters in winter, temperature flips sign in the monsoon, step 04). ML models
can learn such patterns.

## The four models, in plain words

| Model | How it predicts | Why it is in the comparison |
|---|---|---|
| **Ridge** | A weighted sum of the features (a straight-line model), with a penalty that keeps weights small. | Linear reference: the best a straight-line model can do with tuning. |
| **Random forest (RF)** | Many *decision trees*, each grown on a random sample of rows and features; the prediction is their average. | Robust, classic, few settings to tune. Can't extrapolate beyond training values. |
| **LightGBM** | *Gradient boosting*: trees built one after another, each fixing the errors left by the previous ones. | Usually the strongest model for tables of numbers; fast on millions of rows. |
| **MLP** | A small neural network (2–3 layers of 64–128 units) on scaled features. | A different family (smooth function instead of steps); needed for RQ2. |

All four get **exactly the same features, training cells, CV folds and tuning
budget** (8 candidate settings each), so differences come from the models, not
from unequal effort.

## How it works

Code: `src/pm25/features.py` (feature tables), `src/pm25/models.py` (models,
search spaces, CV), `scripts/train_models.py` (runs everything).

### 1. Build the tables

- **Training rows:** every (time, cell) for the 399 training cells of the 5 × 5
  split (step 05): 399 × 14,616 = **5,831,784 rows**.
- **Test rows:** the 95 test cells: 95 × 14,616 = 1,388,520 rows. Not touched
  until step 4 below.
- **Features ("all" set, 14):** aod550, u10, v10, t2m, d2m, sp, lat, lon (the
  brief's list) + wind_speed, rh2m, hour_sin, hour_cos, doy_sin, doy_cos
  (derived in step 03). Stored as float32 (~400 MB in memory).

### 2. Tune each model with block-wise cross-validation

A model's *hyperparameters* (e.g. number of trees, network size) must be chosen
without looking at the test cells, otherwise the test score is no longer an
honest measure (*data leakage*). So:

1. For each model, draw **8 random settings** from its search space (seeded).
   The search space also includes **target = raw or log**: fit on PM2.5 itself
   or on log(1 + PM2.5).
2. Score each setting by **5-fold block-wise CV** on training cells only: the 5
   folds are the groups of whole training blocks from step 05. Fit on 4 folds,
   predict the 5th, repeat 5 times, average the RMSE.
3. Keep the setting with the lowest mean CV RMSE.

To keep this affordable, CV uses a **random 5% of time steps** (730 of 14,616,
all 399 training cells at those times ≈ 290,000 rows). That is plenty to rank
settings; the final model then uses far more data.

Search spaces (`SEARCH_SPACES` in `models.py`):

| Model | Settings searched |
|---|---|
| ridge | alpha (penalty strength) 0.01–1000 |
| rf | trees 100/200; max depth none/20/30; min rows per leaf 5/20/50; features per split 33/50/100%; rows per tree 10/25% |
| lgbm | trees 300/600/1000; learning rate 0.03/0.05/0.1; leaves per tree 31–255; min rows per leaf 20/100/500; row/column subsampling 70/100%; L2 penalty 0/1/10 |
| mlp | layers (64,64)/(128,64)/(128,128,64); L2 penalty 1e-5–1e-3; learning rate 0.001/0.003; batch 2,048; early stopping, max 50 passes |

### 3. Fit the final model

Refit the best setting on the training cells: **all 5.8 M rows** for ridge and
LightGBM; a random **2 M rows** for RF and MLP (on all rows they are too slow and,
for RF, too memory-hungry for this 15 GB machine).

### 4. Score once on the test cells

Predict the 95 test cells for all 14,616 time steps and compute RMSE, MAE, R²
and bias overall, by region, season, region × season and distance to the nearest
training cell, with **the same code as the baselines** (`pm25.metrics`), so the
numbers are directly comparable. The step 06 baselines (nearest, IDW, kriging,
linear regression) are put in the same tables.

### Safety net: clipping

Every model's predictions are clipped to **[0, highest PM2.5 in training]**.
Step 06 showed why: a linear model on log(PM2.5) predicted 55,041 µg m⁻³ once
exp() was applied, and raw linear regression predicted negative PM2.5.

## Outputs

`results/07_models/<feature set>/`:

| File | Content |
|---|---|
| `cv_results.csv` | every setting tried: model, settings, CV RMSE (mean ± std over folds), MAE, R², bias, seconds |
| `best_params.json` | chosen setting per model |
| `metrics_overall.csv`, `metrics_by_region.csv`, `metrics_by_season.csv`, `metrics_by_region_season.csv`, `metrics_by_distance.csv` | test metrics, models + baselines |
| `predictions_test.nc` | test predictions (time, cell) per model + observed |
| `config.json` | all arguments, features, row counts, fit/predict times |
| `model_<name>.joblib` | fitted models (**not in git**: model weights) |

Log: `results/logs/train_models.log`. Tests: `tests/test_models.py` (clipping,
log back-transform, seeded distinct settings, folds, every model fits and stays
in range, CV output), `tests/test_features.py` (row order, broadcasting).

## How to run

```bash
# main run: 4 models, tuned, feature set "all"
conda run -n pm25 python scripts/train_models.py --n-iter 8 --cv-time-frac 0.05
# feature-set variants, reusing the tuned settings
conda run -n pm25 python scripts/train_models.py --feature-set brief \
    --params-from results/07_models/all/best_params.json
```

## Decisions

| Decision | Reason |
|---|---|
| Same budget (8 settings) for every model | Fair comparison for RQ2. |
| Block-wise CV on training cells only | Choosing settings on test cells would leak; random-row CV would be over-optimistic for the same reason as in step 05. |
| CV on 5% of time steps | Pilot: one RF setting takes 30 s and one MLP setting 62 s on 2%; 5% keeps the whole run to ~40 min. Ranking settings doesn't need every row. |
| Raw vs log target tuned, not fixed | Step 06: log helped typical values (lower MAE) but hurt extremes; let CV decide per model. |
| Clip predictions to [0, training max] | Step 06 blow-up and negative predictions. |
| RF and MLP final fit on 2 M rows | Memory (RF trees grow with rows) and time on this machine. Recorded in `config.json`. |
| Hyperparameters chosen on raw-scale RMSE | Same metric as reported; puts weight on getting high-pollution episodes right. |

## Pilot run (19:47, kept only in the log)

One random setting per model, CV on 2% of times, final RF/MLP on 200,000 rows,
written to a scratch folder. Purpose: check the pipeline end to end and time
each model before choosing the budget.

| model | CV time per setting | final fit | test RMSE (untuned) |
|---|---|---|---|
| ridge | < 1 s | 3 s (5.8 M rows) | 72.2 |
| rf | 30 s | 21 s (200 k rows) | — |
| lgbm | 7 s | 29 s (5.8 M rows) | — |
| mlp | 62 s | 27 s (200 k rows) | 62.4 |

Peak memory 3.6 GB, total 200 s. CV RMSE varied a lot between folds (± 17–19
µg m⁻³): folds hold different regions, and one with northeast fire cells is much
harder. Untuned LightGBM already beat linear regression in every region.

## Problems met (and fixed)

1. **Wrong "best" settings picked (bug).** Each model's 8 CV results were put in a
   table with row numbers 0–7, then the 4 tables were joined. The row numbers
   repeated (four rows labelled "0", …), and the lookup "row with the lowest
   RMSE per model" used those labels, so it took rows belonging to *other*
   models. Example: LightGBM was refitted with a setting scoring CV RMSE 40.9
   instead of its best (36.7); MLP got (128,128,64) instead of (128,64).
   The CV scores were right; only the choice was wrong. Found at 22:41 by
   comparing `best_params.json` with the tuning log.
   **Fix:** a small function `best_settings()` in `models.py` that picks the best
   row by position within each model, with a regression test for exactly this
   case; tables are now joined with fresh row numbers (`ignore_index=True`).
2. **Final MLP fit crawled for over 2 hours** (expected ~10 min). The machine's
   15 GB RAM was full (Firefox alone ~6.5 GB) and the 8 GB swap too, with 1.1 GB
   of the training process on disk. Stopped at 22:42 (its settings were wrong
   anyway). **Fix:** the script now frees the gridded data copies once the
   numpy tables are built, and has `--cv-from` to reuse tuning results instead
   of re-tuning. The corrected rerun took 7.5 min in total (MLP fit 4 min).
3. Final-fit row sampling for RF/MLP now uses its own seeded random generator
   (seed + 1), so the same rows are used whether tuning ran in the same run or
   was reused.

## Results (2020–2024, feature set "all", 5 × 5 split)

### Tuning (block-wise CV on training cells, 5% of time steps = 582,939 rows)

| model | best CV RMSE (± std over folds) | CV MAE | CV R² | chosen setting |
|---|---|---|---|---|
| **lgbm** | **36.7** (± 13.2) | **17.3** | **0.69** | log target, 1,000 trees, learning rate 0.1, 255 leaves, ≥ 100 rows per leaf, 70% rows/columns per tree, L2 10 |
| mlp | 37.9 (± 11.9) | 19.0 | 0.67 | log target, layers (128, 64), L2 1e-4, learning rate 0.003 |
| rf | 40.8 (± 15.3) | 20.5 | 0.63 | log target, 100 trees, no depth limit, ≥ 20 rows per leaf, 50% features per split, 25% rows per tree |
| ridge | 47.8 (± 14.8) | 29.6 | 0.48 | raw target (alpha makes no difference) |

- **Log target won for all three non-linear models**; for ridge, log was much
  worse (CV R² −0.10 for every alpha: a straight line in log space can't fit).
- The ± is large because folds differ a lot (one contains northeast fire cells).
  The gap lgbm → mlp (1.2) is smaller than that spread, so lgbm and mlp are close;
  rf and ridge are clearly behind.
- All 32 tried settings: `results/07_models/all/cv_results.csv`.

Final fits: ridge 3 s and lgbm 101 s on all 5.8 M rows; rf 84 s and mlp 235 s on 2 M rows.

### Test cells (95 cells × 14,616 times = 1,388,520 values; scored once)

| method | RMSE | MAE | R² | bias |
|---|---|---|---|---|
| **lgbm** | 55.0 | **19.1** | 0.59 | −7.9 |
| mlp | 58.1 | 21.2 | 0.55 | −10.4 |
| rf | 62.7 | 22.6 | 0.47 | −11.3 |
| ridge | 71.3 | 35.1 | 0.31 | −5.1 |
| *nearest (step 06)* | *53.7* | *24.0* | *0.61* | *−7.9* |
| *idw* | *55.3* | *23.9* | *0.59* | *−7.2* |
| *kriging* | *55.6* | *22.6* | *0.58* | *−6.7* |
| *linreg* | *72.0* | *36.8* | *0.30* | *−6.8* |

RMSE by region (best in bold):

| method | igp | northwest | central | south | northeast | himalaya |
|---|---|---|---|---|---|---|
| lgbm | 73.9 | **23.8** | 16.3 | **22.1** | 147.1 | 24.9 |
| mlp | **73.1** | 26.5 | 18.6 | 27.0 | 157.9 | **24.2** |
| rf | 87.3 | 27.8 | 17.7 | 25.3 | 164.7 | 29.9 |
| ridge | 100.6 | 44.3 | 30.1 | 39.9 | 170.2 | 33.6 |
| *nearest* | 75.8 | 41.1 | 19.1 | 24.2 | **117.3** | 56.7 |
| *kriging* | 81.8 | 30.7 | **15.8** | 23.3 | 137.5 | 30.9 |

R² by region, lgbm vs kriging: central 0.79 vs 0.80, Himalaya 0.80 vs 0.70,
IGP 0.59 vs 0.50, northeast 0.33 vs 0.42, northwest 0.67 vs 0.44, south 0.67 vs 0.63.

RMSE by season: lgbm is best in winter (45.2), monsoon (21.6, tied with mlp) and
post-monsoon (44.7); in the **pre-monsoon** the interpolators win (nearest 76.1 vs
lgbm 89.8) because of the northeast fire season.

RMSE by distance from the test cell to the nearest training cell:

| method | ≤ 100 km | 100–200 km | 200–300 km |
|---|---|---|---|
| lgbm | 54.8 | **51.5** | **68.1** |
| mlp | 57.1 | 55.9 | 71.0 |
| nearest | **44.8** | 58.3 | 74.7 |
| kriging | 49.3 | 57.9 | 74.8 |

### What this means

- **RQ1 (how accurately?)** From the predictors alone, at grid cells never seen in
  training, the best model (LightGBM) explains **59% of the variance**, with a
  typical error (MAE) of **19 µg m⁻³**. A straight-line model only reaches 31%.
- **RQ2 (which model?)** LightGBM > MLP > random forest >> ridge, both in CV and on
  test, with the same ranking. LightGBM is also the fastest of the non-linear
  models. All non-linear models preferred the log target.
- **ML vs interpolation (preview of RQ6).** Overall they are close on R² (lgbm
  0.59, nearest 0.61), but they win in different places:
  - **LightGBM has the lowest MAE of everything (19.1 vs 22.6–24.0)**, i.e. it is
    better on typical values. Interpolation has slightly lower RMSE, which is
    driven by the northeast fire peaks.
  - **The further a test cell is from training cells, the more ML wins**: within
    100 km nearest-neighbour copying is better (44.8 vs 54.8), but at 100–200 km
    LightGBM is better than every interpolator (51.5 vs ≥ 57.9) and at 200–300 km
    too (68.1 vs ≥ 73.4). ML learns *relationships* that transfer to new places;
    interpolation only copies neighbours.
  - LightGBM is clearly better in the **northwest** (RMSE 23.8 vs 30.7) and
    **Himalaya** (24.9 vs 30.9), where conditions change sharply over short distances
    (desert vs coast, mountains vs foothills) and neighbours are poor guides.
  - Interpolation is better in the **northeast** (nearest 117 vs lgbm 147): the
    fire plumes are local events the predictors don't fully describe, while the
    neighbouring cells do see them.
  - This strongly suggests **combining** the two (step 08: ML prediction + kriging
    of its residuals).
- **RQ5 (new locations).** CV R² 0.69 vs test R² 0.59: the test blocks are harder
  than the average CV fold (they include the northeast fire cells and the Punjab
  foothills block). Error grows with distance from training cells for all models,
  but more slowly for ML than for interpolation.
- **Bias:** all models predict too low on average, most in the IGP (lgbm −27):
  the highest winter peaks are underpredicted, which is typical when a model
  averages over similar-looking conditions.

### Feature-set variants (same tuned settings, same split)

Run with `--feature-set brief` and `--feature-set all_no_latlon`, reusing the
settings tuned on `all` (`--cv-from results/07_models/all/cv_results.csv`).

| features | n | lgbm R² | lgbm RMSE | lgbm MAE | mlp R² | rf R² | ridge R² |
|---|---|---|---|---|---|---|---|
| **all** (brief + derived) | 14 | **0.59** | **55.0** | **19.1** | 0.54 | 0.47 | 0.31 |
| all without lat/lon | 12 | 0.48 | 62.4 | 23.5 | 0.40 | 0.39 | 0.31 |
| brief (aod, u10, v10, t2m, d2m, sp, lat, lon) | 8 | 0.44 | 64.6 | 23.7 | 0.45 | 0.41 | 0.29 |

LightGBM RMSE by region:

| features | igp | northwest | central | south | northeast | himalaya |
|---|---|---|---|---|---|---|
| all | **73.9** | **23.8** | **16.3** | **22.1** | **147.1** | **24.9** |
| all without lat/lon | 90.9 | 26.7 | 19.0 | 24.9 | 158.6 | 34.1 |
| brief | 88.1 | 27.9 | 19.7 | 23.8 | 170.5 | 37.5 |

LightGBM RMSE by distance to training: all 54.8 / 51.5 / 68.1; without lat/lon
62.1 / 59.0 / 76.1; brief 64.2 / 61.0 / 79.5 (≤ 100 / 100–200 / 200–300 km).

**What this means**

- **lat/lon help on unseen blocks** (R² 0.59 → 0.48 without them, worse in every
  region). The worry was that coordinates let a model *memorise* training
  locations; that would help on training cells but not on held-out blocks. Here
  they help on held-out blocks, so the model uses them for **smooth large-scale
  patterns the meteorology doesn't explain** (e.g. where emissions are high: the
  IGP, RMSE 74 → 91 without them). With 5 × 5 blocks this is a fair gain; it
  would not transfer to a region outside India.
- **The derived features matter as much** (all → brief: R² 0.59 → 0.44). Mostly
  the **time features**: the strong daily cycle (step 04) can't be seen from
  the brief's predictors alone; plus RH and wind speed.
- Ridge barely changes (0.29–0.31): a straight line can't use these features well.
- The ranking LightGBM > MLP > RF > ridge holds for `all` and `all_no_latlon`; with
  `brief`, MLP ≈ LightGBM.
- **Main feature set for later steps: `all`.**

## Open issues

- Variants reused the settings tuned on `all`; tuning each feature set separately might narrow the gaps slightly.
- CV used 5% of time steps; a larger fraction might pick slightly different settings.
- Northeast fire extremes dominate RMSE (step 06); look at MAE and per-region
  results too.
