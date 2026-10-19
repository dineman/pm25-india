# Step 09: Evaluation

**Status:** **done** 2026-10-19 (test part 23:56; CV check finished 07:32 after
the Mac slept through most of the night; the actual compute was ~25 min).
Run on the Mac. Terms in *italics* are explained in [glossary.md](glossary.md).

## Purpose

Steps 06–08 each scored their own methods on the test cells. This step:

1. puts **all methods in one set of tables** (3 interpolation baselines + Delaunay,
   linear regression, 4 ML models, 2 hybrids), with three new breakdowns:
   **hour of day**, **pollution level** and **year** (RQ1, RQ2, RQ4);
2. adds **uncertainty**: how much would the scores change with different test
   blocks? (*block bootstrap*);
3. checks the conclusions on a **much larger set of held-out places** (*CV check*),
   because the test set turned out to be only 6 blocks (RQ5, RQ6).

## Why the test set alone is not enough

The 95 test cells form only **6 blocks**, and most regions are a single block:

| Test block | Cells | Region |
|---|---|---|
| 10 | 22 | south |
| 32 | 15 | northwest |
| 35 | 25 | central |
| 38 | 7 | northeast |
| 54 | 1 | northeast |
| 50 | 25 | IGP 15, Himalaya 9, northwest 1 |

So "R² in the south" on the test set means "R² on one patch of the south".
Neighbouring cells behave alike, so 95 cells in 6 blocks carry far less
independent information than 95 scattered cells would.

## The idea, in plain words

**Block bootstrap.** Pretend we had drawn a different set of 6 test blocks:
pick 6 blocks *with replacement* from the 6 we have (some twice, some not at
all), recompute every score, repeat 2,000 times. The spread of the results
shows how much the scores depend on which blocks happened to be in the test set.
All methods are scored on the **same** resampled blocks, so we also get a range
for the *difference* between two methods (*paired comparison*), and the share
of resamples in which method A beats method B.

**CV check.** The 399 training cells were already divided into 5 *folds* of
whole blocks (step 05). Predict each fold using only the other 4 folds, exactly
as a test block is predicted from the training cells. Now every training cell
has an honest "held-out" prediction, giving **27 blocks** spread over all
regions (every region has several blocks). Bootstrap over those 27 blocks the same way.

For the hybrid, this needs care (*nested CV*): to correct fold k, we interpolate
the ML errors at the other folds' cells, and those errors must come from models
that **never saw fold k**. Otherwise fold k's true values would leak in through
the residuals. So for each fold k and each other fold j, a LightGBM model is
fitted on the remaining 3 folds and predicts fold j: 5 × 4 = **20 extra fits**.

## How it works

Code: `src/pm25/evaluation.py` (bootstrap, CV interpolation, nested residuals,
AQI levels, IST hours), `scripts/evaluate.py` (runs everything).

1. **Test part.** Load the test predictions saved by steps 06, 07 and 08, check
   they all have the same observed values, build one table (1,388,520 rows) and
   score it overall and by region, season, region × season, distance to the
   nearest training cell, **hour (IST)**, **AQI level** and **year**.
2. **Bootstrap.** Per block and method, store the sums needed for RMSE, MAE, R²
   and bias; a bootstrap sample is then just a weighted sum of blocks, so 2,000
   samples take a second.
3. **CV check.** At the 399 training cells (5,831,784 rows):
   - nearest, IDW, kriging: interpolate PM2.5 from the other 4 folds (kriging
     variograms refitted per fold on the source cells only, same recipe as step 06);
   - LightGBM: the out-of-fold predictions from step 08;
   - hybrids: out-of-fold LightGBM + nested residuals of the other 4 folds,
     kriged (variograms refitted per fold, step 08 recipe) or IDW-interpolated.
   Scored and bootstrapped like the test part.

**Hours** are given in Indian Standard Time (UTC + 5:30): the 8 time steps are
02:30, 05:30, …, 23:30 IST. **Pollution levels** use India's National AQI
breakpoints for PM2.5 (good < 30, satisfactory 30–60, moderate 60–90, poor
90–120, very poor 120–250, severe ≥ 250 µg m⁻³). These are defined on 24-hour
means; here they only serve as a familiar scale for "how polluted".

## Outputs

`results/09_evaluation/`:

| File | Content |
|---|---|
| `test_<grouping>.csv`, `cv_<grouping>.csv` | scores for `overall`, `by_region`, `by_season`, `by_region_season`, `by_distance`, `by_hour`, `by_aqi_level`, `by_year`, `by_block` |
| `test_bootstrap.csv`, `cv_bootstrap.csv` | every score with its 95% block-bootstrap range |
| `test_bootstrap_pairs.csv`, `cv_bootstrap_pairs.csv` | paired differences a − b with 95% range and share of resamples where a is better |
| `rmse_overall_ci.png`, `rmse_by_region.png`, `rmse_by_season.png`, `rmse_by_hour.png`, `bias_by_aqi_level.png` | figures, test and CV side by side |
| `nested_resid_lgbm.nc` | nested residuals (5 × time × 399 cells); **not in git**, cached for re-runs |
| `config.json` | arguments, methods, pairs |

Log: `results/logs/evaluate.log`. Tests: `tests/test_evaluation.py` (bootstrap
sums reproduce plain scores, interval contains the estimate, CV interpolation
uses only other folds, nested residuals never use the outer fold, IST and AQI
labels).

## How to run

```bash
conda run -n pm25 python scripts/evaluate.py            # ~25 min (20 nested fits), cached afterwards
conda run -n pm25 python scripts/evaluate.py --skip-cv  # test part only, ~1 min
```

Needs the prediction files of steps 06–08 (`predictions_test.nc`,
`oof_train_lgbm.nc`), which are not in git: re-run those steps on a new machine.

## Decisions

| Decision | Reason |
|---|---|
| Core + CV check; no 8 × 8 split re-run (user, 2026-10-18) | The CV check addresses the "6 blocks" weakness directly; 8 × 8 would add another small test set. |
| Bootstrap unit = block, 2,000 samples, 95% percentile ranges | Blocks are the independent units of the split; resampling rows would ignore spatial correlation and give far too narrow ranges. |
| Paired differences | Methods share the same blocks; differences between methods vary much less than the scores themselves. |
| Nested residuals for the CV hybrid | Without nesting, the residuals at other folds come from models trained on fold k, so fold k's own PM2.5 would leak into its correction. |
| Kriging variograms refitted per CV fold | Same reason: fitting on all 399 cells would use the predicted fold. |
| Re-run steps 06–08 on the Mac first, with the IDW tie fix | The prediction files aren't in git; all tables must come from the same predictions. See "Problems met". |

## Problems met

- **IDW gave different results on Fedora and on the Mac** although it has no
  randomness. For 23 of 95 test cells the 8th and 9th nearest training cells are
  exactly equally far (regular grid), and `np.argsort` broke the tie differently.
  Fixed with a stable sort (ties go to the lower cell index) + test. IDW test
  RMSE 55.31 → 55.14; the IDW hybrid 44.98 → 45.00.
- **MLP is not reproducible across machines**: test R² 0.545 (Fedora) vs 0.528
  (Mac), same seed. Neural-network training depends on the numeric libraries.
  LightGBM changed only in the 2nd decimal (RMSE 54.97 → 54.95); ridge, RF,
  nearest, kriging identical. All numbers in this step are from the Mac run.
- **The Mac slept during the run** (on battery), stretching ~25 min of work over
  8 hours. Use `caffeinate` and mains power for long runs.

## Results

### Overall, with 95% block-bootstrap ranges

Test cells (6 blocks):

| method | RMSE | 95% range | MAE | R² | R² range | bias |
|---|---|---|---|---|---|---|
| **lgbm_rk** (LightGBM + kriged residuals) | 45.9 | 17.7–81.5 | **17.4** | 0.72 | 0.54–0.83 | −1.3 |
| lgbm_idw (LightGBM + IDW residuals) | **45.0** | 21.1–77.2 | 18.8 | **0.73** | 0.57–0.83 | −1.6 |
| nearest | 53.7 | 25.2–84.1 | 24.0 | 0.61 | 0.35–0.67 | −7.9 |
| lgbm | 55.0 | 19.5–98.2 | 19.1 | 0.59 | 0.36–0.75 | −7.9 |
| idw | 55.1 | 25.3–92.6 | 23.8 | 0.59 | 0.39–0.69 | −7.2 |
| kriging | 55.6 | 21.6–96.1 | 22.6 | 0.58 | 0.42–0.71 | −6.7 |
| mlp | 59.2 | 22.7–105.1 | 21.8 | 0.53 | 0.25–0.71 | −11.0 |
| linear (Delaunay) | 59.8 | 24.6–95.1 | 25.2 | 0.52 | 0.34–0.61 | −10.3 |
| rf | 62.7 | 21.9–111.7 | 22.6 | 0.47 | 0.19–0.67 | −11.3 |
| ridge | 71.3 | 36.1–119.1 | 35.1 | 0.31 | −0.06–0.48 | −5.1 |
| linreg | 72.0 | 37.2–119.3 | 36.8 | 0.30 | −0.18–0.46 | −6.8 |

CV check, training cells (27 blocks):

| method | RMSE | 95% range | MAE | R² | R² range | bias |
|---|---|---|---|---|---|---|
| **lgbm_rk** | **41.1** | 29.5–52.7 | **19.2** | **0.68** | 0.52–0.80 | −0.4 |
| lgbm | 41.9 | 30.4–53.1 | 19.4 | 0.66 | 0.52–0.78 | −6.3 |
| kriging | 42.0 | 31.2–52.8 | 21.1 | 0.66 | 0.53–0.76 | −0.5 |
| lgbm_idw | 45.3 | 32.9–57.5 | 21.8 | 0.61 | 0.42–0.75 | −1.3 |
| idw | 45.8 | 36.1–55.1 | 24.5 | 0.60 | 0.41–0.70 | +2.4 |
| nearest | 47.8 | 38.7–57.1 | 24.5 | 0.56 | 0.37–0.67 | +0.8 |

The test ranges are very wide (RMSE roughly 20 to 100 for every method): with 6
blocks, one block (northeast block 38, RMSE ~130–160) decides much of the score.

### Paired comparisons (RMSE difference a − b, 95% range, share of resamples a is better)

| a vs b | test (6 blocks) | CV (27 blocks) |
|---|---|---|
| lgbm_rk vs lgbm | −9.0 (−15.7 to −1.1), 100% | −0.8 (−3.5 to +1.7), 71% |
| lgbm_rk vs kriging | −9.6 (−15.4 to −3.4), 100% | −0.9 (−6.1 to +3.3), 65% |
| lgbm_rk vs nearest | −7.8 (−18.0 to +2.0), 92% | −6.7 (−14.3 to +0.0), 97% |
| lgbm_rk vs lgbm_idw | +0.9 (−3.6 to +4.4), 33% | **−4.2 (−5.7 to −2.7), 100%** |
| lgbm vs kriging | −0.6 (−5.8 to +5.1), 60% | −0.1 (−3.8 to +2.5), 53% |
| lgbm vs nearest | +1.2 (−10.6 to +15.8), 45% | −5.9 (−14.0 to +0.7), 96% |
| lgbm vs mlp | −4.2 (−7.6 to −2.3), 100% | — |
| lgbm vs rf | −7.8 (−12.4 to −2.4), 100% | — |

In MAE (less driven by the rare extremes) LightGBM does better against
interpolation: it beats nearest on both sets (test −5.0, CV −5.1, both ranges
below 0) and kriging on the test set (−3.5, range −7.4 to −0.8); in the CV check
the MAE gain over kriging (−1.7) is not certain (range −4.9 to +0.7, 89%).

### Why the hybrid's gain differs between test and CV

RMSE change from adding kriged residuals (lgbm → lgbm_rk), block by block:

- **Test:** +25.1 in northeast block 38 and +12.1 in IGP block 50; between −2.7
  and +3.4 in the other four. The big test gain comes from **2 of 6 blocks**.
- **CV:** better in 15 of 27 blocks (up to +14.9, IGP block 44; +12.1 northwest
  block 42), worse in 12 (down to −19.7, northwest block 40; −8.6 northeast
  block 55; −8.1 and −6.9 in two IGP blocks).

The correction helps when the ML errors at the neighbouring blocks have the same
sign and size as at the predicted block (a regional emission pattern the
predictors miss) and hurts when the neighbours' errors don't carry over
(different local sources across ~300–500 km). On average over many places it
helps a little; on the particular test blocks it happened to help a lot.

### By region (R²)

| method | igp | northwest | central | south | northeast | himalaya |
|---|---|---|---|---|---|---|
| test lgbm_rk | 0.74 | 0.68 | 0.86 | 0.69 | 0.53 | 0.86 |
| test lgbm | 0.59 | 0.67 | 0.78 | 0.67 | 0.34 | 0.80 |
| test kriging | 0.50 | 0.44 | 0.80 | 0.63 | 0.42 | 0.70 |
| CV lgbm_rk | 0.62 | 0.61 | 0.79 | 0.53 | 0.24 | 0.80 |
| CV lgbm | 0.58 | 0.60 | 0.75 | 0.57 | 0.26 | **0.81** |
| CV kriging | 0.62 | **0.65** | 0.74 | **0.62** | 0.23 | **−0.05** |
| CV nearest | 0.63 | 0.47 | 0.58 | 0.48 | −0.06 | −1.23 |

- **Himalaya: ML is essential.** Interpolation fails completely in the CV check
  (kriging R² −0.05, nearest −1.23; RMSE 42 vs 18 for LightGBM): PM2.5 drops
  steeply with altitude, and the nearest training cells are often in a valley or
  on the plain. LightGBM knows the altitude (surface pressure) and lat/lon.
- **Northeast is the hardest region everywhere** (R² 0.23–0.53): pre-monsoon
  fire plumes are local and not described by the predictors.
- **IGP, northwest, south:** interpolation is as good as or better than ML in the
  CV check. These regions have dense, smooth pollution fields where neighbours
  are informative.

### By season (R², CV / test)

| method | winter | pre-monsoon | monsoon | post-monsoon |
|---|---|---|---|---|
| lgbm_rk | 0.66 / 0.78 | 0.47 / 0.59 | 0.77 / 0.81 | 0.70 / 0.80 |
| lgbm | 0.64 / 0.72 | 0.44 / 0.39 | 0.74 / 0.72 | 0.70 / 0.74 |
| kriging | 0.63 / 0.61 | 0.48 / 0.47 | 0.76 / 0.67 | 0.68 / 0.65 |

Pre-monsoon (fire season) is the hardest season for every method in both sets.

### By hour of day (RMSE, CV)

Errors follow the daily cycle of PM2.5 itself: **lowest in the afternoon**
(14:30 IST: lgbm_rk 11.6, kriging 16.6) when the boundary layer is deep and
pollution is mixed upwards, **highest at night and early morning** (02:30: ~58
for all methods) when pollution is trapped near the ground and varies sharply
between places. ML and hybrids gain most in the daytime; at night all methods are
about equal. On the test set the hybrid is best at every hour.

### By pollution level (bias)

All methods **under-predict the highest values**: at "severe" levels (≥ 250
µg m⁻³) the bias is −116 (kriging), −121 (LightGBM) and **−62 (hybrid)** in the
CV check (test: −164, −148, −70). The hybrid halves the under-prediction of
peaks, its clearest and most consistent advantage. At "good" levels (< 30) all
methods slightly over-predict (+3 to +13). Severe values are rare (~3% of rows)
but dominate RMSE.

### By year

No trend: every method's ranking is the same in all 5 years. 2021 has the largest
errors in both sets, 2024 the smallest. LightGBM's low bias is present in every year (CV −4 to −8).

### By distance to the nearest training cell (RMSE, CV)

| method | ≤ 100 km | 100–200 km | 200–300 km | > 300 km |
|---|---|---|---|---|
| lgbm_rk | 29.4 | **40.8** | **49.7** | 105.9 |
| lgbm | 30.1 | 41.9 | 50.6 | 106.9 |
| kriging | **25.7** | 41.9 | 56.4 | 102.8 |
| nearest | 29.4 | 47.3 | 67.5 | **98.5** |

Interpolation wins close to known cells (≤ 100 km), ML and the hybrid win at
100–300 km. The > 300 km cells (only in the CV check) are isolated northeast
cells where everything fails.

## What this means for the research questions

- **RQ1 (how accurately?)** At held-out places, the best method explains about
  **two thirds of the variance** (CV R² 0.68, RMSE ~41 µg m⁻³, MAE ~19), with a
  realistic range of roughly 0.5–0.8 depending on which places are tested. The
  step 08 headline (0.72) is at the upper end of that range.
- **RQ2 (which ML model?)** **LightGBM** is clearly best among the ML models
  (beats MLP and RF in 100% of bootstrap samples). Linear models are far behind
  (R² ~0.3), so the relationships are strongly non-linear.
- **RQ4 (regions and seasons)** Best in central India and the monsoon, worst in
  the northeast and pre-monsoon (fires), and at night. Errors are concentrated in
  the rare severe-pollution hours.
- **RQ5 (new locations)** The ranking of methods at new places depends a lot on
  *which* places: test and CV agree on the direction of most comparisons but not
  on the size of the gains. Results from one test block per region should not be
  generalised.
- **RQ6 (ML vs interpolation)** Robust findings, true on both sets:
  (1) ML beats interpolation where neighbours aren't representative: the
  **Himalaya** and **100–300 km** from known cells; (2) the **hybrid removes
  LightGBM's systematic low bias** (−6.3 → −0.4) and **halves the
  under-prediction of severe pollution**; (3) kriging of residuals is better than
  IDW of residuals (CV, 100%). Not robust: the size of the hybrid's overall RMSE
  gain. Over 27 blocks, hybrid, LightGBM and kriging are within ~1 µg m⁻³ of each
  other in RMSE.

## Open issues

- The CV check slightly favours ML: LightGBM's settings were tuned with these
  same 5 folds (step 07), and its out-of-fold models saw 4/5 of the training
  cells versus 3/5 for the nested residual models. Mention in the write-up.
- Step 07 feature-set variants (`brief`, `all_no_latlon`) were not re-run on the
  Mac; their tables still show the pre-fix IDW numbers from Fedora.
- Kashmir boundary: user decided to use India's official boundary; open whether
  only for the maps (step 11) or also for the region mask (re-run steps 02–09).
