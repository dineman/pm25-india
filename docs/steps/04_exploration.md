# Step 04: Exploration and variogram

**Status:** done on the **full 2020–2024 data** (14,616 time steps, 494 India
cells; run 2026-10-18 17:57, 33 s, 3.3 GB). A test run on 2020–2021 came first
(16:47); its numbers were within a few percent and are noted where useful. Block
size **decided: 5 × 5 cells** (main) + 8 × 8 (sensitivity), see the end of this
doc and 05_splits.md.

## Purpose

1. Describe the target before modelling: where and when PM2.5 is high, its
   seasonal and daily cycles per region, and how each predictor relates to it.
2. Measure **how far spatial correlation reaches** (the variogram). That
   distance sets the block size for the spatial train/test split (step 05):
   test cells should not be close copies of training cells, or scores are
   inflated (CLAUDE.md rule).

## How it works

Code: `src/pm25/exploration.py` (tested functions) and `scripts/explore_eac4.py`
(runs everything, writes tables and figures). Only India cells (`in_india`) are used.

1. **Flatten** India cells to a table: one row per (time, cell), with region and
   season. 2020–2024: 7,220,304 rows.
2. **Summaries** of PM2.5 (mean, median, std, 5th/95th percentile) by region,
   season, region × season and region × season × IST hour.
3. **Spearman rank correlation** of each predictor with PM2.5, overall and per
   season. Rank correlation because PM2.5 is very skewed. 300,000 random rows
   per group (seed 42).
4. **Variograms.** For two cells at distance h, semivariance
   γ(h) = ½ · mean[(z(s) − z(s+h))²]. Low γ = similar values.
   - **2,000 random time steps** (seed 42). For each, every pair of the 494 cells
     (121,771 pairs) is put into a **75 km distance bin** (up to 1,500 km);
     distances are great-circle (haversine).
   - Each time step's field is **standardised** (mean 0, std 1 across cells)
     first, so winter days with huge values don't dominate and the sill is ≈ 1.
   - Three fields, each on log(1 + PM2.5) (the log tames extremes):
     1. `log_pm25`: PM2.5 itself.
     2. `residual`: what's left after a linear regression on all predictors
        (log AOD, u10, v10, wind speed, t2m, d2m, RH, sp, hour and day-of-year sin/cos).
     3. `residual_trend`: same, plus a **quadratic lat/lon trend**. This removes
        the country-wide north–south gradient (IGP high, south low) and leaves
        only local correlation. **This is the one that matters for the block
        size**, because the models will use lat/lon and the predictors, so what
        leaks between neighbouring cells is this local part. (Standard practice
        for block CV: Roberts et al. 2017, *Ecography*; Valavi et al. 2019, blockCV.)
   - Fit **spherical** and **exponential** models (nugget + partial sill + range)
     up to 1,000 km, weighted by pair counts, and keep the better fit (lower RMSE).
     For the exponential model the range is the **practical range** (95% of sill).
5. **Block size** = range ÷ cell width (75.6 km for 0.75° at 25 °N), rounded up.

## Outputs

`results/04_exploration/` (full data; the 2020–2021 test run is in `results/04_exploration_2020_2021/`):

| File | Content |
|---|---|
| `summary_region.csv`, `summary_season.csv`, `summary_region_season.csv`, `summary_region_season_hour.csv` | PM2.5 statistics |
| `spearman_pm25_by_season.csv`, `spearman_heatmap.png` | predictor–PM2.5 rank correlations |
| `seasonal_mean_maps.png` | mean PM2.5 map per IMD season (log colour scale) |
| `monthly_cycle_by_region.png` | mean PM2.5 by month, per region |
| `diurnal_cycle_by_region.png` | mean PM2.5 by IST hour, per region, per season |
| `variogram_bins.csv`, `variogram_fits.csv`, `variograms.png` | variograms (3 fields × all/4 seasons) and all fits |
| `summary.json` | data used, seed, chosen fit, range and block size per field |

Log: `results/logs/explore_eac4.log`. Tests: `tests/test_exploration.py`
(haversine, variogram of a constant field, NaN handling, fit recovers a known
range, standardising, residuals, block size, Spearman, table flattening).

## How to run

```bash
# full data (default paths) — what the results below come from
conda run -n pm25 python scripts/explore_eac4.py
# 2020-2021 test run
conda run -n pm25 python scripts/explore_eac4.py \
    --data data/processed/eac4_india_3h_2020_2021.nc --out-dir results/04_exploration_2020_2021
```

Runtime: 33 s and 3.3 GB peak memory for 5 years (25 s for 2 years).

## Decisions

| Decision | Reason |
|---|---|
| log(1 + PM2.5) for variograms | Squared differences of raw values are dominated by a few extreme cells (IGP winter, NE fire season). |
| Standardise each time step | Pools time steps of very different variance; makes the sill ≈ 1 and the range comparable between seasons. |
| Random 2,000 time steps, not all | 2,000 × 121,771 pairs is plenty for stable bins; random (not every n-th) so all 8 hours are sampled evenly. |
| Block size from the **detrended residual** variogram | It measures what a model with predictors + lat/lon cannot explain, which is what leaks between neighbouring train/test cells. |
| Fit only up to 1,000 km | Long lags have few pairs and are dominated by the large-scale gradient. |

## Results (2020–2024)

**PM2.5 levels.** India mean 72.2, median 51.9, 95th percentile 207 µg m⁻³
(2020–2021 test: 71.0 / 50.1 / 205). Mean by region × season (µg m⁻³):

| region | winter | pre-monsoon | monsoon | post-monsoon |
|---|---|---|---|---|
| igp | **217.9** | 116.5 | 77.3 | **207.1** |
| northwest | 85.1 | 69.2 | 67.4 | 84.9 |
| central | 84.7 | 55.4 | 37.5 | 80.7 |
| northeast | 68.4 | **82.3** | 22.5 | 43.9 |
| south | 63.4 | 49.9 | 30.2 | 62.2 |
| himalaya | 51.8 | 37.4 | 29.8 | 46.8 |

- **IGP is 2–4× higher than other regions** in winter and post-monsoon (monthly
  mean 239 in Jan, 241 in Dec; 64 in Jul). Lowest everywhere in the monsoon
  (rain washout).
- **Northeast peaks in March** (monthly mean 139 µg m⁻³, double Feb/Apr): fire
  season. No other region does this.
- **Northwest stays high in the monsoon** (Jun 86 µg m⁻³, close to its winter
  level), probably Thar desert dust.
- **Daily cycle:** highest at night/early morning (02:30–05:30 IST), lowest at
  14:30 IST, everywhere and in every season. Strongest in IGP winter: 279 at
  02:30 vs 113 at 14:30 IST. This is the boundary layer: shallow at night traps
  PM2.5, deep at midday dilutes it. Confirms the 3-hourly data and hour features
  are worth keeping.

**Correlations with PM2.5 (Spearman, all seasons; range across seasons):**

- **AOD 0.60** (0.52–0.77): by far the strongest single predictor. Remember both
  come from the same CAMS aerosol fields (limitation in 00_overview).
- **Surface pressure 0.43** (0.33–0.47): mostly a proxy for **altitude** (low
  plains = high PM2.5, Himalaya/plateau = low).
- **RH: 0.04 overall but 0.50 in winter**: humid winter nights (fog) go with
  high PM2.5. Season-dependent effects like this need a non-linear model or
  interactions.
- **t2m changes sign:** −0.30 in winter, +0.34 in monsoon.
- Wind speed ≈ 0 in every season (|ρ| ≤ 0.06). u10/v10 weak (|ρ| ≤ 0.17).
- Hour of day (hour_cos 0.27) and day of year (doy_cos 0.35) matter, matching
  the cycles above.
- All within ±0.04 of the 2020–2021 test run.

**Variograms (all seasons, standardised, see `variograms.png`):**

| field | best model | range | nugget/sill | fitted range within data? | (2020–21 test) |
|---|---|---|---|---|---|
| log PM2.5 | exponential | 1,648 km | 0.08 | **no**, still rising at 1,500 km | 1,599 km |
| residual (predictors) | exponential | 1,391 km | 0.14 | **no** | 1,258 km |
| residual (predictors + lat/lon trend) | exponential | **897 km** | 0.10 | yes | 846 km |

Detrended residual semivariance by distance (sill ≈ 0.94):

| distance | 75 km | 180 km | 260 km | 340 km | 415 km | 565 km | 710 km | 860 km |
|---|---|---|---|---|---|---|---|---|
| γ | 0.28 | 0.52 | 0.61 | 0.70 | 0.75 | 0.79 | 0.84 | 0.91 |
| % of sill | 30% | 55% | 65% | 74% | 80% | 84% | 89% | 97% |

Detrended range by season: pre-monsoon 738 km, monsoon 809 km, post-monsoon
1,183 km, winter 1,203 km (stagnant winter air → larger polluted patches).

- Even after removing predictors and the trend, **neighbouring cells are strongly
  related**: at 75 km only 30% of the variance is "new".
- Correlation drops fast up to ~400 km (80% of sill), then very slowly.
- Without detrending there is **no sill at all**: the north–south gradient makes
  far-apart cells keep getting more different. The two big ranges are
  extrapolations and should not be used.

## Block size for step 05 (decided 2026-10-18: option 1 + option 3)

Strict rule "block ≥ range" gives **897 km → 12 × 12 cells** (846 km in the
2020–2021 test run; same block size). Number of blocks
that contain India cells, for each block size:

| block | width | blocks | median India cells per block |
|---|---|---|---|
| 3 × 3 | 227 km | 77 | 8 |
| 4 × 4 | 302 km | 47 | 13 |
| **5 × 5** | 378 km | 33 | 15 |
| 6 × 6 | 454 km | 26 | 19 |
| 8 × 8 | 605 km | 16 | 28 |
| 12 × 12 | 907 km | 11 | 30 |

12 × 12 gives only 11 blocks: a 20% test set would be **2 blocks**, which can't
represent the 6 regions, and results would depend on which 2 were picked.

Options:
1. **5 × 5 blocks (~380 km, 33 blocks)**, where ~75–80% of the local
   correlation is gone (74% at 340 km, 80% at 415 km). Enough blocks to stratify by region (~7 test blocks).
   Report error vs distance to the nearest training cell (planned in step 09)
   to show how much nearby training cells still help. *Recommended.*
2. 5 × 5 blocks **plus a buffer**: drop training cells within ~1 block of a test
   block. Stricter, but removes many training cells.
3. 8 × 8 blocks (~600 km, 16 blocks, 90% of sill) as a **sensitivity run** next to
   option 1, to show how scores change with block size (useful for RQ5).

## Open issues

- ~~Rerun on 2020–2024~~ done 17:57; variogram range changed 846 → 897 km, block-size decision unaffected.
- Zero/extreme PM2.5 handling still open from step 03.
