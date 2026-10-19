# Step 10: Feature importance

**Status:** **done** 2026-10-19 07:40–08:00 on the Mac (SHAP ~15 min, permutation
~5 min). Terms in *italics* are explained in [glossary.md](glossary.md).

## Purpose

Answer RQ3: **which atmospheric and aerosol features matter most** for
predicting PM2.5, and does that change by region and season? The model studied is
the final LightGBM of step 07 (feature set `all`, 14 features, fitted on
log(1 + PM2.5)), the best ML model and the ML part of the step 08 hybrid.

## The idea, in plain words

Two questions that sound alike but aren't:

1. **What does the model use?** *SHAP values* split every single prediction into
   contributions, one per feature: "AOD pushed this prediction up by 30%, the hour
   of day pulled it down by 10%, …". Averaging the size of these contributions
   over many predictions gives each feature's importance. Because the model works
   on log(1 + PM2.5), contributions are in log units: **0.1 ≈ ±10% PM2.5**.
2. **What does the accuracy depend on?** *Permutation importance*: shuffle one
   feature's values between rows (so it no longer matches the rest of the row) and
   see how much worse the predictions at the **test cells** get. A feature the
   model relies on and that can't be replaced by others gives a big jump.

Some features carry nearly the same information (temperature, dewpoint and
humidity; the three wind variables; sin and cos of the hour). Shuffling only
one of them underestimates the group, because its partners still give the model
part of that information. So both measures are also computed for **groups of
related features**, shuffled together and with SHAP values summed.

## How it works

Code: `src/pm25/importance.py`, `scripts/feature_importance.py`.

1. **SHAP**: random sample of **200,000 rows** (time step × cell) from **all 494
   India cells**, training and test. SHAP explains the model, not its accuracy,
   so any cell can be used, and this gives every region and season many rows.
   LightGBM computes exact *TreeSHAP* values itself (`pred_contrib=True`); this
   gives the same numbers as the `shap` package, only faster. Summaries: mean |SHAP|
   per feature and group, overall, by region, by season; dependence curves (mean
   SHAP per 5% quantile bin of the feature, per season) for the 6 top features.
2. **Permutation**: random sample of **300,000 test-cell rows**; each of the 14
   features and 7 groups shuffled **5 times** (different random orders); increase
   in RMSE and MAE over the unshuffled predictions.

Groups (`FEATURE_GROUPS`):

| Group | Features |
|---|---|
| aerosol | aod550 |
| wind | u10, v10, wind_speed |
| temperature & humidity | t2m, d2m, rh2m |
| surface pressure | sp |
| location | lat, lon |
| hour of day | hour_sin, hour_cos (local solar hour) |
| day of year | doy_sin, doy_cos |

## Outputs

`results/10_importance/`:

| File | Content |
|---|---|
| `shap_features_overall.csv`, `shap_groups_overall.csv` | mean \|SHAP\| per feature / group |
| `shap_{features,groups}_by_{region,season}.csv` | same, per region / season |
| `shap_dependence.csv` | dependence curves (feature, season, bin value, mean SHAP, rows) |
| `permutation_features.csv`, `permutation_groups.csv` | every repeat: RMSE, MAE and their increase |
| `importance_features.png`, `importance_groups.png` | mean \|SHAP\| next to permutation ΔRMSE |
| `shap_groups_by_region.png`, `shap_groups_by_season.png` | heatmaps |
| `shap_dependence.png` | effect curves of the 6 top features, by season |
| `config.json` | arguments, features, groups |

Log: `results/logs/feature_importance.log` (it also contains the 5,000-row smoke
test at 07:40 that wrote to a scratch folder). Tests: `tests/test_importance.py`
(SHAP values add up to the model's log prediction, informative feature ranked
first by SHAP and permutation, group sums, dependence bins).

## How to run

```bash
conda run -n pm25 python scripts/feature_importance.py      # ~20 min on the Mac
```

Needs `results/07_models/all/model_lgbm.joblib` (not in git; re-run step 07 on a
new machine).

## Decisions

| Decision | Reason |
|---|---|
| Explain LightGBM only | Best ML model and the ML part of the hybrid; RF/MLP are clearly worse (step 09). |
| SHAP on all India cells, permutation on test cells | SHAP describes the model (any input will do, more coverage is better); permutation measures accuracy, which needs held-out cells. |
| LightGBM's own TreeSHAP instead of the `shap` package | Exact same values, faster, one dependency less at run time. |
| Grouped permutation and grouped SHAP | Correlated features (t2m/d2m/rh2m, winds, sin/cos pairs) hide each other's importance when handled one at a time. |
| SHAP in log units | The model is fitted on log(1 + PM2.5), so contributions add up there; they read as relative (%) effects. |

## Results

### Overall

| Group | mean \|SHAP\| (≈ effect) | ΔRMSE when shuffled (µg m⁻³) | ΔMAE |
|---|---|---|---|
| aerosol (aod550) | **0.35** (±~40%) | 11.4 | **12.5** |
| hour of day | 0.22 | 13.9 | 10.6 |
| day of year | 0.17 | 14.5 | 9.3 |
| location (lat, lon) | 0.17 | **18.8** | 13.3 |
| surface pressure | 0.15 | 10.2 | 8.5 |
| temperature & humidity | 0.15 | 7.5 | 6.3 |
| wind | **0.08** (lowest) | **5.5** (lowest) | 4.0 |

(Shuffle-to-shuffle spread: ≤ 0.24 µg m⁻³, so all differences above are real
for this sample.) Single features: aod550 0.35, hour_cos 0.18, doy_cos 0.16,
sp 0.15, lat 0.12, t2m 0.09, lon 0.07, d2m 0.07, hour_sin 0.07, wind_speed
0.06, rh2m 0.04, doy_sin 0.03, u10 0.03, v10 0.02.

- **AOD is the feature the model uses most**, by far: a typical contribution of
  ±35% (log units), twice the next feature. Shuffling it gives the **largest MAE
  increase** (+12.5), i.e. it matters for the *typical* hour.
- **Location matters most for the test-cell RMSE** (+18.8). Lat/lon carry
  large-scale emission patterns (IGP vs south) that no other feature describes,
  so nothing can stand in for them. This agrees with step 07 (dropping lat/lon:
  R² 0.59 → 0.48).
- **Time of day and time of year** are next. The model has no boundary-layer
  height or emission inventory, so it learns those cycles from the clock and the
  calendar.
- **Wind and humidity matter least** on average, despite their physical role;
  their effect is partly already contained in AOD (see "What this means").

### How each feature acts (dependence curves, `shap_dependence.png`)

- **aod550**: strongly increasing and flattening (log-like): from −1.5 (≈ −78%)
  at AOD ≈ 0.05 to +0.7 (≈ +100%) at AOD ≈ 1.5. Same shape in every season; at a
  given AOD, the monsoon effect is slightly higher.
- **hour (hour_cos: +1 = local midnight, −1 = local noon)**: about **+0.25 at
  night and −0.3 around noon**, i.e. ~30% more PM2.5 at night than the average
  and ~25% less at midday. This is the *boundary-layer* effect (shallow mixed
  layer at night). Weakest in the monsoon (rain and wind mix the air at all hours).
- **day of year (doy_cos: +1 = 1 January, −1 = early July)**: +0.2 in winter,
  −0.2 in the monsoon. A seasonal emission/meteorology cycle beyond what the
  weather features explain (crop-residue burning, heating, monsoon washout).
- **sp (surface pressure, hPa ≈ altitude)**: almost a straight line from −0.8
  (≈ −55%) at 600 hPa (~4,000 m) to +0.25 at 1,010 hPa (sea level). Mountains
  have cleaner air.
- **lat**: +0.2 north of 25° N (Indo-Gangetic Plain), −0.15 around 15–21° N.
- **t2m**: **cold raises PM2.5, but only in winter and post-monsoon**: +0.2 at
  10–20 °C, falling steeply above ~22 °C to −0.15 at 32 °C (cold nights trap
  pollution; warm days mix it). In the monsoon and pre-monsoon the effect is
  small (≤ 0.1). Step 04 had seen the same: temperature "flips sign" between
  seasons.

### By region (mean |SHAP| of groups)

| Group | igp | northwest | central | south | northeast | himalaya |
|---|---|---|---|---|---|---|
| aerosol | 0.36 | 0.29 | 0.28 | 0.28 | **0.54** | **0.77** |
| surface pressure | 0.20 | 0.11 | 0.07 | 0.08 | 0.25 | **0.57** |
| location | **0.26** | 0.16 | 0.11 | 0.16 | 0.19 | 0.25 |
| hour of day | 0.23 | 0.20 | 0.21 | 0.20 | 0.26 | 0.25 |
| temperature & humidity | 0.17 | 0.14 | 0.15 | 0.15 | 0.13 | 0.12 |
| wind | 0.08 | 0.08 | 0.07 | **0.10** | 0.06 | 0.08 |

- **Himalaya: AOD and surface pressure dominate** (0.77 and 0.57): altitude
  (pressure) sets the level, AOD the variation. This is why ML beat interpolation
  so clearly there in step 09: interpolation doesn't know the altitude.
- **Northeast: AOD matters most after the Himalaya** (0.54): fire smoke shows up
  in AOD, which is the model's main handle on the fire season.
- **IGP: location matters most here** (0.26): the plain is polluted beyond what
  its weather explains.
- **South: wind matters most here** (0.10, still small): sea breezes and the
  monsoon flow.

### By season (mean |SHAP| of groups)

| Group | winter | pre-monsoon | monsoon | post-monsoon |
|---|---|---|---|---|
| aerosol | 0.38 | 0.32 | 0.35 | 0.38 |
| hour of day | 0.23 | **0.25** | **0.18** | 0.22 |
| day of year | 0.20 | **0.12** | 0.19 | 0.19 |
| temperature & humidity | 0.18 | **0.10** | 0.16 | 0.16 |
| surface pressure | 0.18 | 0.14 | 0.13 | 0.18 |
| location | 0.17 | 0.18 | 0.17 | 0.17 |
| wind | 0.07 | 0.07 | 0.08 | 0.07 |

- **Pre-monsoon: weather explains least** (temperature & humidity 0.10 vs
  0.16–0.18 elsewhere). Fires dominate and the weather features don't describe
  them, which matches pre-monsoon being the hardest season (step 09).
- **Monsoon: the daily cycle matters least** (0.18), as expected with
  well-mixed, rainy conditions.

## What this means (RQ3)

- **Aerosol optical depth is the most important predictor**, followed by the
  daily and seasonal cycles, location and altitude (surface pressure).
  Meteorology (temperature, humidity, wind) matters less on its own; temperature
  matters mainly on winter nights.
- **Caveat (carry into the write-up):** in CAMS, PM2.5 and AOD are both computed
  from the **same simulated aerosol field**, so the strong AOD → PM2.5 link is
  partly built into the data. With satellite AOD and ground-station PM2.5 the
  link would be weaker and meteorology would likely matter more.
- Lat/lon and time features stand in for things the dataset lacks (emission
  inventories, boundary-layer height, fire counts). Adding those would make the
  model more physical and probably more transferable; out of scope here.
- **Importance is not the same as causation**: SHAP shows what the model does
  with the data it was given, not what causes pollution.

## Open issues

- The heatmaps use white text above the 70th percentile; a few mid-blue cells
  (0.23–0.26) have low contrast. Fix the threshold when the figures are
  redone for the write-up.
- Permutation importance was measured on the 6 test blocks only (same caveat as
  step 09); the SHAP part covers all regions.
