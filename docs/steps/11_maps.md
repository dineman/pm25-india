# Step 11: Spatially enhanced PM2.5 maps

**Status:** **done** 2026-10-19 on the Mac (full run 08:12–08:23; re-run
08:23–08:35 with a layout fix only, same numbers). Terms in *italics* are explained in
[glossary.md](glossary.md).

## Purpose

The project brief asks for "spatially enhanced PM2.5 maps over India" (RQ6):
use the trained models to map PM2.5 on a grid **finer than CAMS** (0.75°,
~80 km) and compare that with standard interpolation.

## The idea, in plain words

CAMS gives PM2.5 and the weather/aerosol predictors only every 0.75°. We map at
**0.25°** (~28 km, 3 × finer):

1. **Bilinear** (the standard method): blend the 4 surrounding CAMS PM2.5 values.
2. **Kriging**: weighted average of the 494 India CAMS cells (step 06 recipe).
3. **LightGBM**: blend the 4 surrounding values of each *predictor* (AOD, wind,
   temperature, …) to the fine point, then let the step 07 model predict PM2.5
   from them.
4. **Hybrid** (LightGBM + kriged residuals): LightGBM, plus the model's errors at
   the CAMS cells interpolated by kriging (the step 08 method).

**What this can and can't do:** no data below 0.75° exists here, so a fine map
can't be checked at its own resolution (accepted limitation, user 2026-10-19).
The fine predictors are themselves interpolated, so no method adds *observed*
detail; LightGBM adds detail *from what it learned* about how PM2.5 responds to
the predictors (e.g. altitude via surface pressure). The only possible check is
**consistency**: averaging a fine map back to 0.75° should give CAMS again.

## Boundary (user decision, 2026-10-19)

The maps use **India's official boundary**: Natural Earth 10 m countries,
*India point of view* (`admin_0_countries_ind`), which includes all of Jammu &
Kashmir and Ladakh (PoK, Gilgit-Baltistan, Aksai Chin). Neighbouring countries
are drawn from the same file. **The model and evaluation domain is unchanged**
(the step 02 region mask, 494 cells, de facto line, Mumbai/Chennai cells
excluded), so fine points in PoK/Aksai Chin and along the excluded coasts are
filled by the methods but lie **outside the area the models were trained and
evaluated on**.

## How it works

Code: `src/pm25/maps.py`, `scripts/make_maps.py`.

1. **Fine grid**: 0.25° points from 6.0–37.5° N, 68.25–97.5° E (127 × 118);
   keep the **4,636 points inside the official boundary**.
2. **Predictors at fine points**: bilinear interpolation of PM2.5, AOD, u10, v10,
   t2m, d2m and sp; then wind speed, humidity and the time features (local solar
   hour from the fine point's longitude, day of year) are derived **with the
   step 03 code**, so the model sees inputs built exactly as in training.
3. **Residuals at the CAMS cells** (all 494 India cells): CAMS − prediction of
   **the same final LightGBM** that makes the fine map. Kriged to the fine points
   with the step 08 residual variograms (per season). At a cell centre the hybrid
   therefore reproduces CAMS (exactly in the seasons without a nugget).
4. **Month by month** (60 months, ~10 s each): predict all 4 methods for every
   3-hourly time step, accumulate seasonal sums, keep three example time steps.
5. **Consistency check**: for the 426 cells fully covered by 3 × 3 fine points,
   average the fine values and compare with CAMS (6.2 million values).

Example time steps (chosen automatically from CAMS, 2024):

| Example | Time | Why |
|---|---|---|
| smog IGP | 20 Dec 2024, 02:30 IST | highest IGP-mean PM2.5 of 2024 |
| fire northeast | 15 Mar 2024, 02:30 IST | highest northeast-mean PM2.5 in Mar–May 2024 |
| monsoon afternoon | 26 Jul 2024, 14:30 IST | median India-mean day of July 2024, afternoon |

## Outputs

`results/11_maps/`:

| File | Content |
|---|---|
| `season_means.png` | seasonal mean PM2.5 2020–2024: CAMS, bilinear, hybrid, hybrid − bilinear |
| `examples.png` | the three example time steps: CAMS, bilinear, LightGBM, hybrid |
| `zoom_north_winter.png` | north India (26–37.5° N, 72–82° E), winter mean |
| `consistency_with_cams.csv` | fine maps averaged to 0.75° vs CAMS |
| `season_summary.csv` | India-mean per season and method; mean / 95th percentile of \|hybrid − bilinear\| |
| `maps_0p25.nc` | seasonal means and examples on the 0.25° grid, all methods (**not in git**, ~2 MB) |
| `config.json` | arguments, point counts, example times, boundary source |

Log: `results/logs/make_maps.log` (also contains the two 3-month smoke tests,
written to a scratch folder). Tests: `tests/test_maps.py` (fine axis, points in
polygon, interpolation + derived features incl. local solar hour, aggregation).

## How to run

```bash
conda run -n pm25 python scripts/make_maps.py                  # ~11 min on the Mac
conda run -n pm25 python scripts/make_maps.py --max-months 3 \
    --out-dir /tmp/maps_test                                   # quick test
```

Needs `results/07_models/all/model_lgbm.joblib` and the configs of steps 06 and 08.

## Decisions

| Decision | Reason |
|---|---|
| 0.25° (3 × finer) | Enough to show what the methods do between CAMS points; finer adds points but no information. |
| Interpolate the base predictors, then derive humidity, wind speed and time features | Same order as step 03, so the model's inputs are built exactly as in training. |
| Residuals from the same final model (not step 08's out-of-fold residuals) | See "Problems met". Out-of-fold residuals are right for *evaluating* at held-out cells; for a *map* the correction must match the model being corrected. |
| Use the step 07 model (trained on 399 cells), not a refit on all 494 | Keep the model that was evaluated; the 95 test cells are still covered through their residuals. |
| Bilinear from the whole CAMS box, kriging/hybrid residuals from the 494 India cells | Bilinear is the plain regridding anyone would do; kriging and the hybrid follow the project's domain (steps 06, 08). |
| Fixed colour bins (0–15–30–45–60–90–120–180–250–400) | Same scale on every map; bins include India's AQI breakpoints (30, 60, 90, 120, 250). |

## Problems met

- **Smoke test: `feature_matrix` needs PM2.5 as a shape reference.** Fixed by
  interpolating PM2.5 together with the predictors; that interpolated PM2.5 is
  the bilinear map.
- **Hybrid map double-counted at training cells** (smoke test, 08:10). The first
  version added step 08's *out-of-fold* residuals to the *final* model. The final
  model has seen the training cells and fits them closely, while out-of-fold
  residuals are as large as errors at new places, so the correction overshot:
  pale patches in the south, averaged hybrid vs CAMS RMSE 29.7 and bias +4.4 (3
  months). With residuals from the final model: RMSE 15.8, bias −0.1.
- **Layout**: the first panel title overlapped the figure title; fixed and
  re-run (numbers unchanged).

## Results

### Consistency: fine maps averaged back to 0.75° vs CAMS (426 cells, 2020–2024)

| method | RMSE | MAE | R² | bias |
|---|---|---|---|---|
| bilinear | **8.9** | **3.7** | **0.985** | −0.3 |
| hybrid | 10.5 | 3.9 | 0.979 | −0.2 |
| kriging | 15.8 | 6.7 | 0.952 | −0.3 |
| LightGBM | 28.9 | 12.5 | 0.841 | −3.5 |

Bilinear and the hybrid both stay very close to CAMS (they pass through the CAMS
values at the cell centres). LightGBM alone drifts from CAMS (R² 0.84) and keeps
its known low bias. Kriging from the 494 India cells smooths more than bilinear.

### Seasonal India means (mean over the 4,636 fine points, µg m⁻³)

| season | bilinear | kriging | LightGBM | hybrid | mean \|hybrid − bilinear\| | 95th pct |
|---|---|---|---|---|---|---|
| winter | 94.9 | 95.5 | 91.6 | 95.3 | 6.4 | 19.3 |
| pre-monsoon | 65.2 | 65.7 | 61.1 | 65.5 | 5.5 | 14.7 |
| monsoon | 44.3 | 44.6 | 42.6 | 44.3 | 2.2 | 6.3 |
| post-monsoon | 89.4 | 89.9 | 86.5 | 89.4 | 5.7 | 17.5 |

### What the maps show

- **The large-scale picture is the same in every method**: the Indo-Gangetic
  Plain is the hotspot (winter and post-monsoon means mostly 120–250 µg m⁻³
  from Punjab to Bihar), the Himalaya and the south are cleanest, and
  the monsoon is clean everywhere except the northwest (desert dust).
- **Hybrid vs bilinear differ by ~2–6 µg m⁻³ on average** (up to ~15–20 at the
  95th percentile, most in winter/post-monsoon, least in the monsoon). The
  differences are concentrated in the **IGP**, the **Himalayan foothills**, the
  **northeast**, around **Kolkata/the Bangladesh border**, on the **west coast**
  near Mumbai, and in **PoK** (outside the model domain, see "Boundary").
- **Lattice pattern**: the hybrid − bilinear map shows lines on the 0.75° CAMS
  grid. At every CAMS cell centre the hybrid and bilinear both equal CAMS
  (difference 0); between centres the LightGBM part takes over. The fine maps
  therefore keep an imprint of the 0.75° grid; this is a property of the method,
  not a bug, and should be stated in the write-up.
- **Examples**: in the Dec 2024 smog night all methods agree on the IGP band
  (> 250); in the Mar 2024 fire night all fine maps keep the northeast hotspot
  (Tripura–Mizoram, > 250) that CAMS shows; on the monsoon
  afternoon LightGBM looks grainy (it reacts to small changes in the interpolated
  predictors) and the hybrid is close to bilinear.
- **North India zoom (winter)**: the three fine maps look very similar. Measured
  as the change in log(PM2.5) between neighbouring 0.25° points (26–37.5° N,
  72–82° E), the *average* gradient is the same (bilinear 0.214, LightGBM 0.216,
  hybrid 0.217), but the **steepest 5% of steps are 11–14% steeper** in LightGBM
  (0.375) and the hybrid (0.366) than in bilinear (0.329). Those are the
  plain-to-Himalaya transitions, where the model ties PM2.5 to altitude (surface
  pressure, step 10). The clearest case where the model adds plausible detail;
  a modest effect.

## What this means (RQ6)

- A fine map built **only from interpolated CAMS fields can't contain new
  information**. Bilinear regridding is the most faithful to CAMS (it reproduces
  it best when averaged back), and the hybrid is almost as faithful.
- **The ML-based maps add structure where the model has learned a strong,
  physically sensible link**, mainly altitude at the Himalayan foothills, at the
  cost of a visible 0.75° imprint and of detail that can't be checked.
- **LightGBM alone is not a good mapping method** here: it drifts from CAMS
  (R² 0.84 when averaged back) and is biased low. The hybrid fixes both.
- Real enhancement would need predictors at the fine resolution (e.g. a
  digital elevation model, satellite AOD at 1–10 km, population or emission
  maps) and fine-scale reference data to test against; out of scope here
  (only CAMS is available).

## Open issues

- Fine points outside the model domain (PoK, Aksai Chin, excluded coastal cells)
  are extrapolations; consider shading or hatching them in the write-up figures.
- Hybrid residuals at the 399 training cells are in-sample (from the model that
  saw them), so the hybrid map is closer to CAMS there than at the 95 test cells.
  Doesn't affect the maps' appearance, but don't read the consistency check as accuracy.
