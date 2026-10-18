# Step 03: Preprocessing

**Status:** **done** on all 60 months (2026-10-18 17:56, 60 s, 2.2 GB peak) →
`data/processed/eac4_india_3h.nc`: **14,616 time steps** × 43 lat × 40 lon,
494 India cells, **632 MB**. Earlier: tested on 3 months (Jan–Mar 2020), then run
on 2020–2021 for test runs of steps 04–06 (`eac4_india_3h_2020_2021.nc`, kept).
Full-data checks are in "Results and checks (full run)" below.

## Purpose

Turn 60 raw monthly files into **one tidy dataset** with consistent names and
units, derived features and the India mask attached. This is the input for
every later step.

## How it works

Code: `src/pm25/preprocess.py` (each function does one thing) and
`scripts/preprocess_eac4.py` (runs them and writes the file).

1. **Open all months lazily** with `xr.open_mfdataset` (dask, one chunk per
   ~month), so the whole 5 years never has to fit in memory at once.
2. `standardize`: rename `valid_time/latitude/longitude` → **`time/lat/lon`**
   (project convention), drop extra coordinates, sort latitude **south → north**
   (ADS gives it north → south).
3. `convert_units`:
   - PM2.5: kg m⁻³ × 10⁹ → **µg m⁻³**
   - t2m, d2m: K − 273.15 → **°C**
   - sp: Pa ÷ 100 → **hPa**
4. `add_met_features`:
   - `wind_speed` = √(u10² + v10²), m s⁻¹. Wind direction is not added
     separately: u10 and v10 already carry it.
   - `rh2m` = relative humidity (%) from the Magnus formula:
     RH = 100 · exp(17.625·Td/(243.04+Td)) / exp(17.625·T/(243.04+T)), with T, Td in °C;
     clipped at 100%.
5. `add_time_features`:
   - **Local solar hour** = UTC hour + lon/15, so it differs by ~2 h between the
     west and east of the domain. Encoded as `hour_sin`, `hour_cos`, so 23:00 and
     01:00 are close together. These have dims **(time, lon)**, not (time, lat, lon),
     to save memory; they broadcast when the training table is built.
   - Day of year → `doy_sin`, `doy_cos` (period 365.25 days), dims (time).
   - `season` coordinate (IMD): winter Jan–Feb, pre_monsoon Mar–May,
     monsoon Jun–Sep, post_monsoon Oct–Dec.
6. Attach the mask from step 02 as coordinates `in_india`, `region`, `state`.
   The script first checks the mask grid **exactly matches** the data grid and
   stops if not.
7. Write a compressed NetCDF (zlib level 4), first to `.part`, then rename.

The time axis **stays in UTC**. Local time only enters through the solar-hour features.

## Output dataset

`data/processed/eac4_india_3h.nc`, **full grid** (43 × 40), including non-India
cells, because interpolation baselines and maps need the neighbours.
**Filter with `in_india` for modelling.**

| Variable | Dims | Units | Notes |
|---|---|---|---|
| `pm2p5` | time, lat, lon | µg m⁻³ | target |
| `aod550` | time, lat, lon | – | |
| `u10`, `v10` | time, lat, lon | m s⁻¹ | |
| `t2m`, `d2m` | time, lat, lon | °C | |
| `sp` | time, lat, lon | hPa | |
| `wind_speed` | time, lat, lon | m s⁻¹ | derived |
| `rh2m` | time, lat, lon | % | derived |
| `hour_sin`, `hour_cos` | time, lon | – | local solar hour |
| `doy_sin`, `doy_cos` | time | – | |
| coords `season` (time); `in_india`, `region`, `state` (lat, lon) | | | |

Size: **632 MB** for 5 years (the 3-month test was 30 MB).

## How to run

```bash
PYTHONPATH=src conda run -n pm25 python scripts/build_region_mask.py   # step 02 first
PYTHONPATH=src conda run -n pm25 python scripts/preprocess_eac4.py
```

Log: `results/logs/preprocess_eac4.log`. Tests: `tests/test_preprocess.py`
(renaming, units, wind speed, RH, solar hour, seasons).

## Decisions

| Decision | Reason |
|---|---|
| Local **solar** hour, not IST | The daily cycle of the boundary layer (which traps or mixes PM2.5) follows the sun. IST is the same clock for the whole country although the sun rises ~2 h earlier in the east. |
| sin/cos encoding of hour and day | Makes cyclic time continuous (Dec 31 next to Jan 1). |
| IMD seasons | Official Indian definition; matches the monsoon cycle. |
| Keep full grid in the file | Needed for interpolation baselines (neighbours) and maps. |
| Keep exact-zero PM2.5 values | They are the real CAMS target values; removing them would bias results. Revisit in evaluation. |
| No target transform yet | Decide log(PM2.5) in step 07 by testing both. |

## Results and checks (full run, 2020–2024, India cells only)

- **Time axis:** 2020-01-01 00:00 to 2024-12-31 21:00 UTC, **14,616 steps**, every
  gap exactly 3 h. **No missing values** in any variable.
- Steps per season: monsoon 4,880, pre-monsoon 3,680, post-monsoon 3,680, winter 2,376.
- Ranges: t2m −46.5 to 50.4 °C, d2m −58.0 to 32.1 °C, sp 527–1,024 hPa,
  wind 0–17.9 m s⁻¹, RH 2–100%, AOD 0.00–7.84.
- **PM2.5:** mean 72.2, 1st percentile 1.8, **median 51.9**, 99th 355, 99.9th 633,
  **max 5,651 µg m⁻³** (Assam, 25.5 °N 93.0 °E, 2021-03-22 21:00 UTC: northeast
  fire season). Exact zeros 0.38% (same as the 3-month test).
- **Annual means:** 2020 66.3 (lowest: COVID lockdown), 2021 75.8, 2022 72.1,
  2023 74.0, 2024 72.8 µg m⁻³.

## Results and checks (3-month test, Jan–Mar 2020, India cells only)

- Time axis: 728 steps, evenly spaced every 3 h, no gaps.
- January ranges after conversion: t2m −45 to 36 °C, d2m −52 to 27 °C,
  sp 506–1027 hPa, wind 0–15.7 m s⁻¹, RH 2–100%.
- PM2.5 distribution: 1st percentile 1.8, **median 60.7**, 99th 403,
  99.9th 840 µg m⁻³. **Strongly right-skewed** → a log target is likely better.
- **Exact zeros: ~0.4% of values**, in 128 cells, mostly **Ladakh** (surface
  pressure ~540 hPa, i.e. very high altitude) plus a few south-Indian hill cells,
  and **only at 00–09 UTC (daytime)**. Looks like a lower limit in CAMS, not a
  download error.
- **Extremes up to ~4,700 µg m⁻³ over Mizoram/Meghalaya in mid-March 2020**, with
  high AOD at the same time. This matches the spring fire season in northeast
  India and Myanmar as CAMS represents it. These are real model values, not errors.

## Run history

| When | Command | Result |
|---|---|---|
| 2026-10-18 ~12:19 | 3 months (Jan–Mar 2020), to scratch | 728 times, checks above |
| 2026-10-18 14:24 | `--pattern 'eac4_india_202[01]-*.nc' --out data/processed/eac4_india_3h_2020_2021.nc` | **interrupted** (session ended), `.part` file deleted |
| 2026-10-18 14:30 | same, rerun | done in 14 s: 5,848 × 43 × 40, 494 India cells |
| 2026-10-18 17:55 | default (all 60 months) | **done in 60 s**: 14,616 × 43 × 40, 632 MB |

## Open issues

- ~~Run on all 60 months~~ done 17:56; 14,616 steps, no gaps.
- Decide whether to treat zero and extreme PM2.5 specially (e.g. log1p target,
  robust loss) in step 07.
- `season` is stored as text; convert to categories when building tables, to save memory.
