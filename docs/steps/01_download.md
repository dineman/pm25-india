# Step 01: Download CAMS EAC4 data

**Status:** **done** 2026-10-18 17:55. All **60 monthly files** (2020-01 to
2024-12) in `data/raw/eac4/`, 381 MB, no `.part` files, no failed months.
Started 12:09; took ~5¾ h instead of the ~3.5 h expected, because of ADS server
slowdowns and one restart (see below).

## Purpose

Get the raw data for the whole project: PM2.5 (the target) and the predictor
variables from the CAMS EAC4 global reanalysis, for India, 2020–2024, at the
native 3-hourly time step.

## Data source

- Dataset: **CAMS global reanalysis (EAC4)**, ADS id `cams-global-reanalysis-eac4`
  https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4
- Horizontal grid: **0.75° × 0.75°** (≈ 80 km). Time step: **3 hours**
  (00, 03, …, 21 UTC).
- EAC4 is a *model* reanalysis: a chemistry-weather model with satellite
  observations assimilated. It is not station data.

## What is downloaded

| ADS request name | Short name in file | Role | Raw units |
|---|---|---|---|
| `particulate_matter_2.5um` | `pm2p5` | **target** | kg m⁻³ |
| `total_aerosol_optical_depth_550nm` | `aod550` | predictor | dimensionless |
| `10m_u_component_of_wind` | `u10` | predictor (east–west wind) | m s⁻¹ |
| `10m_v_component_of_wind` | `v10` | predictor (north–south wind) | m s⁻¹ |
| `2m_temperature` | `t2m` | predictor | K |
| `2m_dewpoint_temperature` | `d2m` | predictor (humidity) | K |
| `surface_pressure` | `sp` | predictor (also proxy for altitude) | Pa |

Area requested: **[N 38, W 68, S 6, E 98]**, a box around India. The grid ADS
returns is aligned to 0.75° multiples: **lat 6.0–37.5 (43 points), lon
68.25–97.5 (40 points)**. The box includes sea and neighbouring countries; those
are removed later by the mask (step 02).

## How it works

1. `iter_months` lists every month from `--start` to `--end`.
2. For each month, `build_request` makes one ADS request: all 7 variables, every
   day of the month, all 8 times, the area, NetCDF format.
   One request per month keeps jobs small and lets an interrupted run resume.
3. `download_month`:
   - **skips** the month if `eac4_india_YYYY-MM.nc` already exists;
   - otherwise downloads to `eac4_india_YYYY-MM.nc.part` and **renames** it only
     when complete, so a crash never leaves a half file that would be skipped later.
4. If one month fails, the error is logged and the script moves on; failed months
   are listed at the end and the script exits with code 1. **Re-running retries
   only the missing months.**
5. Network errors are retried automatically by the ADS client (up to 500 tries,
   waiting up to 120 s between them).

Each request goes through the ADS queue: *accepted* → *running* → *successful*
→ download (a few seconds for ~6.5 MB). Most of the ~3–4 min per month is queue time.

## Inputs and outputs

- Input: ADS credentials in `~/.cdsapirc` (url `https://ads.atmosphere.copernicus.eu/api`
  plus your key). **Never commit this file or put the key in code.**
- Output: `data/raw/eac4/eac4_india_YYYY-MM.nc`, 60 files, ~6.5 MB each,
  ~0.4 GB total. `data/raw` is never modified after download.
- Log: `results/logs/download_eac4.log`.

## How to run

```bash
# test one month first
PYTHONPATH=src conda run -n pm25 python scripts/download_eac4.py --start 2020-01 --end 2020-01
# full range (default 2020-01 to 2024-12)
PYTHONPATH=src conda run -n pm25 python scripts/download_eac4.py
```

Code: `src/pm25/download.py` (logic), `scripts/download_eac4.py` (entry point),
`tests/test_download.py` (uses a fake client; no network).

## Decisions

| Decision | Reason |
|---|---|
| 3-hourly (all 8 times) | User choice; keeps the diurnal cycle. Download size is small either way, and coarsening later is always possible, refining is not. |
| One request per month | Small jobs, resumable, a failure costs one month only. |
| Write to `.part`, then rename | Prevents truncated files being mistaken for finished ones. |
| Bounding box, mask later | ADS only accepts rectangles; the exact India shape is applied in step 02. |

## Results and checks (test month January 2020)

- File: 248 times (31 days × 8) × 43 lat × 40 lon; all 7 variables present;
  **no missing values**.
- Dimension names from ADS: `valid_time`, `latitude`, `longitude` (renamed in step 03).
- Value ranges were physically sensible: t2m −45 to 36 °C, sp 506–1027 hPa
  (low values = high mountains), wind 0–16 m s⁻¹, AOD 0.002–5.5.
- PM2.5 over the whole box: median 21 µg m⁻³; **max ≈ 12,500 µg m⁻³ at
  37.5°N 84.75°E (Taklamakan desert dust, outside India)**; ~2% exact zeros,
  over the high Himalaya/Tibet/northern Pakistan.
- Delhi cell (28.5°N, 77.25°E) January mean: **381 µg m⁻³**: the expected winter
  haze, but higher than ground monitors usually show (known CAMS bias over the IGP).

## Issues met and fixed

- **Duplicate log lines:** the ADS client adds its own console handler, so every
  status line appeared twice. Fixed by giving its logger a `NullHandler`
  (`scripts/download_eac4.py`). The run started before the fix still shows duplicates.
- **One connection timeout** (12:13), retried automatically after 120 s; no data lost.
- **Many more connection drops later** (~30 "Retrying in 120 seconds" lines in the
  log, e.g. 12:51–13:19, 15:33–15:38, 16:56–17:07, 17:46). All retried
  automatically; no month failed. Speed varied from ~3 min/month to ~6 min/month.
- **Requests stuck in the ADS queue:** Sep 2023 (16:05–~16:30) and Dec 2023
  (16:40–17:09) sat at "accepted" for 20–30 min with failing status checks, then
  completed on their own. Restarting would have lost the queue position, so we waited.
- **Restart after the Claude session ended (14:24):** the download runs as a
  background job of the Claude session, so closing the session stopped it at
  27/60 months. Restarted 14:30 (session 00e26aa7); finished months were skipped,
  the interrupted month re-requested. **Keep the session open during long downloads**
  (or run the script yourself in a terminal with `nohup`).

## Open issues / to check when finished

- ~~Confirm 60 files, none `.part`, no failures~~ checked 17:55: 60 files, nothing missing or extra.
- ~~Step 03 checks the time axis~~ checked 17:56: 14,616 steps, all 3 h apart.
- **ADS Terms of Use change on 2026-10-28.** Continuing to use ADS after that date
  counts as accepting the new terms. Review the notice if you will download again.
