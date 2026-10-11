# PM2.5 India (CAMS EAC4): Workflow and Results Log

Living document. Update the **Status**, **Experiments** and **Log** sections after every session, then commit.

Repo: https://github.com/dineman/pm25-india

---

## 1. Status checklist

- [x] Mac setup (Homebrew, Miniforge, `pm25` conda env, git)
- [x] ADS account, EAC4 licence accepted, `~/.cdsapirc` set
- [x] EAC4 PM2.5 downloaded 2020-2024 (60 monthly files) and converted to Zarr
- [x] EAC4 predictors downloaded (10 variables) and converted to Zarr
- [x] GitHub repo linked and pushed
- [x] Fedora machine setup
- [x] EDA (maps, seasonal/diurnal cycles, city series, correlations)
- [x] Baselines v1, v2 (features, lags, persistence)
- [ ] `baselines_v3.py` results (spatial holdout + feature importance): **run and record in Section 5**
- [ ] ERA5 `blh` and `tp` download (60 months), `era5_to_zarr.py`, merge into baselines
- [ ] **Decide high-resolution target for spatial enhancement** (ask instructor)
- [ ] Regrid inputs to a common fine grid
- [ ] Tuned gradient boosting
- [ ] Deep model (CNN / U-Net) vs bilinear interpolation
- [ ] Error analysis (season, city, top 10% days, maps)
- [ ] Write-up

---

## 2. Setup reference

### Machines
| Machine | Project path | Notes |
|---|---|---|
| Mac | `~/Project/pm25-india` | has the original raw files and Zarr stores |
| Fedora | `~/Projects/pm25-india` | has raw files and Zarr stores; `caffeinate` does not exist (use `systemd-inhibit`) |

### Environment
```bash
conda activate pm25
```
Packages: xarray, netcdf4, cfgrib, dask, zarr, geopandas, cartopy, matplotlib, pandas, scipy, scikit-learn, jupyterlab, cdsapi, tqdm, pyyaml, torch.

### Credentials (never commit, never paste in chat)
| File | Used for |
|---|---|
| `~/.cdsapirc` | ADS (CAMS EAC4): `url: https://ads.atmosphere.copernicus.eu/api` + token |
| `~/.era5key` | CDS (ERA5): bare 36-character token, read by `src/download_era5.py` |

ADS and CDS are separate accounts with separate tokens.

### Project layout
```
pm25-india/
  data/{raw, raw_pred, raw_era5, interim, processed}   # gitignored
  notebooks/  src/  configs/  results/                 # results/ gitignored
  environment.yml  RESEARCH_LOG.md  WORKFLOW.md  .gitignore
```

---

## 3. Data inventory

| Dataset | Source | Variables | Period | Location | Status |
|---|---|---|---|---|---|
| EAC4 PM2.5 (target) | ADS `cams-global-reanalysis-eac4` | `pm25` (ug/m3) | 2020-2024, 3-hourly, ~0.75 deg | `data/processed/eac4_pm25_2020_2024.zarr` | done |
| EAC4 predictors | same | `t2m, d2m, u10, v10, sp, tcc, aod550, tcco, tcno2, tcso2` | same | `data/processed/eac4_predictors_2020_2024.zarr` | done |
| ERA5 blh, tp | CDS `reanalysis-era5-single-levels` | `blh`, `tp` (mm per 3 h) | 2020-2024, hourly -> 3-hourly | `data/processed/era5_blh_tp_2020_2024.zarr` | **pending** |
| High-res PM2.5 reference | TBD (CPCB / satellite product) | | | | **undecided** |

Area box (all downloads): N 38, W 67, S 5, E 99.

Notes:
- EAC4 did not return `blh` or `tp` even though they were requested; they come from ERA5 instead.
- ERA5 downloads arrive as a ZIP of separate NetCDF files (instantaneous vs accumulated variables).
- CAMS PM2.5 is in kg/m3 and is converted to ug/m3 (x 1e9). Timestamps are UTC.

---

## 4. Commands cheat sheet

### Download and process
```bash
python src/download_all.py              # EAC4 PM2.5, 60 months
python src/to_zarr.py                   # -> eac4_pm25_2020_2024.zarr
python src/download_predictors.py       # EAC4 predictors, 60 months
python src/pred_to_zarr.py              # -> eac4_predictors_2020_2024.zarr
python src/download_era5.py             # ERA5 blh/tp, 60 months
python src/era5_to_zarr.py              # -> era5_blh_tp_2020_2024.zarr
```
Background run (Mac): `nohup caffeinate -i python src/<script>.py > results/<name>.log 2>&1 &`

Progress checks:
```bash
ls data/raw/*.nc | wc -l            # want 60
ls data/raw_pred/*.nc | wc -l       # want 60
ls -d data/raw_era5/20??_?? | wc -l # want 60
tail -n 5 results/<name>.log
```

### Analysis
```bash
python src/eda.py
python src/baselines.py
python src/baselines_v2.py
python src/baselines_v3.py
```

### Git
```bash
git status                      # always check first: no data/, .nc, .zarr, installers
git add -A && git commit -m "message" && git push
git pull                        # before starting on the other machine
```

### Stopping jobs
```bash
pkill -f download_all.py ; pkill -f download_predictors.py ; pkill -f download_era5.py
pkill caffeinate                # Mac only
```

---

## 5. Experiments and results

Evaluation setup: time split, train 2020-2022, validation 2023, test 2024. One sample = one grid cell at one time step. Metrics on the test year unless noted.

### Results table (append a row per experiment)

| Date | Experiment | Features | RMSE | MAE | Bias | R2 | Top-10% RMSE | Top-10% bias | Notes |
|---|---|---|---|---|---|---|---|---|---|
| | Climatology (month x hour mean) | n/a | 54.62 | 14.50 | 0.17 | 0.436 | n/a | n/a | MAM RMSE 83.2 |
| | Linear regression | 10 predictors + time + lat/lon | 61.14 | 22.12 | 0.26 | 0.294 | n/a | n/a | worse than climatology |
| | Gradient boosting v1 | same | 38.24 | 11.18 | -0.37 | 0.724 | n/a | n/a | val RMSE 47.3 |
| | Persistence (24 h ago) | PM2.5 lag | 61.57 | 12.80 | 0.03 | 0.284 | 188.41 | -11.55 | weak baseline |
| | **GB v2 (engineered features)** | 36 features: wind speed, RH, lag 3 h / 24 h, 24 h means | **35.85** | 10.45 | -0.40 | **0.757** | 106.27 | -20.82 | val RMSE 44.0 |
| | GB v2, log1p target | same | 43.86 | 10.06 | -3.26 | 0.637 | 135.40 | -31.80 | underpredicts peaks, dropped |
| | GB v3, all test cells | v2 features | | | | | | | **fill in** |
| | GB v3, seen cells | | | | | | | | **fill in** |
| | GB v3, held-out blocks | | | | | | | | **fill in** |
| | GB v3 + ERA5 blh/tp | v2 + blh, tp (+ lags) | | | | | | | **fill in** |

### Seasonal breakdown (GB v1, test year)
| Season | RMSE | R2 |
|---|---|---|
| DJF | 28.87 | 0.833 |
| MAM | 58.92 | 0.644 |
| JJA | 15.25 | 0.770 |
| SON | 36.11 | 0.748 |

### Feature importance (from `baselines_v3.py`, paste here)
```
(paste the "Top 15 features" list)
```

### Key findings so far
- Non-linear models are needed; linear regression is worse than climatology.
- Engineered features (lags, 24 h means, wind speed, RH) cut RMSE from 38.2 to 35.9.
- Log-transforming the target hurts peak prediction.
- Spring (MAM) is the hardest season.
- Top-10% PM2.5 values are underpredicted by about 21 ug/m3.
- Validation year (2023) is harder than test year (2024); report both.
- Caveat: aerosol and gas columns come from the same reanalysis system as the target, so skill partly reflects EAC4's internal relationships. Validate against observations (e.g. CPCB) before claiming real-world skill.

---

## 6. Decisions and open questions

| Question | Status | Decision / answer |
|---|---|---|
| What is the high-resolution target for "spatial enhancement"? | open | ask instructor (CPCB stations, satellite PM2.5, other) |
| Which machine is the main one for modeling? | open | |
| Which fine-resolution covariates (ERA5 0.25 deg, elevation, land cover, population, fire counts)? | open | |
| Spatial validation scheme | proposed | hold out 8x8-cell blocks |

---

## 7. Problems and fixes

| Problem | Cause | Fix |
|---|---|---|
| `No module named xarray` | shell in `(base)` | `conda activate pm25` |
| Push rejected, file > 100 MB | Miniforge installer committed with `git add -A` | `git reset --soft origin/main`, `git rm --cached`, add to `.gitignore`, recommit |
| ADS 401 Unauthorized | placeholder token in `~/.cdsapirc` | write real ADS token |
| ERA5 401 Unauthorized | wrong/placeholder/ADS token in `~/.era5key` | use CDS token, 36 chars, no quotes |
| `blh`/`tp` missing from EAC4 files | not returned by EAC4 | get them from ERA5 |
| xarray cannot open ERA5 download | file is a ZIP of several NetCDFs | unzip, then merge (done in `download_era5.py`) |
| `Exit 127` on Fedora | `caffeinate` is macOS only | drop it or use `systemd-inhibit` |
| `Exit 2` on download | script file not on that machine | `git pull` or recreate it; push scripts to GitHub |
| `RemoteDisconnected` / DNS errors in logs | internet or sleep drops | harmless, `cdsapi` retries; keep machine awake |
| `baselines_v2.py` missing on Fedora | never committed | commit and push all scripts |

---

## 8. Session log (newest first, add an entry each session)

### YYYY-MM-DD (template)
- Did:
- Results / numbers:
- Problems:
- Next:

### Earlier work (summary)
- Phase 1: Mac setup, project skeleton, environment, ADS access, test download and map.
- Phase 2: EAC4 PM2.5 2020-2024 downloaded, converted to Zarr.
- Phase 3: EAC4 predictors downloaded (10 variables), converted to Zarr.
- Phase 4: GitHub linked; large-file push rejection fixed.
- Phase 5: Fedora setup; ADS 401 fixed.
- Phase 6: EDA and baselines v1-v3 written; v1 and v2 results recorded above.
- Phase 7: ERA5 route chosen for `blh`/`tp`; scripts written; full download pending.
