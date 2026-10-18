# PM2.5 spatial enhancement over India (course project)

Full brief: docs/PROJECT.md. Read it before planning any substantial work.

## In one paragraph
Predict CAMS EAC4 PM2.5 at grid cell s, time t from CAMS predictors at the same s, t
(AOD, 10 m u/v wind, 2 m temperature, 2 m dewpoint, surface pressure, lat/lon), India,
2020–2024. Evaluate on HELD-OUT grid cells. Compare several ML models, analyse feature
importance, break down errors by region and season, and compare ML-based spatial
enhancement against standard interpolation (e.g. bilinear, IDW, kriging).

## Methodology rules (important)
- Spatial hold-out: split train/test by GRID LOCATION, not by random rows. Prefer
  spatial blocks over isolated cells, because neighbouring cells are highly correlated
  and random cell hold-out inflates scores.
- Every model must be compared against simple baselines (interpolation from training
  cells, linear regression).
- Report RMSE, MAE, R² and bias, overall and by region and season.
- Fix random seeds; save configs and metrics for every run in results/.

## Repo layout
- data/raw: ADS downloads, never modified. data/processed: derived datasets.
- src/pm25/: reusable code. scripts/: entry points. notebooks/: exploration only.
- tests/: pytest. results/: figures, metrics, models.

## Environment
- Conda env `pm25` (environment.yml). Run Python with `conda run -n pm25 python ...`.
- New packages go into environment.yml; don't pip install ad hoc.
- ADS credentials live in ~/.cdsapirc. Never put keys in code.

## Conventions
- xarray for gridded data; dims named time, lat, lon.
- ruff for lint/format, type hints, small documented functions.
- Run pytest before declaring a task done.
- Never commit data, model weights or credentials.
