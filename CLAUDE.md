# Project: PM2.5 spatial enhancement over India

## Context
- Data: CAMS EAC4 PM2.5, 2020–2024, India domain.
- Raw downloads go in data/raw/ and are never modified. Derived data goes in data/processed/.

## Environment
- Conda env `pm25`, defined in environment.yml. Run Python via `conda run -n pm25 python ...`.
- If a new package is needed, add it to environment.yml (don't pip install ad hoc).

## Code conventions
- Reusable code in src/pm25/, runnable entry points in scripts/, exploration in notebooks/.
- Use xarray for gridded data; keep dimension names consistent (time, lat, lon).
- Format and lint with ruff. Use type hints and small, documented functions.
- Run `pytest` before declaring a task done.

## Never
- Commit data files, model weights, or credentials (~/.cdsapirc).
- Shuffle time series randomly across train/test splits (use time-based splits).
