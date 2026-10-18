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

## Logging and documentation (keep up to date as you work, not at the end)
- Work is done on two machines (Fedora and a Mac); this repo on GitHub is the shared
  record. `git pull` before starting, push after each step.
- `docs/steps/journal.md`: dated timeline, newest at the bottom. One table per session
  (`Session <id8>` line, then Time | What | Where rows). Add a row whenever a step is
  run, a decision is made, something goes wrong, or a commit is pushed. Mark decisions
  in **bold**, use real times from log timestamps, and keep the "Still open" line current.
- One doc per step, `docs/steps/NN_name.md`: Status, Purpose, plain-words idea, How it
  works, Outputs, How to run, Decisions, Problems met, Results, Open issues. Explain for
  a non-expert; new terms go in `docs/steps/glossary.md` and appear in *italics*.
  Update `docs/steps/00_overview.md` (status table) when a step's status changes.
- Every script logs via `pm25.logging_utils` (`setup_logging`, `log_run_info`) to
  `results/logs/<script>.log`.
- After each step: commit code, tests, docs and small results, push, then add the
  commit to the journal (separate "Journal: record ..." commit).
- Sessions are archived to `logs/sessions/` by the SessionEnd hook; if a session was
  started outside this folder the hook doesn't run, so export it by hand with
  `scripts/export_session.py --transcript <path>`.

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
