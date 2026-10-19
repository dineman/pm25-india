```{=typst}
#pagebreak()
#align(center + horizon)[#text(24pt, weight: "bold")[Part III] #v(0.3em) #text(18pt)[The code, file by file]]
#pagebreak()
```

# How the repository is organised

```text
pm25-india/
├── CLAUDE.md            rules for working on the project (methods, logging)
├── README.md            one-line description
├── environment.yml      every package needed (conda), incl. this project itself
├── pyproject.toml       makes src/pm25 installable as the package "pm25"
├── src/pm25/            the LIBRARY: reusable functions, no file paths, no plots
├── scripts/             the STEPS: one entry point per pipeline step
├── tests/               automatic checks of the library (pytest)
├── docs/
│   ├── report.md        the project report
│   ├── booklet/         this booklet
│   └── steps/           one document per step, journal, glossary
├── data/                raw downloads and processed data (NOT in git)
├── results/             tables, figures, configs and run logs per step
└── logs/sessions/       archived working sessions (NOT in git)
```

**Library vs scripts.** The code is split in two on purpose:

- `src/pm25/` holds **small, documented functions** that do one thing each
  (e.g. "compute inverse-distance weights"). They don't know about file paths or
  figures, so they are easy to test and reuse.
- `scripts/` holds one **entry point per step** (e.g. `run_baselines.py`). A
  script reads the inputs, calls library functions, writes results and figures,
  and records what it did.

**How every script is built.** All scripts follow the same pattern:

```python
def main() -> None:
    args = parse_args()                       # 1. read command-line options
    setup_logging(log_file=Path("results/logs/run_baselines.log"))
    log_run_info(args)                        # 2. record command, git version, settings
    ...                                       # 3. load data, compute, score
    result.to_csv(args.out_dir / "metrics_overall.csv")   # 4. save tables
    (args.out_dir / "config.json").write_text(json.dumps(config))  # 5. save settings

if __name__ == "__main__":                    # run main() only when started as a script
    main()
```

- `parse_args()` uses Python's **argparse**, so every setting can be changed on the
  command line (`--seed 42`, `--out-dir …`) and has a sensible default.
- Every run **appends to a log file** in `results/logs/`: the exact command, the
  git version of the code, every setting and the main results. Together with the
  `config.json` written next to each step's results, this makes every number
  traceable to the code and settings that produced it.
- **Random seeds** are fixed (default 42), so random choices (samples, model
  settings) are the same every time.

**The shape of the data.** Two layouts appear everywhere:

1. **Gridded**: an *xarray* dataset with dimensions `(time, lat, lon)`, e.g.
   PM2.5 has shape 14,616 × 43 × 40.
2. **Table of rows**: for ML, each row is one (time step, cell) pair and each
   column one feature. Rows are **time-major**: all cells of time step 0, then all
   cells of time step 1, and so on. Keeping this order identical everywhere is what
   lets predictions be reshaped back into `(time, cell)` arrays.

# The library: `src/pm25/`

Each section names the file, says what it is for, lists its main functions, and
shows a short piece of the real code with an explanation.

## `download.py`: requesting data from Copernicus

Builds one request per month and downloads it safely. Main functions:
`iter_months` (all months between two dates), `build_request` (the request for one
month), `month_path` (file name such as `eac4_india_2020-01.nc`), `download_month`.

```python
def download_month(client, out_dir, year, month):
    target = month_path(out_dir, year, month)
    if target.exists():                       # already done: skip it
        logger.info("Skipping %s (already downloaded)", target.name)
        return target
    partial = target.with_name(target.name + ".part")
    client.retrieve(DATASET, build_request(year, month), str(partial))
    partial.rename(target)                    # only a complete file gets the real name
    return target
```

The `.part` trick means a crash in the middle of a download leaves a `.part` file
that the next run ignores and replaces, never a broken file with the final name.

## `regions.py`: India mask and regions

`load_india_states` reads state outlines from Natural Earth; `cell_box` makes the
square of one grid cell; `build_region_mask` computes, for every cell, the share
of its area inside India, the main state and the region; `check_state_names`
stops the program if a state has no region (so a spelling difference can't go
unnoticed).

```python
for i, la in enumerate(lat):
    for j, lo in enumerate(lon):
        cell = cell_box(float(la), float(lo))
        overlaps = {names[k]: cell.intersection(states[names[k]]).area / cell.area
                    for k in tree.query(cell)}       # only states near this cell
        ...
in_india = fraction >= min_fraction                  # the 50% rule
```

`tree.query` uses an *STRtree*, a spatial index that quickly finds which state
shapes are near a cell, instead of testing all 36 states for all 1,720 cells.

## `preprocess.py`: units and derived variables

`standardize` (names and order), `convert_units`, `relative_humidity`,
`add_met_features` (wind speed, humidity), `add_time_features` (solar hour, day of
year, season), and `preprocess`, which runs them all in order.

```python
utc_hour = ds["time"].dt.hour + ds["time"].dt.minute / 60.0
solar_hour = (utc_hour + ds["lon"] / 15.0) % 24.0       # the Sun moves 15° per hour
out["hour_sin"] = np.sin(2 * np.pi * solar_hour / 24.0)
out["hour_cos"] = np.cos(2 * np.pi * solar_hour / 24.0)
```

`% 24.0` wraps the hour around midnight. The sine/cosine pair puts the 24 hours on
a circle, so 23:00 and 01:00 end up close together. `hour_cos` is +1 at local
midnight and −1 at local noon.

## `exploration.py`: summaries, correlations, variograms

Functions for step 04: `india_cells`, `to_table`, `summarize`,
`spearman_with_target`, `haversine_km` (distance between two points on the Earth's
surface), `standardize_rows`, `empirical_variogram`, the variogram models
`spherical` and `exponential`, `fit_variogram`, `linear_residuals` (remove what the
predictors explain) and `block_size_cells` (block size for a given range).

```python
def exponential(h, nugget, psill, rng):
    return nugget + psill * (1.0 - np.exp(-3.0 * h / rng))
```

This is the curve fitted to the variogram: at distance `h = 0` it starts at the
*nugget* (variation even between very close cells), and it rises towards
`nugget + psill` (the *sill*), reaching 95% of it at the *range* `rng`.
`empirical_variogram` computes, for all pairs of cells, half the average squared
difference in each distance bin, processing 32 time steps at a time to keep memory
use small.

## `splits.py`: blocks, test set and folds

`assign_blocks` gives each cell a block number from its grid position;
`region_counts` counts cells per region in each block; `pick_test_blocks` chooses
the test blocks; `assign_folds` deals training blocks into 5 balanced folds;
`make_split` does everything for one block size; `split_summary` reports the result.

```python
for _ in range(n_candidates):                       # try 5,000 random orders
    order = rng.permutation(len(cells))
    cumulative = np.cumsum(per_block[order])
    k = int(np.argmin(np.abs(cumulative / total - test_fraction))) + 1
    chosen = order[:k]                              # first k blocks ≈ 20% of cells
    share = cells[chosen].sum(axis=0) / region_total
    score = np.abs(share - test_fraction).mean()    # how far each region is from 20%
    ...
    if score < best_score:
        best_score, best = score, chosen
```

A simple but effective search: shuffle the blocks, take as many as needed to reach
about 20% of all cells, score how evenly the regions are covered, and keep the best
of 5,000 tries. Because `rng` is seeded, the same split comes out every time.

## `features.py`: from grids to tables

`stack_cells` turns `(lat, lon)` into one `cell` dimension and keeps only the cells
we want; `feature_matrix` builds the `(rows, features)` array; `target_vector` gives
PM2.5 in the same row order. `FEATURE_SETS` defines the three feature lists used in
step 07 (`brief`, `all`, `all_no_latlon`).

```python
FEATURE_SETS = {
    "brief": BRIEF,                       # aod550, u10, v10, t2m, d2m, sp, lat, lon
    "all": BRIEF + DERIVED,               # + wind_speed, rh2m, hour_sin/cos, doy_sin/cos
    "all_no_latlon": [f for f in BRIEF + DERIVED if f not in ("lat", "lon")],
}
```

## `metrics.py`: how every method is scored

`scores` computes RMSE, MAE, R², bias and the number of values; `scores_by_group`
does it per region, season, etc.; `evaluation_table` builds the long table of
(time, test cell) rows with all predictions; `all_scores` runs every grouping.

```python
err = pred - obs
sst = np.sum((obs - obs.mean()) ** 2)          # variation around the mean
return {
    "rmse": float(np.sqrt(np.mean(err**2))),
    "mae": float(np.mean(np.abs(err))),
    "r2": float(1.0 - np.sum(err**2) / sst),   # share of the variation explained
    "bias": float(np.mean(err)),
}
```

Because every step uses this one function, numbers from different steps are
directly comparable.

## `baselines.py`: interpolation as weight matrices

`nearest_weights`, `idw_weights`, `delaunay_weights` and
`ordinary_kriging_weights` each return a **weight matrix** W with one row per test
cell and one column per training cell; `apply_weights` multiplies it with the data.

```python
def apply_weights(weights, values_train):
    return values_train @ weights.T        # (time, n_train) @ (n_train, n_test)
```

The key insight: training and test cells are in the same places at every time
step, so the weights only need to be computed **once**. Predicting all 14,616 time
steps is then a single matrix multiplication (`@`), which takes seconds even for
kriging. IDW sorts neighbours with a *stable* sort so that ties (common on a
regular grid) are broken the same way on every computer (see chapter 24).

## `models.py`: the ML models and tuning

`build_estimator` creates ridge, random forest, LightGBM or MLP;
`ClippedTargetModel` wraps any of them; `make_model` combines the two;
`SEARCH_SPACES` lists the settings to try; `sample_params` draws random settings;
`fold_indices` and `cross_validate` run block-wise cross-validation;
`best_settings` picks the winner per model.

```python
class ClippedTargetModel(RegressorMixin, BaseEstimator):
    def fit(self, x, y):
        self.y_max_ = float(np.max(y))
        y_fit = np.log1p(y) if self.target == "log" else y      # optional log target
        self.estimator_ = clone(self.estimator).fit(x, y_fit)
        return self

    def predict(self, x):
        pred = self.estimator_.predict(x)
        if self.target == "log":
            pred = np.expm1(np.clip(pred, None, np.log1p(self.y_max_)))  # undo the log
        return np.clip(pred, 0.0, self.y_max_)                  # never < 0 or > max seen
```

This wrapper gives every model the same two safety features: optionally learning
log(1 + PM2.5), and clipping predictions to between 0 and the largest value seen in
training (the lesson from the 55,041 µg m⁻³ prediction in step 06).

```python
def cross_validate(name, params, x, y, folds, seed):
    per_fold = []
    for _, tr, va in fold_indices(folds):            # each fold = whole blocks
        model = make_model(name, params, seed).fit(x[tr], y[tr])
        per_fold.append(scores(y[va], model.predict(x[va])))
    ...                                              # average over the 5 folds
```

## `hybrid.py`: regression kriging

`out_of_fold_predictions` predicts every training row with a model that did not
see its fold; `add_residual_correction` adds interpolated residuals to the ML
prediction.

```python
def add_residual_correction(ml_test, residual_train, weights):
    return np.clip(ml_test + residual_train @ weights.T, 0.0, None)
```

One line holds the whole hybrid idea: the ML prediction plus the weighted average of
nearby errors, clipped so PM2.5 can't become negative.

## `evaluation.py`: bootstrap and CV check

`ist_hour` (Indian time labels), `aqi_level` (India's air-quality categories),
`add_time_columns`, `block_sums`, `scores_from_sums`, `block_bootstrap`,
`cv_interpolate` (predict each fold from the others) and `nested_oof_residuals`.

```python
draws = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))   # 2,000 × 6 picks
counts = np.stack([np.bincount(d, minlength=len(blocks)) for d in draws])
boot = scores_from_sums(np.einsum("rb,bms->rms", counts, cube))
```

A clever shortcut: instead of re-scoring millions of rows 2,000 times, the code
first stores, for each block, a few **sums** (number of values, sum of errors, sum
of squared errors, …). A bootstrap sample is then just "block 3 twice, block 5 not
at all, …", i.e. a weighted sum of those block sums, from which RMSE, MAE, R² and
bias follow exactly. All 2,000 samples take about a second.

## `importance.py`: SHAP and permutation importance

`FEATURE_GROUPS` lists related variables; `shap_values` asks LightGBM for exact
SHAP values; `group_shap` adds them up per group; `mean_abs_by` averages their size;
`permutation_importance` shuffles columns and measures the damage;
`binned_dependence` makes the "effect vs value" curves.

```python
shuffled = x.copy()
perm = rng.permutation(len(x))
shuffled[:, idx] = x[perm][:, idx]       # same shuffle for all columns of a group
s = scores(y, model.predict(shuffled))
```

## `maps.py`: the 0.25° maps

`load_india_official` loads India's official boundary; `fine_axis` builds the fine
coordinate axis; `points_inside` keeps grid points inside a shape;
`predictors_at_points` interpolates PM2.5 and the predictors to the fine points and
derives the extra variables with the **same code as step 03**;
`coarse_membership` and `aggregate_to_coarse` average fine values back to the
0.75° cells for the consistency check.

```python
points = {"lat": xr.DataArray(lat, dims="cell"), "lon": xr.DataArray(lon, dims="cell")}
out = ds[["pm2p5", *BASE_PREDICTORS]].interp(points, method="linear")
out = add_met_features(out)              # wind speed, humidity (step 03 code)
return add_time_features(out)            # solar hour from each point's longitude
```

## `logging_utils.py`: recording every run

`setup_logging` sends messages to the screen and to a log file; `git_version`
finds the current code version (marked `-dirty` if there are unsaved changes);
`log_run_info` writes the run header.

```python
logger.info("Run started: %s", " ".join(sys.argv))
logger.info("Git version: %s | Python %s", git_version(), sys.version.split()[0])
for name, value in sorted(vars(args).items()):
    logger.info("  %s = %s", name, value)
```

# The scripts: `scripts/`

Each script is one step of the pipeline. All are run the same way:
`conda run -n pm25 python scripts/<name>.py`, from the repository root.

- **`download_eac4.py`** (step 01): downloads every month (skips finished ones), lists failures. Output: `data/raw/eac4/*.nc`.
- **`build_region_mask.py`** (step 02): India mask, states, regions; draws the region map. Output: `data/processed/india_regions.nc`, `results/figures/region_mask.png`.
- **`preprocess_eac4.py`** (step 03): joins 60 months, converts units, adds derived variables. Output: `data/processed/eac4_india_3h.nc`.
- **`explore_eac4.py`** (step 04): summaries, correlations, seasonal maps, daily/monthly cycles, variograms. Output: `results/04_exploration/`.
- **`make_splits.py`** (step 05): 5 × 5 and 8 × 8 block splits, folds, distance to training, maps. Output: `data/processed/splits/`, `results/05_splits/`.
- **`run_baselines.py`** (step 06): nearest, IDW, Delaunay, kriging, linear regression; scores and figures. Output: `results/06_baselines/`.
- **`train_models.py`** (step 07): tunes and trains ridge, RF, LightGBM, MLP; scores on test cells. Output: `results/07_models/<feature set>/`.
- **`run_hybrid.py`** (step 08): out-of-fold residuals, residual variograms, regression kriging. Output: `results/08_hybrid/`.
- **`evaluate.py`** (step 09): all methods together, new breakdowns, block bootstrap, nested CV check. Output: `results/09_evaluation/`.
- **`feature_importance.py`** (step 10): SHAP and permutation importance, heatmaps, dependence curves. Output: `results/10_importance/`.
- **`make_maps.py`** (step 11): 0.25° maps for 4 methods, examples, zoom, consistency check. Output: `results/11_maps/`.
- **`report_figures.py`** (step 12): rebuilds the report figures from saved results (no model runs). Output: `results/12_report/`.
- **`export_session.py`** (tool): archives each working session as raw + readable Markdown. Output: `logs/sessions/`.

A few details worth knowing:

- **`train_models.py`** has two shortcuts: `--cv-from` reuses tuning results
  already saved (so only the final fits run, ~4 minutes instead of ~40), and
  `--params-from` reuses chosen settings for another feature set.
- **`run_hybrid.py`** and **`evaluate.py`** *cache* their slowest part (the
  out-of-fold and nested model fits) in `.nc` files, so a re-run skips it.
- **`make_maps.py`** works **month by month** and only keeps running sums, so the
  millions of fine-grid predictions never need to be in memory at once.
- **`report_figures.py`** imports plotting functions from `feature_importance.py`
  and `make_maps.py` instead of copying them, so a figure looks the same whichever
  script draws it.
- **`export_session.py`** is started automatically when a working session ends (a
  *hook* in `.claude/settings.json`) and uses only Python's standard library, so it
  works even without the conda environment.

# The tests: `tests/`

A **test** is a small program that runs a library function on a tiny example
whose correct answer is known, and fails loudly if the answer is wrong. Running
`conda run -n pm25 python -m pytest` executes all **73 tests** in a few seconds.
Tests were run before every commit, and whenever a bug was found a test was added
so that the same mistake can't come back unnoticed.

| Test file | Checks, for example, that … |
|---|---|
| `test_download.py` | month ranges cross year boundaries; leap days are included; finished files are skipped (with a fake client, no internet) |
| `test_regions.py` | the 50% rule and "largest state wins" work; an unknown state name stops the program |
| `test_preprocess.py` | units are converted; humidity is lower for drier air; solar hour shifts with longitude; IMD seasons are right |
| `test_exploration.py` | 1° of latitude is ~111 km; a constant field has a zero variogram; a known range is recovered by the fit |
| `test_splits.py` | a block is never split between train and test or between folds; every region gets test cells |
| `test_features.py` | table rows are in time-major order and line up with PM2.5 |
| `test_metrics.py` | RMSE, MAE, R² and bias give known values; missing values are ignored |
| `test_baselines.py` | IDW weights sum to 1; Delaunay reproduces a linear field; our kriging matches the PyKrige library; distance ties go to the lower index |
| `test_models.py` | predictions stay in range; the log target is undone correctly; folds cover every row once; every model fits |
| `test_hybrid.py` | a fold is never predicted by a model that saw it; the correction is added and clipped |
| `test_evaluation.py` | bootstrap sums reproduce the plain scores; nested residuals never use the outer fold; IST is UTC + 5:30 |
| `test_importance.py` | SHAP values add up to the model's prediction; the informative feature ranks first |
| `test_maps.py` | interpolation and derived variables at points; aggregation back to coarse cells |
| `test_logging_utils.py` | logs record the settings and go to the file |
| `test_export_session.py` | a session is archived as raw and readable files |

A good example of a test that guards a subtle rule:

```python
def test_nested_residuals_never_use_the_outer_fold():
    # 3 folds, one cell each, a constant PM2.5 per fold. For outer fold k, the
    # residual at fold j must come from a model fitted only on the remaining fold.
    ...
    assert np.isnan(resid[0][:, 0]).all()               # the outer fold itself: empty
    assert np.allclose(resid[0][:, 1], 20.0 - 60.0)     # fitted on fold 2 only
```

The data is made so simple that the right answer can be worked out by hand: if
any information from the held-out fold leaked into the model, these exact numbers
would change and the test would fail.

# Documentation and record keeping

The project keeps four kinds of records, and all of them are updated **while**
working, not afterwards:

1. **Step documents** (`docs/steps/NN_*.md`): one per step, always with the same
   sections: purpose, idea in plain words, how it works, outputs, how to run,
   decisions (with reasons), problems met, results, open issues.
2. **The journal** (`docs/steps/journal.md`): a dated timeline of every action,
   decision, problem and commit, with real times taken from the logs.
3. **Run logs** (`results/logs/*.log`): what each script printed, with the exact
   command, code version and settings.
4. **Session archives** (`logs/sessions/`): the full working conversations.

**git** records every change to the code and documents. After each step the work
is *committed* (saved as a named version) and *pushed* to GitHub, followed by a
small commit that adds the commit itself to the journal. Data files, model files
and large prediction files are kept out of git on purpose (they are big and can be
regenerated); the `.gitignore` file lists them.
