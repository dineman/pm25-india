```{=typst}
#pagebreak()
#align(center + horizon)[#text(24pt, weight: "bold")[Part IV] #v(0.3em) #text(18pt)[Using and judging the project]]
#pagebreak()
```

# Running the project yourself

## What you need

- A computer with **conda** (Miniforge is a good choice) and about 5 GB of free
  disk space and 16 GB of memory.
- A free **Copernicus ADS account**. Its API key goes into a file called
  `~/.cdsapirc` in your home folder. The key is personal: it must never be put
  into the code or into git.
- **git**, to get the code.

## Setting up

```bash
git clone git@github.com:dineman/pm25-india.git
cd pm25-india
conda env create -f environment.yml       # creates the "pm25" environment
conda run -n pm25 python -m pytest        # all 73 tests should pass
```

The environment also installs the project's own code (`src/pm25`) as a package,
so every script can simply `import pm25`.

## Running the steps

Run from the repository folder, in this order (times measured on a 16 GB Mac):

| Step | Command (each starts with `conda run -n pm25 python`) | Time |
|---|---|---|
| 01 | `scripts/download_eac4.py` | ~6 h (server queue) |
| 02 | `scripts/build_region_mask.py` | seconds |
| 03 | `scripts/preprocess_eac4.py` | ~1 min |
| 04 | `scripts/explore_eac4.py` | ~30 s |
| 05 | `scripts/make_splits.py` | seconds |
| 06 | `scripts/run_baselines.py` | ~15 s |
| 07 | `scripts/train_models.py --n-iter 8 --cv-time-frac 0.05` | ~40 min |
| 08 | `scripts/run_hybrid.py` | ~9 min |
| 09 | `scripts/evaluate.py` | ~25 min |
| 10 | `scripts/feature_importance.py` | ~20 min |
| 11 | `scripts/make_maps.py` | ~11 min |
| 12 | `scripts/report_figures.py` | ~10 s |

Tips:

- **Try small first.** Step 01 can download a single month
  (`--start 2020-01 --end 2020-01`), and step 11 can run on a few months
  (`--max-months 3 --out-dir /tmp/test`).
- **Long runs need a computer that stays awake.** On a laptop, keep it plugged in
  (chapter 24 tells why).
- **Moving to another computer**: data and model files are not in git. Either copy
  the `data/` folder (about 1 GB) and re-run steps 06–11, or run everything again.

## Where to find things afterwards

- Numbers: CSV tables in `results/<step>/`.
- Figures: PNG files in the same folders.
- What happened in a run: `results/logs/<script>.log`.
- Why things were done this way: `docs/steps/<step>.md` and the journal.

# Mistakes we made and what they taught us

Real projects contain mistakes. What matters is finding them, fixing them and
recording them honestly. These are the most instructive ones from this project,
all documented in the journal.

**1. A log-transformed straight line exploded (step 06).** Linear regression on
log(PM2.5), transformed back with the exponential, predicted 55,041 µg m⁻³ for
one hour. *Lesson:* a model can extrapolate wildly outside the data it has seen.
*Fix:* every later model clips its predictions to [0, largest value in training].

**2. A first test split was unbalanced (step 05).** The first version put 49% of
the Himalaya into the test set. *Lesson:* check the result of every random
procedure, not just that it ran. *Fix:* a search that balances the test share of
every region.

**3. The wrong "best" settings (step 07, on Fedora).** When the tuning tables of
the four models were joined, each kept its own row numbers 0–7; looking up "the
best row" then returned rows from the wrong model (LightGBM got a setting scoring
40.9 instead of 36.7). The final models trained in that run were invalid.
*Lesson:* an innocent-looking table operation can silently mix things up.
*Fix:* a dedicated `best_settings` function, a test for it, and a re-run.

**4. The same method gave different numbers on two computers (step 09).** IDW has
no randomness, yet its error was 55.31 on Fedora and 55.16 on the Mac. On a regular
grid, the 8th and 9th nearest cells are often exactly equally far away; the sorting
routine broke such ties differently on each machine. *Fix:* a *stable* sort, which
always keeps the earlier cell, plus a test.

**5. Neural networks are not perfectly reproducible.** The MLP scored R² 0.545 on
Fedora and 0.528 on the Mac with the same seed, because the underlying number
libraries differ. Tree models and interpolation matched exactly. *Lesson:* a fixed
seed guarantees the same result on the same machine, not on every machine. The
report states which machine produced its numbers.

**6. A promising result was partly luck (step 09).** The hybrid improved R² from
0.59 to 0.72 on the 6 test blocks, but checking on 27 blocks showed the gain came
mostly from 2 blocks and is small on average. *Lesson:* with few test areas,
always estimate uncertainty and check on more data before believing a big gain.

**7. The hybrid map double-counted (step 11).** The first version added the
*out-of-fold* errors from step 08 to the *final* model. But the final model has
already seen the training cells and fits them well, so adding errors from models
that had not seen them over-corrected: pale patches appeared in south India.
*Fix:* for a map, the errors must come from the same model that draws the map.
A three-month trial run caught this before the full run.

**8. The computer fell asleep (step 09).** On battery, the Mac went to sleep
during the night; 25 minutes of work took 8 hours. *Lesson:* plan long runs on
mains power, and always do a short trial run first.

# What the results can and cannot tell us

- **CAMS is not the real world.** Every "accuracy" here is agreement with a
  computer model of the atmosphere. CAMS is known to overestimate PM2.5 over the
  Indo-Gangetic Plain. Real ground measurements would be needed to judge true
  accuracy.
- **AOD and PM2.5 come from the same model.** In CAMS both are computed from the
  same simulated particles, so their strong link (and AOD's top importance) is
  partly built in. With satellite AOD and station PM2.5, the task would be harder.
- **Few test areas.** Six test blocks, mostly one per region, make per-region test
  results anecdotal. The 27-block check is more reliable, but it slightly favours
  LightGBM, whose settings were tuned on the same folds.
- **Finer maps can't be checked.** There is no data below 0.75° (80 km) to test
  the 0.25° maps against; their extra detail comes from what the model learned,
  not from new measurements. Real enhancement would need fine-resolution inputs
  (terrain height, satellite AOD at 1–10 km, emission maps) and fine-scale
  reference data.
- **What is left out.** Cells less than half inside India (including Mumbai and
  Chennai) and the island territories are not in the training or test data. On
  the maps, which follow India's official boundary, areas outside the model's
  domain are hatched.
- **Importance is not cause.** The feature-importance results show what the model
  does with the data it was given, not what causes pollution.

Within these limits, the project gives a careful, honest answer to its six
questions, and every number in it can be traced back to the code and settings that
produced it.
