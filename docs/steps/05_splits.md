# Step 05: Spatial block splits

**Status:** done (2026-10-18). Main split = **5 × 5-cell blocks**; 8 × 8 built as a
sensitivity split (weak, see below). **Must be re-run if the India mask changes**
(Mumbai/Chennai decision in step 02). Final run 2026-10-18 17:57 with the full
2020–2024 data for the PM2.5 balance check; the split itself is identical to the
17:10 run (it depends only on the mask).

## Purpose

Decide which grid cells the models may learn from (**train**) and which are kept
back for the final score (**test**), so that RQ5 ("how well does the model work
at locations not used in training?") is answered honestly.

Neighbouring cells are strongly correlated (step 04: at 75 km only ~30% of the
local variance is new). If test cells were picked one by one at random, almost
every test cell would have a training cell right next to it, and the model
(or plain interpolation) would look much better than it really is. So whole
**blocks** of cells go to test together.

## How it works

Code: `src/pm25/splits.py` (tested functions) and `scripts/make_splits.py`.
Input: only the India mask (`data/processed/india_regions.nc`), 494 cells.

1. **Blocks.** The grid is cut into squares of N × N cells, aligned to the
   south-west corner of the domain (6.0 °N, 68.25 °E). Block id =
   (lat index ÷ N) × (blocks per row) + lon index ÷ N. Blocks at the coast or
   border contain fewer India cells.
2. **Test blocks** (target 20% of cells, seed 42). Blocks often span two regions,
   so balance is measured on **cells per region**, not on a block label.
   5,000 random candidates are tried. Each shuffles the blocks and takes the
   first k, with k chosen so the total test share is closest to 20%. Score =
   mean over regions of |region test share − 20%| + |total share − 20%|, plus a
   penalty of 1 if any region has no test cells. The lowest score wins.
3. **CV folds** (5, for tuning in step 07). Training blocks, largest first, are
   each put into the fold with the fewest cells so far, so folds have nearly
   equal size. A whole block is always in one fold.
4. **Distance to the nearest training cell** (great-circle) is stored for every
   test cell. Step 09 uses it to show how error grows with distance.
5. **Check:** time-mean PM2.5 of train vs test cells per region (needs the
   preprocessed data; skipped if not present).

### First version (rejected, kept here for the record)

The first version picked blocks per region using each block's majority region,
and forced ≥ 1 test block per region. Result: 5 × 5 gave Himalaya 49% test and
central 14%; 8 × 8 gave **50% test overall** (16 blocks, 6 regions each forcing a
block) and very uneven folds (28–74 cells). Replaced by the cell-based search above.

## Outputs

| File | Content |
|---|---|
| `data/processed/splits/split_block5.nc`, `split_block8.nc` | on the (lat, lon) grid: `block` (−1 outside India), `set` ("train"/"test"/""), `fold` (0–4; −1 for test/outside), `dist_to_train_km`; attribute `block_size_cells`. **Later steps read these.** |
| `results/05_splits/cells_block{5,8}.csv` | one row per India cell: indices, lat, lon, region, block, set, fold, distance |
| `results/05_splits/summary_block{5,8}.csv` | cells and blocks per region and set, test share |
| `results/05_splits/pm25_balance_block{5,8}.csv` | mean PM2.5 train vs test per region |
| `results/05_splits/split_block{5,8}.png` | maps: train/test with block outlines; CV folds with fold numbers |
| `results/05_splits/summary.json` | config (all arguments) and key numbers per split |

Log: `results/logs/make_splits.log`. Tests: `tests/test_splits.py` (blocks never
split between sets or folds, every region has test cells near the target share,
folds only for training cells and balanced, distance 0 for train and ≥ 1 cell for
test, same seed → same split).

## How to run

```bash
conda run -n pm25 python scripts/make_splits.py                      # 5 and 8, full data check (final run)
conda run -n pm25 python scripts/make_splits.py \
    --data data/processed/eac4_india_3h_2020_2021.nc                 # earlier test run
```

Runtime ~6 s with the full-data balance check.

## Decisions

| Decision | Reason |
|---|---|
| **5 × 5 blocks (~380 km) as the main split** | Step 04: strict "block ≥ range (846 km)" gives 12 × 12 blocks = only 11 blocks, so a 20% test set would be 2 blocks and couldn't cover the 6 regions. At ~380 km, 75–80% of the local correlation is gone, and there are 33 blocks. Agreed with the user 2026-10-18. |
| 8 × 8 (~600 km) as a sensitivity split | To see how scores change when test cells are further from training cells (RQ5). |
| Balance regions on cell counts | Blocks span regions; evaluation is per region by cell. |
| 20% test, 5 CV folds, seed 42 | Usual sizes; seed fixed for reproducibility (CLAUDE.md). |
| No buffer zone | A buffer (dropping training cells next to test blocks) would remove ~25% more training cells. Distance-to-train is stored instead, so the effect of nearby training cells can be measured in step 09. |
| Split depends only on the mask, not on the data | Same split for every model, every year range, every run. |

## Results

### 5 × 5 (main)

33 blocks with India cells. **Test: 95 cells (19.2%) in 6 blocks.** Training: 399
cells in 27 blocks; CV fold sizes 81, 80, 79, 79, 80 cells.

| region | test cells | train cells | test share | mean PM2.5 test | mean PM2.5 train |
|---|---|---|---|---|---|
| central | 25 | 125 | 16.7% | 57.9 | 61.1 |
| himalaya | 9 | 28 | 24.3% | 61.6 | 32.5 |
| igp | 15 | 69 | 17.9% | 156.6 | 139.7 |
| northeast | 8 | 35 | 18.6% | 60.0 | 48.2 |
| northwest | 16 | 67 | 19.3% | 55.1 | 79.9 |
| south | 22 | 75 | 22.7% | 53.8 | 47.1 |
| **all** | **95** | **399** | **19.2%** | **72.6** | **72.1** |

(PM2.5 = 2020–2024 time mean, µg m⁻³; the 2020–2021 test run gave 71.6 vs 70.9 overall.)

Test-cell distance to the nearest training cell: median **83 km**, max 283 km.
49 test cells ≤ 100 km, 37 in 100–200 km, 9 in 200–300 km.

Test blocks: one in the western Himalaya/Punjab foothills, one in central India,
one at the southern tip, the Gujarat coast (Kutch), and two small pieces in the
northeast. See `split_block5.png`.

### 8 × 8 (sensitivity)

16 blocks. **Test: 92 cells (18.6%) in 7 blocks.** Fold sizes 66, 64, 91, 92, 89.
Test-cell distance: median 152 km, max 417 km (12 cells > 300 km).

**Weaknesses:** Himalaya gets only **1** test cell, central 10%. Almost all test
cells are in **partial blocks on the coast or in the northeast**: the big inland
blocks all stayed in training, because with only 16 blocks, small edge pieces
are the only way to get near 20% per region. Overall test mean PM2.5 is 11 µg m⁻³
higher than train (81.5 vs 70.1, 2020–2024). Use this split only as a rough check on distance, not for
per-region results.

## Checks

- Overall PM2.5 balance for 5 × 5 is good (72.6 vs 72.1 µg m⁻³, 2020–2024).
- **Within regions, test and train differ**, especially Himalaya (test 62 vs train
  33: the test block is the polluted Punjab foothills, not the high mountains) and
  northwest (test 55 vs 80: Kutch coast vs Thar desert). With 1–2 test blocks per
  region this is unavoidable. **Per-region test scores therefore describe those
  particular blocks**, not the whole region. Say so in the write-up.

## Open issues

- **Re-run if the Mumbai/Chennai mask decision changes the cell set.**
- Per-region results rest on 1–2 test blocks each. Option to reduce this luck of
  the draw: repeat the 5 × 5 split with several seeds (e.g. 5) and report the
  spread of scores. Decide in step 09.
- 8 × 8 is a weak sensitivity test (see above). Alternatives: drop it and use
  error vs distance-to-train (step 09) instead, or add a buffer around 5 × 5 test blocks.
- ~~Re-run the PM2.5 balance check with the full data~~ done 17:57.
