# Step 02: India land mask and region labels

**Status:** done (2026-10-18). **Open issue:** the Mumbai and Chennai cells are
excluded by the 50% rule (see below); decide before step 05.

## Purpose

1. Decide which 0.75° grid cells count as **India**: only these are used for
   training and evaluation.
2. Label each India cell with its **state** and one of **6 regions**, so results
   can be broken down by region (RQ4) and the spatial split can be balanced by region.

## How it works

1. Load Indian state/UT polygons from **Natural Earth 10 m admin-1**
   (`load_india_states`, cached by cartopy at
   `~/.local/share/cartopy/shapefiles/natural_earth/cultural/`). 36 states/UTs.
2. Check every state name has a region (`check_state_names`). If Natural Earth
   ever spells a name differently, the code **stops with an error** instead of
   silently dropping that state.
3. For each grid cell (centre ± 0.375°), intersect the cell square with every
   state polygon (`build_region_mask`):
   - `india_fraction` = share of the cell's area inside any Indian state (0–1);
   - `state` = the state covering the **largest** share of the cell.
4. A cell is **in India if `india_fraction ≥ 0.5`**. Cells below that get empty
   state and region.
5. `region` = the state's region from `REGION_BY_STATE` in `src/pm25/regions.py`.

Areas are computed in degrees, not km². Within one 0.75° cell the distortion is
uniform, so the *fraction* is accurate enough.

## Regions

| Region | States/UTs |
|---|---|
| `igp` (Indo-Gangetic Plain) | Punjab, Haryana, Delhi, Chandigarh, Uttar Pradesh, Bihar, West Bengal |
| `northwest` | Rajasthan, Gujarat, Dadra & Nagar Haveli and Daman & Diu |
| `central` | Madhya Pradesh, Chhattisgarh, Jharkhand, Odisha, Maharashtra |
| `south` | Andhra Pradesh, Telangana, Karnataka, Goa, Kerala, Tamil Nadu, Puducherry, Lakshadweep, Andaman & Nicobar |
| `northeast` | Assam, Arunachal Pradesh, Meghalaya, Manipur, Mizoram, Nagaland, Tripura, Sikkim |
| `himalaya` | Jammu & Kashmir, Ladakh, Himachal Pradesh, Uttarakhand |

## Inputs and outputs

- Input: grid from `data/raw/eac4/eac4_india_2020-01.nc`; Natural Earth shapefile.
- Output: `data/processed/india_regions.nc` with variables on (lat, lon):
  `india_fraction` (float), `in_india` (bool), `state` (str), `region` (str);
  attribute `min_fraction = 0.5`.
- Figure: `results/figures/region_mask.png` (map of regions).
- Log: `results/logs/build_region_mask.log`.

## How to run

```bash
PYTHONPATH=src conda run -n pm25 python scripts/build_region_mask.py            # default threshold 0.5
PYTHONPATH=src conda run -n pm25 python scripts/build_region_mask.py --min-fraction 0.3
```

Takes ~2 s. Code: `src/pm25/regions.py`, `scripts/build_region_mask.py`,
`tests/test_regions.py`.

## Decisions

| Decision | Reason |
|---|---|
| Natural Earth admin-1 | Free, all 36 states/UTs, cartopy downloads it. **Uses the de facto line of control in Kashmir, not India's official boundary.** If the course needs the official boundary, replace the source (e.g. Survey of India). |
| Threshold 50% | Keeps cells that are mostly Indian land; drops cells that are mostly sea or another country. Trade-off: see open issue. |
| State = largest overlap | Each cell needs exactly one label. Small UTs never win a cell (see below). |

## Results and checks

- **494 India cells** out of 1,720 in the box.
- Per region: igp 84, northwest 83, central 150, south 97, **northeast 43,
  himalaya 37** (small regions: the split must make sure both are in the test set).
- 29 states/UTs own at least one cell. **7 own none:** Delhi, Chandigarh, Goa,
  Puducherry, Lakshadweep, Andaman & Nicobar, Dadra & Nagar Haveli and Daman & Diu.
  They are too small, or are islands.
  - **Delhi's cell (28.5°N, 77.25°E) is labelled Haryana**, because Haryana covers
    more of it. The region is still `igp`, so regional results are unaffected; just
    don't look for `state == "Delhi"`.
  - The Andaman & Nicobar and Lakshadweep islands are excluded entirely.
- **127 border/coast cells** overlap India by more than 0% but less than 50%, and
  are excluded.
- Map checked visually: regions follow state borders as expected.
- Major cities:

| City | Cell | India fraction | Included? |
|---|---|---|---|
| Delhi | 28.5, 77.25 | 1.00 | yes (as Haryana, igp) |
| Kolkata | 22.5, 88.5 | 0.99 | yes |
| Bengaluru, Hyderabad, Ahmedabad, Lucknow, Patna, Guwahati, Srinagar | | 1.00 | yes |
| Thiruvananthapuram | 8.25, 77.25 | 0.54 | yes (barely) |
| **Mumbai** | 18.75, 72.75 | **0.30** | **no** |
| **Chennai** | 12.75, 80.25 | **0.47** | **no** |

## Open issues

- **Mumbai and Chennai cells are excluded.** These are coastal megacities, and the
  0.75° cell is mostly sea. Options:
  1. keep 50% (cleaner land signal; say so in the report);
  2. lower the threshold (e.g. 0.25–0.3) to keep coastal cells, at the cost of
     more sea-dominated cells;
  3. keep 50% but add specific cells by hand.

  Must be decided **before step 05** (splits), because it changes the cell set.
- The boundary source (de facto vs official Kashmir boundary) may matter for the
  course. Confirm.
