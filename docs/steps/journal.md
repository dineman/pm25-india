# Project journal

A dated timeline of what was done, decided or changed, newest at the bottom.
Details live in the step docs (`NN_*.md`); every script run is also in
`results/logs/<script>.log`; full Claude conversations are in `logs/sessions/`.
Add an entry whenever a step is run, a decision is made, or something goes wrong.

## 2026-10-18

Session 934f6594 (`logs/sessions/2026-10-18_934f6594.md`)

| Time | What | Where |
|---|---|---|
| 11:45 | Repo created (structure, CLAUDE.md). | commits 8baffaa, 6179db3 |
| ~12:00 | Project summary and 12-step plan agreed. **Decision: native 3-hourly data.** | 00_overview.md |
| 12:07 | One-month download test (Jan 2020) succeeded; variables, units and ranges checked. | 01_download.md |
| ~12:10 | Full download 2020–2024 started (one file per month, resumable). | 01_download.md |
| 12:18 | India mask + 6 regions built; 50% land rule leaves out Mumbai and Chennai cells (**open**). | 02_region_mask.md |
| 12:21 | Commit: download, preprocessing, region mask. | 5b1674a |
| 12:25 | Commit: sessions archived to `logs/sessions/` on session end. | 0c6071f |
| 12:37 | Region mask rebuilt (state-name check added). | build_region_mask.log |
| 12:51–13:19 | Download: repeated connection errors to ADS, retried automatically. | download_eac4.log |
| ~13:xx | Step docs (`docs/steps/`) and per-script run logs introduced at the user's request. | 00_overview.md |
| ~14:20 | Packages lightgbm, pykrige, shap added; project made installable (`pyproject.toml`). | environment.yml |
| 14:24 | Preprocessing of 2020–2021 started. **Session ended** (Claude closed): download, install and preprocessing all stopped. | preprocess_eac4.log |

Session 00e26aa7 (this one; transcript saved at session end)

| Time | What | Where |
|---|---|---|
| 14:30 | Resumed: download restarted (28/60 months done), install restarted, 2020–2021 preprocessed (14 s). | 03_preprocessing.md |
| ~14:35 | Install finished: lightgbm 4.7.0, pykrige 1.7.3, shap 0.51.0; 17 tests pass. | conda_install.log |
| 15:38–16:23 | Download slowed by many ADS connection drops; Sep 2023 stuck ~20 min, recovered alone. | download_eac4.log |
| 16:47–16:49 | **Step 04** run on 2020–2021 (3 runs). Raw variogram has no sill; added a detrended (predictors + lat/lon) variogram: range 846 km. | 04_exploration.md |
| 16:40–17:09 | Download stuck on Dec 2023 (~29 min), recovered alone. | download_eac4.log |
| ~17:05 | **Decision: 5 × 5-cell blocks (~380 km) main split, 8 × 8 as sensitivity** (846 km → 12 × 12 would give only 11 blocks). | 04/05 docs, 00_overview.md |
| 17:09 | **Step 05** first version **rejected** (Himalaya 49% test; 8 × 8 had 50% test). | 05_splits.md |
| 17:10 | Step 05 rewritten (cell-based region balance, balanced folds): 5 × 5 → 95 test cells (19.2%), all regions 17–24%. | 05_splits.md |
| 17:16 | **Step 06** baselines test run on 2020–2021 (13 s, 1.8 GB). Best: nearest/kriging/IDW, R² ≈ 0.57–0.59; linear regression R² 0.29. | 06_baselines.md |
| 17:20 | Logging check: step 03 doc updated with the 2020–2021 run; this journal added. | journal.md |
| ~17:25 | User chose to **wait for the full data** before step 07 (ML models). | — |
| 17:31–17:55 | Download: Sep 2024 slow (connection drops), then finished. | download_eac4.log |
| **17:55** | **Step 01 done: all 60 months**, nothing missing, no failures (~5¾ h in total). | 01_download.md |
| 17:56 | **Step 03 full run**: 14,616 steps, no gaps, no NaN, 632 MB, 60 s. Max PM2.5 5,651 (Assam, Mar 2021). 2020 lowest annual mean (COVID). | 03_preprocessing.md |
| 17:57 | **Step 04 full run** (33 s): results within a few % of the test run; detrended variogram range 897 km (was 846) → 5 × 5 decision unchanged. | 04_exploration.md |
| 17:57 | **Step 05 full run**: split identical (mask-only); train/test mean PM2.5 72.1 vs 72.6. | 05_splits.md |
| 17:57 | **Step 06 full run** (21 s, 3.6 GB): nearest R² 0.61, IDW 0.59, kriging 0.58 (best MAE, best in 4/6 regions), linreg 0.30. **Log regression blew up** (max prediction 55,041; R² −0.93) → clip/check back-transform in step 07. | 06_baselines.md |
| 18:00 | Docs 00, 01, 03, 04, 05, 06 updated with final numbers. | docs/steps/ |

**Still open (as of 18:00):** Mumbai/Chennai cells; Kashmir boundary; nothing
committed since 12:25. Next: step 07 (ML models) on the full data.
