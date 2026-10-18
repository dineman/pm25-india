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

| 19:44 | **Commit 17742a9 pushed**: steps 04–06 code, tests, docs, journal, results (figures/CSV/configs, 2.6 MB). `*.egg-info/` added to .gitignore. Not in git by design: data, `.nc` files, run logs (`*.log`), session transcripts (`logs/`). | git |
| 19:44 | **Step 07 (ML models) started** on the full data. | 07_models.md |
| 19:46 | Refactor: evaluation table and metric groupings moved to `pm25.metrics` (shared by steps 06 and 07); baselines re-run, metrics **identical**. | metrics.py |
| 19:47 | Step 07 **pilot** (1 setting per model, CV on 2% of times, final rf/mlp on 200k rows; output to scratch, not kept): pipeline works, 200 s, 3.6 GB. Per-candidate CV time: ridge <1 s, rf 30 s, lgbm 7 s, mlp 62 s. Untuned lgbm already R² ~0.6 on test. | train_models.log |
| 19:53 | **Step 07 main run started**: 4 models × 8 settings, CV on 5% of time steps, final rf/mlp on 2 M rows, feature set `all`. Expected ~35–45 min. | train_models.log |
| ~19:55 | Glossary (`glossary.md`) and step 07 doc (explanations, pilot) written at the user's request for proper explanations. Note: times 19:44–19:53 corrected from earlier estimates using log timestamps. | glossary.md, 07_models.md |

| 19:53–20:29 | Tuning finished: 4 models × 8 settings (ridge 4 s, rf 3 min, lgbm 6 min, mlp 27 min). Best CV RMSE: lgbm 36.7, mlp 37.9, rf 40.8, ridge 47.8. | cv_results.csv |
| 20:29–20:32 | Final fits ridge, rf, lgbm done. | train_models.log |
| 20:32–22:42 | **Final MLP fit crawled for 2 h+** (expected ~10 min): RAM full (Firefox ~6.5 GB), swap full (8/8 GB), 1.1 GB of the training process in swap. | — |
| 22:41 | **Bug found: wrong "best" settings.** Joining the per-model CV tables kept each table's row numbers 0–7, so labels repeated and the best-row lookup took rows from the wrong models (e.g. lgbm got a setting with CV RMSE 40.9 instead of 36.7). Tuning scores themselves were correct. Final fits of this run therefore invalid. | 07_models.md |
| 22:42 | Run stopped (TaskStop; a plain `kill` was blocked by the permission check). Fix: `best_settings()` in `models.py` + regression test; `ignore_index=True`; new `--cv-from` to reuse tuning results; free data copies before fitting; separate seeded RNG for final-fit row sampling. 54 tests pass. | models.py, train_models.py |
| 22:44 | **Final fits re-run** with the correct settings (`--cv-from results/07_models/all/cv_results.csv`, no re-tuning). | train_models.log |

| 22:44–22:51 | **Step 07 final fits (correct settings)**: test R² lgbm 0.59 (MAE 19.1, lowest of all methods), mlp 0.55, rf 0.47, ridge 0.31; interpolation nearest 0.61, kriging 0.58. ML beats interpolation beyond 100 km from training cells and in NW/Himalaya; interpolation wins in the NE (fires). | 07_models.md |
| 22:52–23:07 | **Feature variants**: lgbm R² without lat/lon 0.48, brief set 0.44 → keep `all`. lat/lon help on unseen blocks (large-scale emission patterns), time features matter. | 07_models.md |

| 23:08 | **Commit 52e1855 pushed**: step 07 code, tests, docs, glossary, results (fitted models `*.joblib` and `.nc` predictions not in git). | git |

| 23:23 | Commit 99d5041 pushed (journal line). | git |
| 23:25 | **Step 08 (regression kriging) started**: out-of-fold LightGBM residuals at training cells (5 fold fits), kriged to test cells and added to the step 07 prediction. | 08_hybrid.md |

| 23:25–23:34 | **Step 08 done**: out-of-fold LightGBM R² 0.66 on training cells; residual variogram range ~490 km (reaches a sill). **Hybrid test R² 0.72 (kriged residuals) / 0.73 (IDW)** vs 0.59 ML only and 0.58–0.61 interpolation only; bias −7.9 → −1.4; best at every distance and in every season. | 08_hybrid.md |

| 23:35 | **Commit b50b243 pushed**: step 08 code, test, docs, results (`.nc` files not in git). | git |

**Still open (as of 23:35):** Mumbai/Chennai cells; Kashmir boundary.
