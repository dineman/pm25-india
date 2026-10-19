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

Session 25e49b66 (**first session on the Mac**; 2026-10-18 evening to 2026-10-19 morning)

| Time | What | Where |
|---|---|---|
| ~23:30 | Repo cloned to the Mac (`~/Project/pm25-india`); `data/` (raw 60 months + processed) copied over by hand from Fedora; all raw files open, processed set 14,616 × 43 × 40. | — |
| ~23:33 | Conda env `pm25` updated from environment.yml (lightgbm, pykrige, shap, pytest, `pm25` package); 54 tests pass. | environment.yml |
| ~23:36 | `git pull`: step 08 commits from Fedora arrived. | git |
| 23:39 | **Decision: logging rules written into CLAUDE.md** (were only in Fedora's Claude memory), so both machines follow the same routine. Commit 7196e7c pushed. | CLAUDE.md |
| 23:39 | Not on the Mac (not in git): fitted models `*.joblib` and prediction `.nc` files of steps 06–08, needed by step 09. | — |
| 23:40–23:45 | **Rebuild on the Mac**: step 06 in 11 s, step 07 final fits (`--cv-from`, no re-tuning) in 4 min (lgbm 68 s vs 101 s on Fedora), step 08. Compared with the committed Fedora metrics: ridge, rf, nearest, kriging, linreg **identical**; lgbm RMSE 54.97 → 54.95; **MLP R² 0.545 → 0.528** (neural-net training isn't bit-reproducible across machines); **IDW 55.31 → 55.16 although IDW has no randomness**. | run_baselines.log, train_models.log, run_hybrid.log |
| ~23:45 | **Step 09 (evaluation) started.** Found: the 95 test cells form only **6 blocks**, most regions = 1 block (south 1, northwest 1, central 1, northeast 2 incl. a 1-cell block, IGP + Himalaya share 1). | 09_evaluation.md |
| ~23:45 | **Decision (user): step 09 = core + CV check.** Core: all methods in one table, new breakdowns by hour (IST), pollution level, year; block bootstrap for error bars. CV check: the same comparison on the 399 training cells with the 5 block folds (27 blocks). 8 × 8 split not repeated. | 09_evaluation.md |
| 23:50 | **Bug found: IDW neighbour ties.** For 23 of 95 test cells the 8th and 9th nearest training cells are at exactly the same distance (regular grid); `np.argsort` broke the tie differently on the two machines. Fix: stable sort (ties → lower cell index), test added; 64 tests pass. | baselines.py |
| 23:51–23:55 | Steps 06, 07, 08 re-run on the Mac with the IDW fix, so all step 09 tables use the same predictions. IDW test RMSE 55.14, lgbm_rk 45.94, lgbm_idw 45.00. | logs |
| 23:56 | **Step 09** test part done; nested CV fits started (20 LightGBM fits, ~70 s each). | evaluate.log |
| ~00:15 | **Decisions (user):** Mumbai/Chennai cells stay excluded (keep 50% land rule); Kashmir must follow **India's official boundary**; the "no truth below 0.75°" limitation is accepted (CAMS is the only data available). | 00_overview.md |
| 00:14–07:32 | Step 09 slowed to a crawl: the Mac (on battery) slept most of the night; `caffeinate` (started 01:15) doesn't prevent sleep on battery with the lid closed. The 20 nested fits were done at 07:32. Lesson: run long jobs on mains power. | evaluate.log |
| 07:32 | **Step 09 done.** CV check (27 blocks): RMSE hybrid 41.1, LightGBM 41.9, kriging 42.0, nearest 47.8; R² 0.68 / 0.66 / 0.66 / 0.56. **The hybrid's large test gain (R² 0.59 → 0.72) comes mostly from 2 of 6 test blocks** (northeast 38, IGP 50); over 27 blocks it helps in 15, hurts in 12. Robust: ML ≫ interpolation in the Himalaya; hybrid removes LightGBM's low bias (−6.3 → −0.4) and halves under-prediction of severe pollution; kriged residuals beat IDW residuals. Errors highest at night, lowest ~14:30 IST. | 09_evaluation.md |
| 07:32–07:36 | Docs: 09_evaluation.md written; glossary (bootstrap, nested CV, AQI, IST, boundary layer); overview status and decisions; re-run notes added to 06, 07, 08 docs. `.tmp.driveupload/` (Google Drive sync files in results/) added to .gitignore. | docs/steps/ |
| 07:36 | **Commit 915e99a pushed**: step 09 code, tests, docs, results (CSV + figures; `nested_resid_lgbm.nc` not in git); IDW fix; re-run results of steps 06–08. | git |
| ~07:37 | **Decision (user): run step 10 next, automatically.** Kashmir scope (maps only vs region mask) still to be answered; doesn't affect step 10. | — |
| 07:40 | **Step 10 (feature importance)** code: `pm25.importance` (TreeSHAP via LightGBM `pred_contrib`, grouped permutation importance, dependence curves) + 5 tests; smoke test on 5,000 rows OK (25 s). Full run started (200k SHAP rows from all 494 India cells; permutation on 300k test rows × 5 repeats, 14 features + 7 groups). Mac still on battery → `caffeinate`. | feature_importance.log |
| 07:55 | Step 10 SHAP done after ~15 min (200k rows × 1,000 trees; slower than the 2–4 min estimated). | feature_importance.log |
| 08:00 | **Step 10 done.** Mean \|SHAP\| (log units): **AOD 0.35**, hour of day 0.22, day of year 0.17, location 0.17, surface pressure 0.15, temperature & humidity 0.15, **wind 0.08**. Permutation on test cells (ΔRMSE): **location 18.8**, day of year 14.5, hour 13.9, AOD 11.4 (largest ΔMAE 12.5), wind 5.5. Himalaya: AOD 0.77 + pressure 0.57 (altitude); pre-monsoon: weather explains least. Night ≈ +30%, noon ≈ −25% (boundary layer); cold raises PM2.5 only in winter/post-monsoon. | 10_feature_importance.md |
| 08:01 | Docs: 10_feature_importance.md, glossary (SHAP, TreeSHAP, permutation importance, dependence curve), overview status. | docs/steps/ |

**Still open (as of 08:01):** Kashmir boundary scope (maps only, or also the region mask); step 07 feature-set variants not re-run on the Mac; heatmap label contrast (step 10).
