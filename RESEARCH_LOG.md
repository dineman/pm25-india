
## Feature-engineered baselines (lags, 24h means, wind speed, RH)
- GB raw target: RMSE 35.85, R2 0.757; val RMSE 44.0
- GB log1p target: worse (RMSE 43.86), underpredicts peaks
- Persistence: RMSE 61.57
- Top 10% of PM2.5 underpredicted by ~21 ug/m3 (RMSE 106)

## Spatial CV (5 folds of 8x8-cell blocks), whole download box
- WITH lat/lon: pooled RMSE 49.47, R2 0.509, top10% RMSE 149.6 (bias -20.9)
- WITHOUT lat/lon: pooled RMSE 50.18, R2 0.495, top10% RMSE 151.9 (bias -23.3)
- Coordinates are not the main cause of poor spatial transfer
- Fold 2 (RMSE 92, R2 0.32) dominated by a block near 35N 82E (RMSE 217, mean PM2.5 38.5): Tibetan Plateau / Taklamakan edge, outside India, likely dust spikes
- Box includes non-India regions (Tibet, Pakistan, Nepal, Myanmar): evaluate on an India mask next

## India spatial CV with ERA5 blh/tp (train India only, no lat/lon, test 2024)
- Pooled RMSE 35.97 (was 37.57), R2 0.748 (was 0.725); all 5 folds improved
- Top 10%: RMSE 95.8 (was 100.7), bias -34.2 (was -39.6): still underpredicts peaks
- Whole-box training: RMSE 38.00, R2 0.718
- Hardest blocks unchanged: 16.9N 70.1E (RMSE 87.9, mostly sea?), 22.9N 94.1E (RMSE 85.6)
- Next: peak-weighted training (src/spatial_cv_weighted.py)

## India spatial CV with ERA5 blh/tp (train India only, no lat/lon, test 2024)
- Pooled RMSE 35.97 (was 37.57), R2 0.748 (was 0.725); all 5 folds improved
- Top 10%: RMSE 95.8 (was 100.7), bias -34.2 (was -39.6): still underpredicts peaks
- Whole-box training: RMSE 38.00, R2 0.718
- Hardest blocks unchanged: 16.9N 70.1E (RMSE 87.9, mostly sea?), 22.9N 94.1E (RMSE 85.6)
- Next: peak-weighted training (src/spatial_cv_weighted.py)

## Peak-weighted training (India CV, ERA5 features, test 2024)
- Unweighted: RMSE 36.22, R2 0.744, top10% RMSE 96.5 bias -34.7, clean-half bias +5.1
- Weight 1+y/50: RMSE 38.61, R2 0.709, top10% RMSE 96.1 bias -27.1
- Weight (1+y/50)^2: RMSE 46.47, R2 0.579, top10% RMSE 103.2 bias -16.9
- Weighting reduces peak bias but not peak RMSE and hurts everything else: keep unweighted
- Run-to-run noise from subsampling is about 0.3-0.5 RMSE
- West coast block (3 cells near 72.8E): underpredicted all Oct-Mar (Dec obs 231 vs pred 119)
- NE India block (24 cells): underpredicted in Mar-Apr (163 vs 103, 120 vs 72): likely fire season, no fire predictor yet
