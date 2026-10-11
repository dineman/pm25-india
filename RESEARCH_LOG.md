
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
