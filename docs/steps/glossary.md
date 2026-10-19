# Glossary

Plain-language meaning of the technical terms used in the step docs. Grouped by
topic; the step where a term first matters is in brackets.

## Data

- **CAMS EAC4** (01): the Copernicus Atmosphere Monitoring Service global
  reanalysis. A computer model of the atmosphere, corrected with satellite
  observations, giving air quality and weather everywhere on a 0.75° grid
  (~80 km) every 3 hours. Our "truth" is this model's output, not ground
  measurements.
- **Reanalysis** (01): a best-estimate reconstruction of past conditions from a
  model plus all available observations.
- **Grid cell** (02): one 0.75° × 0.75° square of the grid. India has 494 of them.
- **PM2.5** (01): mass of particles smaller than 2.5 µm per m³ of air (µg m⁻³).
  The target we predict.
- **AOD** (01): Aerosol Optical Depth, how much the aerosol in the whole air
  column dims sunlight. No units. Related to PM2.5, but PM2.5 is at the surface.
- **u10, v10** (01): east-west and north-south wind at 10 m (m s⁻¹).
- **t2m, d2m** (01): temperature and dewpoint at 2 m. Dewpoint = temperature at
  which the air would become saturated; close to t2m means humid air.
- **Relative humidity (RH)** (03): how close the air is to saturation, in %.
  Computed from t2m and d2m.
- **Surface pressure (sp)** (03): mostly tells altitude: ~1,000 hPa at sea level,
  ~530 hPa in Ladakh.
- **IMD seasons** (03): India Meteorological Department seasons: winter (Jan–Feb),
  pre-monsoon (Mar–May), monsoon (Jun–Sep), post-monsoon (Oct–Dec).
- **Local solar hour** (03): clock time set by the sun at that longitude (noon =
  sun highest). Used instead of IST because the sun rises ~2 h earlier in the
  east of India than in the west.
- **sin/cos encoding** (03): writing a cyclic value (hour, day of year) as two
  numbers sin and cos, so 23:00 and 01:00 come out close together.
- **IGP** (02): Indo-Gangetic Plain, the flat, densely populated north (Punjab to
  West Bengal); the most polluted region.

## Statistics and spatial analysis

- **Percentile** (03): the value below which that % of the data falls. Median =
  50th percentile.
- **Skewed distribution** (03): most values small, a few very large (PM2.5:
  median 52, max 5,651). Means are pulled up by the large ones.
- **log1p transform** (03): log(1 + x). Squeezes large values together so the
  extremes dominate less; the +1 handles zeros. Undo with exp(x) − 1 (`expm1`).
- **Spearman correlation** (04): correlation of the *ranks* of two variables,
  from −1 to 1. Measures whether one tends to go up when the other does, even
  if not in a straight line; insensitive to extreme values.
- **Variogram** (04): how different two cells' values are, on average, as a
  function of the distance between them. Semivariance γ(h) = ½ · mean of
  (difference)² for pairs at distance h. Small γ = similar values.
- **Nugget** (04): variogram value at (almost) zero distance: variation that
  even neighbouring cells don't share (noise, very local effects).
- **Sill** (04): the level where the variogram flattens: cells this far apart
  are no more alike than random pairs.
- **Range** (04): the distance at which the sill is (practically) reached, i.e.
  how far spatial correlation reaches. For the exponential model we use the
  *practical* range (95% of sill).
- **Detrending** (04): removing a smooth large-scale pattern (here: the
  north-south gradient, via a lat/lon polynomial) before measuring local
  correlation. Without it, the variogram never flattens.
- **Residual** (04): observed value minus what a model predicts; the part the
  model can't explain.
- **Standardise** (04): subtract the mean and divide by the standard deviation,
  so the values have mean 0 and spread 1.

## Splitting and evaluation

- **Train / test split** (05): training cells are used to build a model; test
  cells are hidden until the very end and used once to measure how well it
  works on locations it has never seen.
- **Spatial block split** (05): whole squares of cells (blocks) go to test
  together, so test cells are not right next to training cells. Random
  single-cell hold-out would give over-optimistic scores, because neighbours
  are nearly copies of each other.
- **Cross-validation (CV)** (05, 07): repeat "fit on some training data, score
  on the rest" several times with different parts held out, and average the
  scores. Used to *choose* settings without touching the test set.
- **Fold** (05): one of the parts training cells are divided into for CV. With
  5 folds, each fold is held out once while the other 4 are used for fitting.
- **Block-wise (grouped) CV** (07): CV where whole blocks are kept together in
  a fold, for the same reason as the spatial test split.
- **Data leakage** (05, 07): information from the test set influencing training
  or choices, which makes scores look better than they really are.
- **RMSE** (06): root mean squared error = √mean(error²), in µg m⁻³. Punishes
  big errors heavily, so a few extreme values (fire peaks) dominate it.
- **MAE** (06): mean absolute error = mean(|error|), in µg m⁻³. The typical size
  of an error; less sensitive to extremes than RMSE.
- **R²** (06): 1 − (squared error of the model) / (squared error of always
  predicting the mean). 1 = perfect, 0 = no better than the mean, negative =
  worse than the mean.
- **Bias** (06): mean(prediction − observation). Negative = predicts too low on
  average.
- **Bootstrap** (09): estimate how much a result could vary by re-drawing the
  data *with replacement* many times and recomputing it each time.
- **Block bootstrap** (09): bootstrap that re-draws whole spatial blocks, not
  single rows, because neighbouring cells are not independent.
- **95% range (percentile interval)** (09): the middle 95% of the bootstrap
  results; a rough "plausible range" for the score.
- **Paired comparison** (09): comparing two methods on exactly the same
  re-drawn blocks, so the difference isn't blurred by which blocks were drawn.
- **Nested CV** (09): CV inside CV. Used when a method itself needs out-of-fold
  values (the hybrid's residuals), so that the fold being scored is never seen
  by any model involved in predicting it.
- **AQI categories** (09): India's National Air Quality Index bands for PM2.5
  (good < 30, satisfactory 30–60, moderate 60–90, poor 90–120, very poor
  120–250, severe ≥ 250 µg m⁻³, defined on 24 h means).
- **IST** (09): Indian Standard Time = UTC + 5:30. The CAMS time steps 00, 03, …,
  21 UTC are 05:30, 08:30, …, 02:30 IST.
- **Boundary layer** (09): the lowest part of the atmosphere, mixed by the
  ground's heating. Deep in the afternoon (pollution spread out), shallow at
  night (pollution trapped near the ground).

## Interpolation (baselines)

- **Spatial interpolation** (06): estimating a value at an unobserved place from
  values at nearby observed places, using only distance and position.
- **Nearest neighbour** (06): copy the value of the closest observed cell.
- **IDW** (06): inverse-distance weighting, a weighted average of nearby cells,
  closer cells weighted more (weight 1/distance²).
- **Bilinear / Delaunay-linear** (06): bilinear blends the 4 surrounding grid
  points; it needs a complete grid. With holes in the grid we use the scattered
  version: join observed cells into triangles (Delaunay triangulation) and
  blend the 3 corners of the triangle containing the point.
- **Ordinary kriging** (06): a weighted average whose weights come from the
  variogram, chosen to be the best unbiased estimate given how correlation
  decays with distance. Also handles clustering of observed points
  ("screening": nearer points hide those behind them).
- **Regression kriging** (08): predict with a regression/ML model, then krige
  the model's residuals from nearby training cells and add them back.

## Machine learning (step 07)

- **Feature / predictor** (07): an input variable (AOD, wind, …). **Target**:
  what we predict (PM2.5).
- **Hyperparameter / setting** (07): a choice made *before* training that
  controls how a model learns (e.g. number of trees, network size), as opposed
  to what it learns from the data.
- **Tuning** (07): trying several hyperparameter settings and keeping the one
  with the best CV score.
- **Random search** (07): tuning by trying a fixed number of randomly chosen
  settings (with a fixed seed so it is repeatable), instead of every
  combination.
- **Seed** (all): starting number for random choices, fixed so results can be
  reproduced exactly.
- **Overfitting** (07): a model learns details of its training data that don't
  carry over to new data; it scores well in training and worse on test.
- **Extrapolation** (06, 07): predicting outside the range of the training data.
  Linear models and neural networks can produce absurd values there; tree
  models can't go beyond the values they have seen.
- **Clipping** (07): forcing predictions into an allowed range, here 0 to the
  highest PM2.5 seen in training.
- **Ridge regression** (07): linear regression with a penalty (alpha) that keeps
  coefficients small, making it more stable when features are correlated.
- **Decision tree** (07): a flowchart of yes/no questions on features ("is AOD >
  0.8?") ending in a predicted value.
- **Random forest** (07): many decision trees, each built on a random sample of
  rows and features; prediction = their average. Robust, few settings to tune.
- **Gradient boosting / LightGBM** (07): trees built one after another, each
  correcting the errors of the ones before. LightGBM is a fast implementation.
  Usually the strongest model on tabular data.
- **Learning rate** (07, boosting/MLP): how big each correction step is. Smaller =
  slower but often more accurate.
- **MLP (multilayer perceptron)** (07): a small neural network: layers of units,
  each computing a weighted sum of its inputs passed through a simple non-linear
  function. Needs features scaled to similar ranges.
- **Early stopping** (07): stop training a neural network when the score on a
  held-out part stops improving, to avoid overfitting.

## Feature importance (step 10)

- **Feature importance** (10): how much a model relies on each feature.
- **SHAP value** (10): for one prediction, the share contributed by one feature,
  measured against the model's average prediction. A prediction's SHAP values
  add up to the prediction minus the average. Comes from game theory (Shapley
  values): a fair way of sharing a total among players.
- **TreeSHAP** (10): an exact, fast way to compute SHAP values for tree models;
  LightGBM has it built in (`pred_contrib=True`).
- **Mean |SHAP|** (10): average size of a feature's contribution, ignoring sign;
  the usual SHAP importance score. In log units here: 0.1 ≈ ±10% PM2.5.
- **Permutation importance** (10): shuffle one feature's values between rows and
  measure how much the error grows. Big growth = the accuracy depends on it.
- **Grouped permutation** (10): shuffle related features together (e.g.
  temperature, dewpoint, humidity), so they can't stand in for each other.
- **Dependence curve** (10): average SHAP value of a feature plotted against its
  value: shows *how* the feature acts (rising, falling, threshold).

## Maps (step 11)

- **Regridding** (11): moving a gridded field to another grid, e.g. from 0.75°
  to 0.25°. Bilinear regridding blends the 4 surrounding values.
- **Downscaling / spatial enhancement** (11): producing a finer map than the
  input data. Statistical downscaling uses a model to add detail; without
  fine-scale inputs or reference data the added detail can't be checked.
- **Consistency check** (11): averaging a fine map back to the coarse grid and
  comparing with the original coarse data. It tests faithfulness, not accuracy
  at the fine scale.
- **Point-of-view boundaries** (11): Natural Earth versions of country borders
  as officially claimed by one country. The India version includes all of Jammu
  & Kashmir and Ladakh (PoK, Gilgit-Baltistan, Aksai Chin).
