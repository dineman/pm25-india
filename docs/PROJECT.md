# Project 3: Machine Learning for Spatial Enhancement of Air Quality Prediction over India

## Objective
Develop and evaluate ML models that estimate PM2.5 over India from atmospheric and
aerosol conditions in CAMS EAC4 reanalysis data. Given CAMS information at location s
and time t, estimate PM2.5 at the same s, t, and use the model outputs to generate
spatially enhanced PM2.5 maps over India.

## Data
- Source: CAMS global reanalysis EAC4, Atmosphere Data Store (ADS)
  https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4
- Region: India. Period: 2020–2024.
- Target: CAMS PM2.5
- Predictors (may include):
  - Aerosol Optical Depth (AOD)
  - 10 m u and v wind
  - 2 m temperature
  - 2 m dewpoint temperature
  - Surface pressure
  - Latitude and longitude

## Evaluation
- Reference values = CAMS PM2.5 at HELD-OUT grid locations (not used in training).
- Trained models are then combined with spatial interpolation to produce
  finer-resolution PM2.5 maps.

## Research questions
1. How accurately can PM2.5 be estimated from CAMS atmospheric and aerosol variables?
2. How do different ML models compare?
3. Which atmospheric and aerosol features are most important for prediction?
4. How does performance vary across regions and seasons of India?
5. How well does the model generalize to CAMS grid locations not used during training?
6. How does ML-based spatial enhancement compare with standard spatial interpolation methods?
