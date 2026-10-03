import xarray as xr

ds = xr.open_mfdataset("data/raw_pred/eac4_pred_*.nc", combine="by_coords")
if "valid_time" in ds.dims or "valid_time" in ds.coords:
    ds = ds.rename({"valid_time": "time"})
ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.coords], errors="ignore")
ds = ds.sortby("time").chunk({"time": 240})
ds.to_zarr("data/processed/eac4_predictors_2020_2024.zarr", mode="w")
print(ds)
