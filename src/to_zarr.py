import xarray as xr

ds = xr.open_mfdataset("data/raw/eac4_pm25_*.nc", combine="by_coords")

# newer CDS/ADS files use valid_time instead of time
if "valid_time" in ds.dims or "valid_time" in ds.coords:
    ds = ds.rename({"valid_time": "time"})
ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.coords], errors="ignore")

name = list(ds.data_vars)[0]
pm25 = (ds[name] * 1e9).rename("pm25")   # kg/m3 -> ug/m3
pm25.attrs["units"] = "ug/m3"

pm25 = pm25.sortby("time").chunk({"time": 240})
pm25.to_dataset().to_zarr("data/processed/eac4_pm25_2020_2024.zarr", mode="w")
print(pm25)
