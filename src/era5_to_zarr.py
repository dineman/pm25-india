import glob
import numpy as np
import xarray as xr

months = []
for folder in sorted(glob.glob("data/raw_era5/20??_??")):
    parts = [xr.open_dataset(f) for f in sorted(glob.glob(f"{folder}/*.nc"))]
    ds = xr.merge(parts)
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})
    ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.coords], errors="ignore")
    months.append(ds.load())
ds = xr.concat(months, dim="time").sortby("time")
print(ds)

out = xr.Dataset({
    "blh": ds["blh"],
    "tp": ds["tp"].rolling(time=3).sum() * 1000,   # m -> mm, over the previous 3 h
})
out["tp"].attrs["units"] = "mm per 3h"
keep = np.where(out.time.dt.hour.values % 3 == 0)[0]
out = out.isel(time=keep)

ref = xr.open_zarr("data/processed/eac4_pm25_2020_2024.zarr")
out = out.sortby("latitude").sortby("longitude")
out = out.interp(latitude=ref.latitude, longitude=ref.longitude, method="linear")
out = out.sel(time=ref.time.values).chunk({"time": 240})
out.to_zarr("data/processed/era5_blh_tp_2020_2024.zarr", mode="w")
print(out)
print("NaNs per variable:", {v: int(out[v].isnull().sum().compute()) for v in out.data_vars})
