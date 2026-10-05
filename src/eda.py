import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

y = xr.open_zarr("data/processed/eac4_pm25_2020_2024.zarr").pm25.load()
x = xr.open_zarr("data/processed/eac4_predictors_2020_2024.zarr").load()

# 1. Mean PM2.5 map
ax = plt.axes(projection=ccrs.PlateCarree())
y.mean("time").plot(ax=ax, cmap="magma_r", cbar_kwargs={"label": "PM2.5 (µg/m³)"})
ax.coastlines(); ax.set_title("Mean PM2.5, 2020-2024")
plt.savefig("results/eda/mean_map.png", dpi=150, bbox_inches="tight"); plt.close()

# 2. Seasonal cycle (monthly mean over India) and diurnal cycle
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
y.mean(["latitude", "longitude"]).groupby("time.month").mean().plot(ax=axs[0], marker="o")
axs[0].set_title("Seasonal cycle")
y.mean(["latitude", "longitude"]).groupby("time.hour").mean().plot(ax=axs[1], marker="o")
axs[1].set_title("Diurnal cycle (UTC hour)")
plt.savefig("results/eda/cycles.png", dpi=150, bbox_inches="tight"); plt.close()

# 3. City time series (daily mean)
cities = {"Delhi": (28.6, 77.2), "Kanpur": (26.4, 80.3), "Kolkata": (22.6, 88.4),
          "Mumbai": (19.1, 72.9), "Chennai": (13.1, 80.3), "Bengaluru": (13.0, 77.6)}
fig, ax = plt.subplots(figsize=(12, 4))
for name, (la, lo) in cities.items():
    y.sel(latitude=la, longitude=lo, method="nearest").resample(time="1D").mean().plot(ax=ax, label=name, lw=0.7)
ax.legend(ncol=3); ax.set_title("Daily PM2.5 at nearest grid cell")
plt.savefig("results/eda/cities.png", dpi=150, bbox_inches="tight"); plt.close()

# 4. Predictor correlations with PM2.5 (daily means, per grid cell, then averaged)
yd = y.resample(time="1D").mean()
rows = []
for v in x.data_vars:
    xd = x[v].resample(time="1D").mean()
    r = xr.corr(yd, xd, dim="time")
    rows.append((v, float(r.mean()), float(r.min()), float(r.max())))
rows.sort(key=lambda t: -abs(t[1]))
print(f"{'variable':35s} mean_r   min_r   max_r")
for v, m, lo, hi in rows:
    print(f"{v:35s} {m:6.2f} {lo:7.2f} {hi:7.2f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh([r[0] for r in rows], [r[1] for r in rows])
ax.set_xlabel("Mean correlation with PM2.5"); ax.invert_yaxis()
plt.savefig("results/eda/correlations.png", dpi=150, bbox_inches="tight"); plt.close()
print("saved figures in results/eda/")
