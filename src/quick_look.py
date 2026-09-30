import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

ds = xr.open_dataset("data/raw/eac4_pm25_2020_01.nc")
print(ds)
name = list(ds.data_vars)[0]
da = ds[name]
other = [d for d in da.dims if d not in ("latitude", "longitude")]
mean = da.mean(dim=other) * 1e9  # kg/m3 -> ug/m3

ax = plt.axes(projection=ccrs.PlateCarree())
mean.plot(ax=ax, cmap="magma_r", cbar_kwargs={"label": "PM2.5 (µg/m³)"})
ax.coastlines()
ax.set_title("EAC4 mean PM2.5, Jan 2020")
plt.savefig("results/quick_look.png", dpi=150, bbox_inches="tight")
print("saved results/quick_look.png")
