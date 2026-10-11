import numpy as np
import xarray as xr
import geopandas as gpd
import shapely
import cartopy.io.shapereader as shpreader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.default_rng(42)
y = xr.open_zarr("data/processed/eac4_pm25_2020_2024.zarr").pm25
x = xr.open_zarr("data/processed/eac4_predictors_2020_2024.zarr")
x = xr.merge([x, xr.open_zarr("data/processed/era5_blh_tp_2020_2024.zarr")])

def es(T):
    c = T - 273.15
    return 6.112 * np.exp(17.62 * c / (243.12 + c))

x = x.assign(ws=np.hypot(x.u10, x.v10), rh=100 * es(x.d2m) / es(x.t2m))
for v in [v for v in ["aod550", "tcco", "tcno2", "ws", "t2m", "rh", "blh", "tp"] if v in x]:
    x[f"{v}_lag1"] = x[v].shift(time=1)
    x[f"{v}_lag8"] = x[v].shift(time=8)
    x[f"{v}_mean24h"] = x[v].rolling(time=8).mean()

splits = {"train": slice("2020-01-01", "2022-12-31"), "test": slice("2024-01-01", "2024-12-31")}

def build(split, stride):
    ds = x.sel(time=splits[split]).isel(time=slice(None, None, stride))
    ys = y.sel(time=splits[split]).isel(time=slice(None, None, stride))
    a = ds.to_array("f").transpose("time", "latitude", "longitude", "f")
    T, H, W, F = a.shape
    X = a.values.reshape(-1, F).astype("float32")
    t = ds.time.dt
    m = np.broadcast_to(t.month.values[:, None, None], (T, H, W)).reshape(-1)
    h = np.broadcast_to(t.hour.values[:, None, None], (T, H, W)).reshape(-1)
    # no lat/lon features
    X = np.column_stack([X, np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
                         np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)]).astype("float32")
    return X, ys.values.reshape(-1), ys

def rmse(o, p): return float(np.sqrt(np.mean((p - o) ** 2)))
def r2(o, p): return float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2))

Xtr, ytr, ys_tr = build("train", 2)
Xte, yte, ys_te = build("test", 2)
Ttr, H, W = ys_tr.shape
Tte = ys_te.shape[0]
lat, lon = ys_te.latitude.values, ys_te.longitude.values

# ---- India mask (0.4 degree buffer so coastal/border cells are kept)
world = gpd.read_file(shpreader.natural_earth(resolution="50m", category="cultural",
                                              name="admin_0_countries"))
geom = world[world["ADMIN"] == "India"].geometry
india = geom.union_all() if hasattr(geom, "union_all") else geom.unary_union
lon2d, lat2d = np.meshgrid(lon, lat)
mask = shapely.contains_xy(india.buffer(0.4), lon2d.ravel(), lat2d.ravel())   # (H*W,)
print(f"India cells: {mask.sum()} of {H*W}")
plt.figure(figsize=(5, 5))
plt.pcolormesh(lon, lat, mask.reshape(H, W), cmap="Greens")
plt.title("India mask"); plt.savefig("results/india_mask.png", dpi=120, bbox_inches="tight")
np.save("data/processed/india_mask.npy", mask.reshape(H, W))

# ---- spatial folds over 8x8-cell blocks that contain India cells
block = ((np.arange(H)[:, None] // 8) * 1000 + (np.arange(W)[None, :] // 8)).reshape(-1)
india_blocks = np.unique(block[mask])
perm = rng.permutation(india_blocks)
fold_of_block = {b: i % 5 for i, b in enumerate(perm)}
cell_fold = np.array([fold_of_block.get(b, -1) for b in block])   # -1 = outside India blocks
cells_tr = np.tile(np.arange(H * W), Ttr)
cells_te = np.tile(np.arange(H * W), Tte)
fold_tr, fold_te = cell_fold[cells_tr], cell_fold[cells_te]
mask_tr, mask_te = mask[cells_tr], mask[cells_te]
block_te = block[cells_te]

block_rows = []
for vname in ["train on WHOLE box", "train on INDIA only"]:
    allo, allp = [], []
    print(f"\n=== {vname}, test on India cells, no lat/lon ===")
    for f in range(5):
        trm = (fold_tr != f) & np.isfinite(ytr)
        if vname == "train on INDIA only":
            trm &= mask_tr
        tr = np.where(trm)[0]
        tr = rng.choice(tr, size=min(600_000, len(tr)), replace=False)
        te = np.where((fold_te == f) & mask_te)[0]
        te = te[np.isfinite(yte[te])]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          min_samples_leaf=100, early_stopping=False,
                                          random_state=42).fit(Xtr[tr], ytr[tr])
        p, o = m.predict(Xte[te]), yte[te]
        print(f"fold {f}: RMSE {rmse(o,p):6.2f}  R2 {r2(o,p):6.3f}  bias {np.mean(p-o):7.2f}")
        allo.append(o); allp.append(p)
        if vname == "train on INDIA only":
            bt = block_te[te]
            for b in np.unique(bt):
                k = bt == b
                i0, j0 = (b // 1000) * 8, (b % 1000) * 8
                block_rows.append((rmse(o[k], p[k]), float(o[k].mean()),
                                   float(lat[i0:i0+8].mean()), float(lon[j0:j0+8].mean())))
    o, p = np.concatenate(allo), np.concatenate(allp)
    hi = o >= np.percentile(o, 90)
    print(f"POOLED: RMSE {rmse(o,p):6.2f}  R2 {r2(o,p):6.3f}  bias {np.mean(p-o):6.2f} | "
          f"top10%: RMSE {rmse(o[hi],p[hi]):6.2f} bias {np.mean(p[hi]-o[hi]):7.2f}")

print("\nHardest India blocks:  RMSE | mean PM2.5 | block-centre lat, lon")
for r in sorted(block_rows, reverse=True)[:6]:
    print(f"  {r[0]:7.1f} | {r[1]:6.1f} | {r[2]:5.1f}, {r[3]:5.1f}")
