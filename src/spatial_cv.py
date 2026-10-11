import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.default_rng(42)
y = xr.open_zarr("data/processed/eac4_pm25_2020_2024.zarr").pm25
x = xr.open_zarr("data/processed/eac4_predictors_2020_2024.zarr")

def es(T):
    c = T - 273.15
    return 6.112 * np.exp(17.62 * c / (243.12 + c))

x = x.assign(ws=np.hypot(x.u10, x.v10), rh=100 * es(x.d2m) / es(x.t2m))
for v in [v for v in ["aod550", "tcco", "tcno2", "ws", "t2m", "rh", "blh", "tp"] if v in x]:
    x[f"{v}_lag1"] = x[v].shift(time=1)
    x[f"{v}_lag8"] = x[v].shift(time=8)
    x[f"{v}_mean24h"] = x[v].rolling(time=8).mean()

splits = {"train": slice("2020-01-01", "2022-12-31"),
          "test":  slice("2024-01-01", "2024-12-31")}

def build(split, stride):
    ds = x.sel(time=splits[split]).isel(time=slice(None, None, stride))
    ys = y.sel(time=splits[split]).isel(time=slice(None, None, stride))
    a = ds.to_array("f").transpose("time", "latitude", "longitude", "f")
    T, H, W, F = a.shape
    X = a.values.reshape(-1, F).astype("float32")
    t = ds.time.dt
    m = np.broadcast_to(t.month.values[:, None, None], (T, H, W)).reshape(-1)
    h = np.broadcast_to(t.hour.values[:, None, None], (T, H, W)).reshape(-1)
    la = np.broadcast_to(ds.latitude.values[None, :, None], (T, H, W)).reshape(-1)
    lo = np.broadcast_to(ds.longitude.values[None, None, :], (T, H, W)).reshape(-1)
    # lat, lon are the LAST two columns so they can be dropped easily
    X = np.column_stack([X, np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
                         np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), la, lo]).astype("float32")
    return X, ys.values.reshape(-1), ys

def rmse(o, p): return float(np.sqrt(np.mean((p - o) ** 2)))
def r2(o, p): return float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2))

Xtr, ytr, ys_tr = build("train", 2)
Xte, yte, ys_te = build("test", 2)
Ttr, H, W = ys_tr.shape
Tte = ys_te.shape[0]
lat, lon = ys_te.latitude.values, ys_te.longitude.values

# 8x8-cell blocks, randomly assigned to 5 spatial folds
block = ((np.arange(H)[:, None] // 8) * 1000 + (np.arange(W)[None, :] // 8)).reshape(-1)
perm = rng.permutation(np.unique(block))
fold_of_block = {b: i % 5 for i, b in enumerate(perm)}
cell_fold = np.array([fold_of_block[b] for b in block])
fold_tr = cell_fold[np.tile(np.arange(H * W), Ttr)]
fold_te = cell_fold[np.tile(np.arange(H * W), Tte)]
block_te = block[np.tile(np.arange(H * W), Tte)]

variants = {"WITH lat/lon": slice(None), "WITHOUT lat/lon": slice(0, -2)}
block_rows = []

for vname, cols in variants.items():
    allo, allp = [], []
    print(f"\n=== {vname} ===")
    for f in range(5):
        tr = np.where((fold_tr != f) & np.isfinite(ytr))[0]
        tr = rng.choice(tr, size=min(600_000, len(tr)), replace=False)
        te = np.where(fold_te == f)[0]
        te = te[np.isfinite(yte[te])]
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          min_samples_leaf=100, early_stopping=False,
                                          random_state=42)
        m.fit(Xtr[tr][:, cols], ytr[tr])
        p = m.predict(Xte[te][:, cols])
        o = yte[te]
        print(f"fold {f}: RMSE {rmse(o,p):6.2f}  R2 {r2(o,p):6.3f}  bias {np.mean(p-o):7.2f}")
        allo.append(o); allp.append(p)
        if vname == "WITHOUT lat/lon":
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

print("\nHardest blocks (WITHOUT lat/lon):  RMSE | mean PM2.5 | block-centre lat, lon")
for r in sorted(block_rows, reverse=True)[:8]:
    print(f"  {r[0]:7.1f} | {r[1]:6.1f} | {r[2]:5.1f}, {r[3]:5.1f}")
