import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.default_rng(42)
y = xr.open_zarr("data/processed/eac4_pm25_2020_2024.zarr").pm25
x = xr.open_zarr("data/processed/eac4_predictors_2020_2024.zarr")
x = xr.merge([x, xr.open_zarr("data/processed/era5_blh_tp_2020_2024.zarr")])

def es(T):
    c = T - 273.15
    return 6.112 * np.exp(17.62 * c / (243.12 + c))

x = x.assign(ws=np.hypot(x.u10, x.v10), rh=100 * es(x.d2m) / es(x.t2m))
for v in ["aod550", "tcco", "tcno2", "ws", "t2m", "rh", "blh", "tp"]:
    x[f"{v}_lag1"] = x[v].shift(time=1)
    x[f"{v}_lag8"] = x[v].shift(time=8)
    x[f"{v}_mean24h"] = x[v].rolling(time=8).mean()

# spatial context features (inputs only, no target information)
NEW = []
for v in ["aod550", "tcco", "tcno2", "blh", "ws"]:
    x[f"{v}_nb3"] = x[v].rolling(latitude=3, longitude=3, center=True, min_periods=1).mean()
    x[f"{v}_anom3"] = x[v] - x[f"{v}_nb3"]
    NEW += [f"{v}_nb3", f"{v}_anom3"]
for v in ["aod550", "tcco"]:
    x[f"{v}_nb5"] = x[v].rolling(latitude=5, longitude=5, center=True, min_periods=1).mean()
    NEW.append(f"{v}_nb5")

NAMES = list(x.data_vars) + ["month_sin", "month_cos", "hour_sin", "hour_cos"]
cols = {"base features": [i for i, n in enumerate(NAMES) if n not in NEW],
        "+ spatial context": list(range(len(NAMES)))}

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
    X = np.column_stack([X, np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
                         np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)]).astype("float32")
    return X, ys.values.reshape(-1), ys

def rmse(o, p): return float(np.sqrt(np.mean((p - o) ** 2)))
def r2(o, p): return float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2))

Xtr, ytr, ys_tr = build("train", 3)
Xte, yte, ys_te = build("test", 2)
Ttr, H, W = ys_tr.shape
Tte = ys_te.shape[0]

mask = np.load("data/processed/india_mask.npy").reshape(-1)
block = ((np.arange(H)[:, None] // 8) * 1000 + (np.arange(W)[None, :] // 8)).reshape(-1)
india_blocks = np.unique(block[mask])
perm = rng.permutation(india_blocks)
fold_of_block = {b: i % 5 for i, b in enumerate(perm)}
cell_fold = np.array([fold_of_block.get(b, -1) for b in block])
cells_tr, cells_te = np.tile(np.arange(H * W), Ttr), np.tile(np.arange(H * W), Tte)
fold_tr, fold_te = cell_fold[cells_tr], cell_fold[cells_te]
mask_tr, mask_te = mask[cells_tr], mask[cells_te]

preds = {k: np.full(len(yte), np.nan, dtype="float32") for k in cols}
for f in range(5):
    tr = np.where((fold_tr != f) & mask_tr & np.isfinite(ytr))[0]
    tr = rng.choice(tr, size=min(600_000, len(tr)), replace=False)
    te = np.where((fold_te == f) & mask_te & np.isfinite(yte))[0]
    for name, c in cols.items():
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          min_samples_leaf=100, early_stopping=False, random_state=42)
        m.fit(Xtr[tr][:, c], ytr[tr])
        preds[name][te] = m.predict(Xte[te][:, c])
    print("fold", f, "done", flush=True)

ok = np.isfinite(preds["base features"])
o = yte[ok]
hi = o >= np.percentile(o, 90)
months = np.repeat(ys_te.time.dt.month.values, H * W)[ok]
print("\nIndia, spatial CV, pooled over 5 folds")
print(f"{'variant':18s} {'RMSE':>6s} {'R2':>6s} | top10% RMSE  bias | DJF   MAM   JJA   SON  (RMSE)")
for name in cols:
    p = preds[name][ok]
    s = {k: rmse(o[np.isin(months, ms)], p[np.isin(months, ms)])
         for k, ms in {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}.items()}
    print(f"{name:18s} {rmse(o,p):6.2f} {r2(o,p):6.3f} | {rmse(o[hi],p[hi]):10.2f} {np.mean(p[hi]-o[hi]):7.2f} | "
          f"{s['DJF']:5.1f} {s['MAM']:5.1f} {s['JJA']:5.1f} {s['SON']:5.1f}")
