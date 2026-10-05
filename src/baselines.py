import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.default_rng(42)
y = xr.open_zarr("data/processed/eac4_pm25_2020_2024.zarr").pm25
x = xr.open_zarr("data/processed/eac4_predictors_2020_2024.zarr")
VARS = list(x.data_vars)
print("predictors:", VARS)

splits = {
    "train": slice("2020-01-01", "2022-12-31"),
    "val":   slice("2023-01-01", "2023-12-31"),
    "test":  slice("2024-01-01", "2024-12-31"),
}

def build(split, stride=1):
    xs = x.sel(time=splits[split]).isel(time=slice(None, None, stride))
    ys = y.sel(time=splits[split]).isel(time=slice(None, None, stride))
    T, H, W = ys.shape
    feats = [xs[v].values.astype("float32") for v in VARS]
    t = xs.time.dt
    m = np.broadcast_to(t.month.values[:, None, None], (T, H, W))
    h = np.broadcast_to(t.hour.values[:, None, None], (T, H, W))
    lat = np.broadcast_to(xs.latitude.values[None, :, None], (T, H, W))
    lon = np.broadcast_to(xs.longitude.values[None, None, :], (T, H, W))
    feats += [np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
              np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), lat, lon]
    X = np.stack([f.astype("float32") for f in feats], -1).reshape(-1, len(feats))
    return X, ys.values.reshape(-1), ys, T, H, W

def metrics(obs, pred):
    err = pred - obs
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    r2 = float(1 - np.sum(err**2) / np.sum((obs - obs.mean())**2))
    return rmse, mae, bias, r2

def report(name, obs, pred, ys):
    T, H, W = ys.shape
    r = metrics(obs, pred)
    print(f"{name:22s} RMSE {r[0]:6.2f}  MAE {r[1]:6.2f}  bias {r[2]:6.2f}  R2 {r[3]:5.3f}")
    months = ys.time.dt.month.values
    o3, p3 = obs.reshape(T, H, W), pred.reshape(T, H, W)
    for s, ms in {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}.items():
        k = np.isin(months, ms)
        rs = metrics(o3[k].ravel(), p3[k].ravel())
        print(f"    {s}: RMSE {rs[0]:6.2f}  R2 {rs[3]:5.3f}")

# training data (every 2nd step, then a random 1.5M subsample to keep memory low)
Xtr, ytr, ys_tr, *_ = build("train", stride=2)
idx = rng.choice(len(Xtr), size=min(1_500_000, len(Xtr)), replace=False)
Xtr, ytr = Xtr[idx], ytr[idx]
ok = np.isfinite(Xtr).all(1) & np.isfinite(ytr)
Xtr, ytr = Xtr[ok], ytr[ok]

Xva, yva, ys_va, *_ = build("val")
Xte, yte, ys_te, *_ = build("test")
print("train samples:", Xtr.shape, " test samples:", Xte.shape)

# Baseline 0: climatology (training mean by month and hour per grid cell)
ytr_full = y.sel(time=splits["train"])
key = lambda a: a.time.dt.month * 100 + a.time.dt.hour
clim = ytr_full.groupby(key(ytr_full)).mean("time").compute()
te_key = key(ys_te)
clim_pred = clim.sel(group=te_key.values).values.reshape(-1) if "group" in clim.dims \
    else clim.sel({clim.dims[0]: te_key.values}).values.reshape(-1)
report("Climatology", yte, clim_pred, ys_te)

# Baseline 1: linear regression
lr = LinearRegression().fit(Xtr, ytr)
report("Linear regression", yte, lr.predict(Xte), ys_te)

# Baseline 2: gradient boosting
gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42)
gb.fit(Xtr, ytr)
report("Gradient boosting", yte, gb.predict(Xte), ys_te)
print("val RMSE (GB):", metrics(yva, gb.predict(Xva))[0])
