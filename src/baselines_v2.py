import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor

rng = np.random.default_rng(42)
y = xr.open_zarr("data/processed/eac4_pm25_2020_2024.zarr").pm25
x = xr.open_zarr("data/processed/eac4_predictors_2020_2024.zarr")

# ---- feature engineering (done on the full series so lags cross split borders)
def es(T):  # saturation vapour pressure (hPa), T in K
    c = T - 273.15
    return 6.112 * np.exp(17.62 * c / (243.12 + c))

x = x.assign(ws=np.hypot(x.u10, x.v10), rh=100 * es(x.d2m) / es(x.t2m))
for v in [v for v in ["aod550", "tcco", "tcno2", "ws", "t2m", "rh", "blh", "tp"] if v in x]:
    x[f"{v}_lag1"] = x[v].shift(time=1)                  # 3 h earlier
    x[f"{v}_lag8"] = x[v].shift(time=8)                  # 24 h earlier
    x[f"{v}_mean24h"] = x[v].rolling(time=8).mean()      # last 24 h average
print("features:", list(x.data_vars))

splits = {"train": slice("2020-01-01", "2022-12-31"),
          "val":   slice("2023-01-01", "2023-12-31"),
          "test":  slice("2024-01-01", "2024-12-31")}

def build(split, stride=1):
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
    X = np.column_stack([X, np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12),
                         np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24), la, lo]).astype("float32")
    return X, ys.values.reshape(-1), ys

def metrics(o, p):
    e = p - o
    return (float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e))),
            float(np.mean(e)), float(1 - np.sum(e**2) / np.sum((o - o.mean())**2)))

def report(name, o, p, ys):
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    r = metrics(o, p)
    hi = o >= np.percentile(o, 90)
    rh = metrics(o[hi], p[hi])
    print(f"{name:26s} RMSE {r[0]:6.2f} MAE {r[1]:6.2f} bias {r[2]:6.2f} R2 {r[3]:5.3f} | top10%: RMSE {rh[0]:6.2f} bias {rh[2]:7.2f}")

Xtr, ytr, _ = build("train", stride=2)
ok = np.isfinite(ytr)
Xtr, ytr = Xtr[ok], ytr[ok]
idx = rng.choice(len(Xtr), size=min(1_500_000, len(Xtr)), replace=False)
Xtr, ytr = Xtr[idx], ytr[idx]
Xva, yva, ys_va = build("val")
Xte, yte, ys_te = build("test")
print("train:", Xtr.shape, "test:", Xte.shape)

# persistence reference: PM2.5 24 h earlier
pers = y.shift(time=8).sel(time=splits["test"]).values.reshape(-1)
report("Persistence (24h ago)", yte, pers, ys_te)

# GB on raw target
gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=42).fit(Xtr, ytr)
report("GB (raw target)", yte, gb.predict(Xte), ys_te)

# GB on log1p target
gbl = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=42).fit(Xtr, np.log1p(ytr))
report("GB (log1p target)", yte, np.expm1(gbl.predict(Xte)), ys_te)
print("val RMSE (raw GB):", metrics(yva[np.isfinite(yva)], gb.predict(Xva)[np.isfinite(yva)])[0])
