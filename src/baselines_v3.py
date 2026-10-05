import numpy as np
import xarray as xr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

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
NAMES = list(x.data_vars) + ["month_sin", "month_cos", "hour_sin", "hour_cos", "lat", "lon"]

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

def rmse(o, p): return float(np.sqrt(np.mean((p - o) ** 2)))
def r2(o, p): return float(1 - np.sum((p - o) ** 2) / np.sum((o - o.mean()) ** 2))

def report(name, o, p):
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    hi = o >= np.percentile(o, 90)
    print(f"{name:28s} RMSE {rmse(o,p):6.2f}  R2 {r2(o,p):5.3f}  bias {np.mean(p-o):6.2f} | top10%: RMSE {rmse(o[hi],p[hi]):6.2f} bias {np.mean(p[hi]-o[hi]):7.2f}")

# ---- spatial blocks: hold out ~20% of 8x8-cell blocks from training
Xtr, ytr, ys_tr = build("train", stride=2)
Ttr, H, W = ys_tr.shape
bid = (np.arange(H)[:, None] // 8) * 1000 + (np.arange(W)[None, :] // 8)
ids = np.unique(bid)
held_ids = rng.choice(ids, size=max(1, len(ids) // 5), replace=False)
held_cells = np.isin(bid, held_ids).reshape(-1)
print(f"held-out blocks: {len(held_ids)} of {len(ids)}")

keep = np.isfinite(ytr) & ~np.tile(held_cells, Ttr)
Xtr, ytr = Xtr[keep], ytr[keep]
idx = rng.choice(len(Xtr), size=min(1_500_000, len(Xtr)), replace=False)
Xtr, ytr = Xtr[idx], ytr[idx]

Xva, yva, ys_va = build("val", stride=4)
kv = np.isfinite(yva) & ~np.tile(held_cells, ys_va.shape[0])
Xva, yva = Xva[kv], yva[kv]

Xte, yte, ys_te = build("test")
held_te = np.tile(held_cells, ys_te.shape[0])

# ---- train with many iterations, pick the best one on the validation year
gb = HistGradientBoostingRegressor(max_iter=800, learning_rate=0.05, max_leaf_nodes=63,
                                   min_samples_leaf=100, early_stopping=False, random_state=42)
gb.fit(Xtr, ytr)
val_curve = [rmse(yva, p) for p in gb.staged_predict(Xva)]
best = int(np.argmin(val_curve)) + 1
print(f"best iteration: {best} of 800, val RMSE {val_curve[best-1]:.2f}")

for i, p in enumerate(gb.staged_predict(Xte), 1):
    if i == best:
        pred = p
        break

report("Test - all cells", yte, pred)
report("Test - cells seen in training", yte[~held_te], pred[~held_te])
report("Test - HELD-OUT blocks", yte[held_te], pred[held_te])

# ---- which features matter (permutation importance on 150k validation samples)
sub = rng.choice(len(Xva), size=min(150_000, len(Xva)), replace=False)
pi = permutation_importance(gb, Xva[sub], yva[sub], n_repeats=3, random_state=42,
                            scoring="neg_root_mean_squared_error", n_jobs=-1)
order = np.argsort(-pi.importances_mean)
print("\nTop 15 features (RMSE increase when shuffled):")
for k in order[:15]:
    print(f"  {NAMES[k]:18s} {pi.importances_mean[k]:7.2f}")
