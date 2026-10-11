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

mask = np.load("data/processed/india_mask.npy").reshape(-1)       # saved by spatial_cv_india.py
block = ((np.arange(H)[:, None] // 8) * 1000 + (np.arange(W)[None, :] // 8)).reshape(-1)
india_blocks = np.unique(block[mask])
perm = rng.permutation(india_blocks)
fold_of_block = {b: i % 5 for i, b in enumerate(perm)}
cell_fold = np.array([fold_of_block.get(b, -1) for b in block])
cells_tr, cells_te = np.tile(np.arange(H * W), Ttr), np.tile(np.arange(H * W), Tte)
fold_tr, fold_te = cell_fold[cells_tr], cell_fold[cells_te]
mask_tr, mask_te = mask[cells_tr], mask[cells_te]

variants = {"unweighted": lambda v: None,
            "weight 1+y/50": lambda v: 1 + v / 50.0,
            "weight (1+y/50)^2": lambda v: (1 + v / 50.0) ** 2}
preds = {k: np.full(len(yte), np.nan, dtype="float32") for k in variants}

for f in range(5):
    tr = np.where((fold_tr != f) & mask_tr & np.isfinite(ytr))[0]
    tr = rng.choice(tr, size=min(600_000, len(tr)), replace=False)
    te = np.where((fold_te == f) & mask_te & np.isfinite(yte))[0]
    for name, wf in variants.items():
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63,
                                          min_samples_leaf=100, early_stopping=False, random_state=42)
        m.fit(Xtr[tr], ytr[tr], sample_weight=wf(ytr[tr]))
        preds[name][te] = m.predict(Xte[te])
    print("fold", f, "done", flush=True)

ok = np.isfinite(preds["unweighted"])
o = yte[ok]
hi = o >= np.percentile(o, 90)
lo = o <= np.median(o)
print("\nIndia, spatial CV, pooled over 5 folds")
print(f"{'variant':20s} {'RMSE':>6s} {'R2':>6s} {'MAE':>6s} | top10% RMSE  bias | clean-half bias")
for name in variants:
    p = preds[name][ok]
    print(f"{name:20s} {rmse(o,p):6.2f} {r2(o,p):6.3f} {np.mean(np.abs(p-o)):6.2f} | "
          f"{rmse(o[hi],p[hi]):10.2f} {np.mean(p[hi]-o[hi]):7.2f} | {np.mean(p[lo]-o[lo]):7.2f}")

# ---- the two hard blocks (unweighted predictions)
months = np.repeat(ys_te.time.dt.month.values, H * W)
for name, (clat, clon) in {"Arabian Sea / west coast": (16.9, 70.1), "NE India / Myanmar border": (22.9, 94.1)}.items():
    found = None
    for b in india_blocks:
        i0, j0 = (b // 1000) * 8, (b % 1000) * 8
        if abs(lat[i0:i0+8].mean() - clat) < 0.5 and abs(lon[j0:j0+8].mean() - clon) < 0.8:
            found = b
    if found is None:
        print(f"\n{name}: block not found"); continue
    cells = np.where((block == found) & mask)[0]
    print(f"\n{name}: {len(cells)} India cells; lat {lat[cells // W].min():.1f}-{lat[cells // W].max():.1f}, "
          f"lon {lon[cells % W].min():.1f}-{lon[cells % W].max():.1f}")
    sel = ok & np.isin(cells_te, cells)
    print("  month: mean obs / mean pred")
    for mth in range(1, 13):
        k = sel & (months == mth)
        if k.sum():
            print(f"  {mth:2d}: {yte[k].mean():6.1f} / {preds['unweighted'][k].mean():6.1f}")
