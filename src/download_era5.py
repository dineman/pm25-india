import calendar, os, time, zipfile
import cdsapi

key = open(os.path.expanduser("~/.era5key")).read().strip()
c = cdsapi.Client(url="https://cds.climate.copernicus.eu/api", key=key)
os.makedirs("data/raw_era5", exist_ok=True)
HOURS = [f"{h:02d}:00" for h in range(24)]

for year in range(2020, 2025):
    for month in range(1, 13):
        folder = f"data/raw_era5/{year}_{month:02d}"
        if os.path.isdir(folder) and any(f.endswith(".nc") for f in os.listdir(folder)):
            print("skip", folder, flush=True)
            continue
        ndays = calendar.monthrange(year, month)[1]
        tmp = f"data/raw_era5/{year}_{month:02d}.download"
        for attempt in range(3):
            try:
                c.retrieve(
                    "reanalysis-era5-single-levels",
                    {
                        "product_type": ["reanalysis"],
                        "variable": ["boundary_layer_height", "total_precipitation"],
                        "year": [str(year)],
                        "month": [f"{month:02d}"],
                        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
                        "time": HOURS,
                        "data_format": "netcdf",
                        "area": [38, 67, 5, 99],
                    },
                    tmp,
                )
                os.makedirs(folder, exist_ok=True)
                if zipfile.is_zipfile(tmp):
                    with zipfile.ZipFile(tmp) as z:
                        z.extractall(folder)
                    os.remove(tmp)
                else:
                    os.replace(tmp, f"{folder}/data.nc")
                print("done", folder, flush=True)
                break
            except Exception as e:
                print("retry", folder, attempt, e, flush=True)
                time.sleep(30)
