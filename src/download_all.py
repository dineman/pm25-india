import calendar, os, time
import cdsapi

c = cdsapi.Client()
TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]

for year in range(2020, 2025):
    for month in range(1, 13):
        out = f"data/raw/eac4_pm25_{year}_{month:02d}.nc"
        if os.path.exists(out):
            print("skip", out, flush=True)
            continue
        last = calendar.monthrange(year, month)[1]
        for attempt in range(3):
            try:
                c.retrieve(
                    "cams-global-reanalysis-eac4",
                    {
                        "variable": ["particulate_matter_2.5um"],
                        "date": [f"{year}-{month:02d}-01/{year}-{month:02d}-{last}"],
                        "time": TIMES,
                        "data_format": "netcdf",
                        "area": [38, 67, 5, 99],
                    },
                    out + ".part",
                )
                os.rename(out + ".part", out)
                print("done", out, flush=True)
                break
            except Exception as e:
                print("retry", out, attempt, e, flush=True)
                time.sleep(30)
