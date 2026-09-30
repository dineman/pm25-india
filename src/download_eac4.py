import sys
import cdsapi

year, month = int(sys.argv[1]), int(sys.argv[2])
import calendar
last = calendar.monthrange(year, month)[1]

c = cdsapi.Client()
c.retrieve(
    "cams-global-reanalysis-eac4",
    {
        "variable": ["particulate_matter_2.5um"],
        "date": [f"{year}-{month:02d}-01/{year}-{month:02d}-{last}"],
        "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
        "data_format": "netcdf",
        "area": [38, 67, 5, 99],  # N, W, S, E
    },
    f"data/raw/eac4_pm25_{year}_{month:02d}.nc",
)
print("done", year, month)
