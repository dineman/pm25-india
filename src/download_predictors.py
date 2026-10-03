import calendar, os, sys, time
import cdsapi

VARS = [
    "2m_temperature", "2m_dewpoint_temperature",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "boundary_layer_height", "surface_pressure",
    "total_precipitation", "total_cloud_cover",
    "total_aerosol_optical_depth_550nm",
    "total_column_carbon_monoxide",
    "total_column_nitrogen_dioxide",
    "total_column_sulphur_dioxide",
]
TIMES = ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"]
os.makedirs("data/raw_pred", exist_ok=True)
c = cdsapi.Client()

if len(sys.argv) == 3:
    periods = [(int(sys.argv[1]), int(sys.argv[2]))]   # test mode
else:
    periods = [(y, m) for y in range(2020, 2025) for m in range(1, 13)]

for year, month in periods:
    out = f"data/raw_pred/eac4_pred_{year}_{month:02d}.nc"
    if os.path.exists(out):
        print("skip", out, flush=True); continue
    last = calendar.monthrange(year, month)[1]
    for attempt in range(3):
        try:
            c.retrieve(
                "cams-global-reanalysis-eac4",
                {
                    "variable": VARS,
                    "date": [f"{year}-{month:02d}-01/{year}-{month:02d}-{last}"],
                    "time": TIMES,
                    "data_format": "netcdf",
                    "area": [38, 67, 5, 99],
                },
                out + ".part",
            )
            os.rename(out + ".part", out)
            print("done", out, flush=True); break
        except Exception as e:
            print("retry", out, attempt, e, flush=True); time.sleep(30)
