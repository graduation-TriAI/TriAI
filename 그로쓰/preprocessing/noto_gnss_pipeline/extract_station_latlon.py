from pathlib import Path
import pandas as pd
from shared.paths import GNSS_NOTO_CSV

CSV_DIR = GNSS_NOTO_CSV / "enu"
OUT_CSV = GNSS_NOTO_CSV / "stations_latlon.csv"

def get_station_id(fp: Path):
    return fp.stem.split("_")[-2]

def get_latlon_from_csv(fp: Path) -> tuple[float, float] | None:
    try:
        df = pd.read_csv(
            fp,
            usecols=["Latitude", "Longitude"],
            nrows=1
        )
        lat = float(df["Latitude"].iloc[0])
        lon = float(df["Longitude"].iloc[0])
        return lat, lon
    except Exception:
        return None
    
rows = []

for fp in sorted(CSV_DIR.glob("*.csv")):
    station = get_station_id(fp)
    latlon = get_latlon_from_csv(fp)

    if latlon is None:
        continue

    lat, lon = latlon
    rows.append({
        "station": station,
        "latitude": lat,
        "longitude": lon,
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(df_out.head())