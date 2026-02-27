from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent.parent

TXT_PATH = BASE_DIR / "data" / "txt"
CSV_PATH = BASE_DIR / "data" / "csv"
IN_PATH = TXT_PATH / "seismic_station_metadata.txt"
OUT_PATH = CSV_PATH / "stations_latlon.csv"

def parse_channels_table_line(line: str):
    parts = line.strip().split()
    if len(parts) < 15:
        return None  # 최소한 lon(15번째)까지는 있어야 함

    station_code = parts[3]          # [4]
    lat = float(parts[13])           # [14]
    lon = float(parts[14])           # [15]

    return station_code, lat, lon


stations = {}  # station_code -> (lat, lon)

with IN_PATH.open("r", encoding="utf-8", errors="replace") as f:
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        parsed = parse_channels_table_line(line)
        if not parsed:
            continue

        code, lat, lon = parsed

        if code not in stations:
            stations[code] = (lat, lon)

with OUT_PATH.open("w", newline="", encoding="utf-8") as fo:
    w = csv.writer(fo)
    w.writerow(["station", "latitude", "longitude"])
    for code in sorted(stations.keys()):
        lat, lon = stations[code]
        w.writerow([code, f"{lat:.6f}", f"{lon:.6f}"])

print(f"Saved: {OUT_PATH}  (stations: {len(stations)})")