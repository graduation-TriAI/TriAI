
"""
Parse GNSS .tab files to extract station ID, latitude, and longitude
from file headers, then save a station-level summary CSV.
"""

from scripts.common import CSV_DIR, DATASET_DIR
from pathlib import Path
import re
import pandas as pd

OUT = CSV_DIR / "stations_latlon_1221.csv"

#Extract station ID like GNET0001 from filename
ST_RE = re.compile(r"(GNET\d{4})", re.I)

#Extract lat/lon from header lines (Coverage or Event section)
COVER_RE = re.compile(r"Coverage:.*?LATITUDE:\s*([-\d.]+).*?LONGITUDE:\s*([-\d.]+)", re.I)
EVENT_RE = re.compile(r"Event\(s\):.*?LATITUDE:\s*([-\d.]+).*?LONGITUDE:\s*([-\d.]+)", re.I)

def get_station_id(filename: str) -> str | None:
    m = ST_RE.search(filename)
    return m.group(1).upper() if m else None

def get_latlon_from_header(fp: Path, max_lines: int = 120) -> tuple[float, float] | None:
    """
    Read the first 'max_lines' lines of a .tab file and 
    extract (latitude, longitude) from header using regex.
    Returns None if not found.
    """
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            m = COVER_RE.search(line)
            if m:
                return float(m.group(1)), float(m.group(2))
            m = EVENT_RE.search(line)
            if m:
                return float(m.group(1)), float(m.group(2))
    return None

rows = []
miss_station = 0
miss_latlon = 0

for fp in sorted(DATASET_DIR.glob("*.tab")):
    st = get_station_id(fp.name)
    if st is None:
        miss_station += 1
        continue

    latlon = get_latlon_from_header(fp)
    if latlon is None:
        miss_latlon += 1
        continue

    lat, lon = latlon
    rows.append({"station": st, "latitude": lat, "longitude": lon})

df = pd.DataFrame(rows)

df.to_csv(OUT, index=False, encoding="utf-8-sig")

print("[OK] saved:", OUT)
print("files scanned:", len(list(DATASET_DIR.glob('*.tab'))))
print("rows extracted:", len(df))
print("unique stations:", df.shape[0])
print("missing station-id in filename:", miss_station)
print("missing lat/lon in header:", miss_latlon)
print(df.head(10).to_string(index=False))
