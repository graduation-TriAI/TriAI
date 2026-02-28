
"""
Extract unique seismic station coordinates from a metadata text file.

This script parses a channel metadata table (text format),
extracts station codes with their latitude and longitude,
and saves the unique stations to a CSV file.
"""

from scripts.common import RAW_DIR, CSV_DIR
import csv

IN_PATH = RAW_DIR / "seismic_station_metadata.txt"
OUT_PATH = CSV_DIR / "stations_latlon.csv"

def parse_channels_table_line(line: str):
    """
    Parse a single line of the channel metadata table.

    Expected format (based on README specification):
        [4]  station code
        [14] latitude
        [15] longitude

    Returns:
        (station_code, lat, lon) if valid,
        None if the line does not match the expected format.
    """
    parts = line.strip().split()
    if len(parts) < 15:
        return None 

    station_code = parts[3]          # [4]
    lat = float(parts[13])           # [14]
    lon = float(parts[14])           # [15]

    return station_code, lat, lon


stations = {}  

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