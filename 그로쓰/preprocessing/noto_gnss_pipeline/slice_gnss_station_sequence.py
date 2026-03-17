"""
Slice raw GNSS tabular files into 600-second windows with 300-second overlap.
Normalize each station's E/N/U time series (station-wise, component-wise),
group 8 windows per station, and save a single compressed NPZ with shape
(num_stations, 8, 600, 3).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from shared.paths import GNSS_NOTO_CSV, GNSS_NOTO_PROC
from shared.config import WIN, STRIDE, GNSS_SAMPLING_RATE as SAMPLING_RATE


GNSS_CSV_DIR = GNSS_NOTO_CSV / "enu"
OUT_PATH = GNSS_NOTO_PROC / f"gnss_station_seq_{WIN}_{STRIDE}.npz"
STATION_LIST = GNSS_NOTO_CSV / "stations_within_250km.csv"
LATLON_CSV = GNSS_NOTO_CSV / "stations_latlon.csv"


stations_df = pd.read_csv(STATION_LIST)
target_stations = set(stations_df["station"].astype(str).str.upper())
print("Number of target stations:", len(target_stations))


def get_station_id(fp: Path) -> str | None:
    """
    Extract station ID from filename.

    Example:
        kin_20240101_0001.csv -> 0001
    """
    parts = fp.stem.split("_")
    if len(parts) < 2:
        return None
    return parts[-2].upper()


def load_station_latlon(csv_path: Path) -> dict[str, tuple[float, float]]:
    df = pd.read_csv(csv_path)

    station_dict = {}
    for station, lat, lon in zip(df["station"], df["latitude"], df["longitude"]):
        station_dict[str(station).upper()] = (float(lat), float(lon))

    return station_dict


station_dict = load_station_latlon(LATLON_CSV)
print("Number of stations with lat/lon:", len(station_dict))


def find_header_line(fp: Path, max_lines: int = 200) -> int:
    """
    Return the line index where the table header starts.
    Assumes the header line begins with 'Date/Time'.
    """
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for i in range(max_lines):
            line = f.readline()
            if not line:
                break
            if line.startswith("Date/Time"):
                return i

    raise ValueError(f"Header line not found in first {max_lines} lines: {fp.name}")


def load_gnss_csv(fp: Path) -> pd.DataFrame:
    """
    Load one GNSS file into a DataFrame, skipping metadata lines above
    the actual tabular header.
    """
    header_idx = find_header_line(fp)
    return pd.read_csv(fp, sep=",", skiprows=header_idx, engine="python")


def pick_col(df: pd.DataFrame, key: str) -> str:
    """
    Pick a column name matching 'key' (case-insensitive),
    tolerant to extra text in headers.
    """
    key_lower = key.lower()

    # 1) Prefer exact first-token match
    for col in df.columns:
        first_word = str(col).lower().split()[0]
        if first_word == key_lower:
            return col

    # 2) Fallback: substring match
    for col in df.columns:
        if key_lower in str(col).lower():
            return col

    raise KeyError(f"Column for '{key}' not found. Available columns: {list(df.columns)}")


def slice_windows(arr: np.ndarray, win: int, stride: int, fs: float):
    """
    Slice (T, C) array into:
      - windows: (N, win, C)
      - start_sec: (N,) window start time in seconds relative to series start
    """
    n = len(arr)

    if n < win:
        empty_w = np.empty((0, win, arr.shape[1]), dtype=arr.dtype)
        empty_s = np.empty((0,), dtype=np.float32)
        return empty_w, empty_s

    start_idx = np.arange(0, n - win + 1, stride, dtype=np.int64)
    windows = np.stack([arr[s:s + win] for s in start_idx], axis=0)
    start_sec = (start_idx / fs).astype(np.float32)

    return windows, start_sec


total_windows = 0
processed = 0
skipped_small = 0
skipped_wrong_windows = 0
failed_read = 0
failed_cols = 0
failed_latlon = 0
failed_empty = 0

all_X = []
all_station = []
all_lat = []
all_lon = []
all_start_sec = []
all_end_sec = []


for csv_file in sorted(GNSS_CSV_DIR.glob("*.csv")):
    station = get_station_id(csv_file)

    if station is None:
        print(f"Skipped {csv_file.name}: could not parse station ID")
        continue

    if station not in target_stations:
        continue

    if station not in station_dict:
        failed_latlon += 1
        print(f"Lat/Lon not found for station: {station}")
        continue

    lat, lon = station_dict[station]
    print(f"Processing: {station}")

    try:
        df = load_gnss_csv(csv_file)
    except Exception as e:
        failed_read += 1
        print(f"Failed to read file: {csv_file.name} -> {type(e).__name__}: {e}")
        continue

    try:
        east_col = pick_col(df, "East")
        north_col = pick_col(df, "North")
        up_col = pick_col(df, "Up")
    except Exception as e:
        failed_cols += 1
        print(f"Failed to find required columns in {csv_file.name}: {type(e).__name__}: {e}")
        continue

    # Convert to numeric and drop rows with NaNs
    data = (
        df[[east_col, north_col, up_col]]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .to_numpy(dtype=np.float32)
    )

    if len(data) == 0:
        failed_empty += 1
        print(f"Skipped {station}: no valid numeric ENU rows")
        continue

    # Station-wise, component-wise normalization
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-8] = 1.0

    data = (data - mean) / std

    windows, start_sec = slice_windows(data, WIN, STRIDE, SAMPLING_RATE)

    if windows.shape[0] == 0:
        skipped_small += 1
        print(f"Skipped {station}: too short for one window")
        continue

    if windows.shape[0] != 8:
        skipped_wrong_windows += 1
        print(f"Skipped {station}: expected 8 windows, got {windows.shape[0]}")
        continue

    all_X.append(windows.astype(np.float32))
    all_station.append(station)
    all_lat.append(lat)
    all_lon.append(lon)
    all_start_sec.append(start_sec.astype(np.float32))
    all_end_sec.append((start_sec + WIN).astype(np.float32))

    total_windows += windows.shape[0]
    processed += 1


if len(all_X) == 0:
    raise RuntimeError("No valid stations found. Nothing to save.")


X_all = np.stack(all_X, axis=0)                 # (num_stations, 8, 600, 3)
start_sec_all = np.stack(all_start_sec, axis=0) # (num_stations, 8)
end_sec_all = np.stack(all_end_sec, axis=0)     # (num_stations, 8)

station_all = np.array(all_station)
lat_all = np.array(all_lat, dtype=np.float32)
lon_all = np.array(all_lon, dtype=np.float32)


np.savez_compressed(
    OUT_PATH,
    X=X_all,
    start_sec=start_sec_all,
    end_sec=end_sec_all,
    fs=np.array([SAMPLING_RATE], dtype=np.float32),
    station=station_all,
    lat=lat_all,
    lon=lon_all,
)

print("\nDone.")
print("Saved to:", OUT_PATH)
print("Processed stations:", processed)
print("Stations skipped (too short):", skipped_small)
print("Stations skipped (not 8 windows):", skipped_wrong_windows)
print("Failed to read files:", failed_read)
print("Failed to find required columns:", failed_cols)
print("Failed to find lat/lon:", failed_latlon)
print("Failed empty numeric data:", failed_empty)
print("Total windows generated:", total_windows)
print("Final X shape:", X_all.shape)