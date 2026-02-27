
"""
Slice raw GNSS .tab files into 600-second windows with 300-second overlap.
Save windows per station as compressed NPZ files, and report stations skipped
due to insufficient data length or parsing issues.
"""

from pathlib import Path
from scripts.common import CSV_DIR, NPZ_DIR, DATASET_DIR
import re
import pandas as pd
import numpy as np

STATION_LIST = CSV_DIR / "stations_within_300km.csv"
NPZ_DIR.mkdir(parents=True, exist_ok=True)

WIN = 600
STRIDE = 300

STATION_RE = re.compile(r"(GNET\d{4})", re.I)

stations_df = pd.read_csv(STATION_LIST)
target_stations = set(stations_df["station"].astype(str).str.upper().tolist())
print("Number of target stations:", len(target_stations))


def find_header_line(fp: Path, max_lines: int = 200) -> int:
    """Return the line index where the tabular header starts (line begins with 'Date/Time')."""
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for i in range(max_lines):
            line = f.readline()
            if not line:
                break
            if line.startswith("Date/Time"):
                return i
    raise ValueError(f"Header line not found in first {max_lines} lines: {fp.name}")


def load_gnss_tab(fp: Path) -> pd.DataFrame:
    """Load a GNSS .tab file into a DataFrame, skipping metadata lines before the 'Date/Time' header."""
    header_idx = find_header_line(fp)
    df = pd.read_csv(fp, sep="\t", skiprows=header_idx, engine="python")
    return df


def slice_windows(arr: np.ndarray, win: int, stride: int) -> np.ndarray:
    """Slice (T,C) array into (N, win, C) windows using a fixed stride."""
    n = len(arr)
    if n < win:
        return np.empty((0, win, arr.shape[1]), dtype=arr.dtype)
    starts = range(0, n - win + 1, stride)
    return np.stack([arr[s:s+win] for s in starts], axis=0)


total_windows = 0
processed = 0
skipped_small = 0
failed_read = 0
failed_cols = 0

for tab_file in sorted(DATASET_DIR.glob("*.tab")):
    m = STATION_RE.search(tab_file.name)
    if m is None:
        continue
    station = m.group(1).upper()
    if station not in target_stations:
        continue

    print("Processing...:", station)

    try:
        df = load_gnss_tab(tab_file)
    except Exception as e:
        failed_read += 1
        print("Failed to read file:", tab_file.name, "->", type(e).__name__, e)
        continue

    def pick_col(df, key: str) -> str:
        """Pick a column name that matches 'key' (case-insensitive), tolerant to extra text in headers."""
        key_lower = key.lower()

        #Prefer exact first-token match (e.g., "East", "East(cm)")
        for col in df.columns:
            first_word = str(col).lower().split()[0]
            if first_word == key_lower:
                return col

        #Fallback: substring match
        for col in df.columns:
            if key_lower in str(col).lower():
                return col

        raise KeyError(f"Column for '{key}' not found. Available columns: {list(df.columns)}")

    try:
        east_col = pick_col(df, "East")
        north_col = pick_col(df, "North")
        up_col = pick_col(df, "Up")
    except Exception as e:
        failed_cols += 1
        print("Failed to find required columns:", type(e).__name__, e)
        continue

    #Convert to numeric and drop rows with NaNs (non-numeric/missing samples)
    data = df[[east_col, north_col, up_col]].apply(pd.to_numeric, errors="coerce").dropna().to_numpy()
    # data shape: (T,3)

    windows = slice_windows(data, WIN, STRIDE)
    if windows.shape[0] == 0:
        skipped_small += 1
        continue

    out_path = NPZ_DIR / f"{station}_600s_300s.npz"
    np.savez_compressed(out_path, X=windows)

    total_windows += windows.shape[0]
    processed += 1

print("\nDone.")
print("Processed stations:", processed)
print("Stations with no windows (too short, etc.):", skipped_small)
print("Failed to read files:", failed_read)
print("Failed to find required columns:", failed_cols)
print("Total windows generated:", total_windows)
