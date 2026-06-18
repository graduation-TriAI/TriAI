"""
Slice raw GNSS .tab files into sliding windows
after upsampling to 100 Hz.

Use the pre-matched station CSV as the source of allowed GNSS stations.
Normalize each station's E/N/U time series per component,
slice windows, and save one compressed NPZ.

Expected output shape:
    X : (num_stations, num_windows, sliding_window_sec * 100, 3)
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from shared.paths import GNSS_TOHOKU_PROC, GNSS_TOHOKU_RAW, PAIRS_TOHOKU_CSV
from shared.config import WIN, STRIDE

# =============
# Config
# =============
PAIRS_CSV = PAIRS_TOHOKU_CSV / "tohoku_station_pairs_ver_30km.csv"
TARGET_FS = 100 #upsample to 100 Hz

WIN_SEC = int(WIN)          # 480
STRIDE_SEC = int(STRIDE)    # 240

WIN_SAMPLES = WIN_SEC * TARGET_FS
STRIDE_SAMPLES = STRIDE_SEC * TARGET_FS

EXPECTED_WINDOWS = 21

base_dir = GNSS_TOHOKU_PROC / f"{WIN}_{STRIDE}" / "100hz"
base_dir.mkdir(parents=True, exist_ok=True)

OUT_NPZ = base_dir / (
    f"tohoku_gnss_station_seq_{WIN_SEC}s_{STRIDE_SEC}s_{TARGET_FS}hz_from_pairs30km.npz"
)

STATION_RE = re.compile(r"(GNET\d{4})", re.I)


# ==============================================
# Load allowed station metadata from matched CSV
# ==============================================
def normalize_station(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()

pairs_df = pd.read_csv(PAIRS_CSV)

pairs_df["gnss_station"] = normalize_station(pairs_df["gnss_station"])
pairs_df["seismic_station"] = normalize_station(pairs_df["seismic_station"])
pairs_df["pgv"] = pd.to_numeric(pairs_df["pgv"], errors="coerce")
pairs_df["gnss_lat"] = pd.to_numeric(pairs_df["gnss_lat"], errors="coerce")
pairs_df["gnss_lon"] = pd.to_numeric(pairs_df["gnss_lon"], errors="coerce")
pairs_df["distance_km"] = pd.to_numeric(pairs_df["distance_km"], errors="coerce")

pairs_df = pairs_df.dropna(
    subset=["gnss_station", "seismic_station", "pgv", "gnss_lat", "gnss_lon"]
).reset_index(drop=True)

# gnss_station should ideally be unique
dup_counts = pairs_df.groupby("gnss_station").size()
multi_matched = dup_counts[dup_counts > 1]
if len(multi_matched) > 0:
    print("[Warning] duplicated gnss_station found in pairs CSV:")
    print(multi_matched.head(20))
    pairs_df = pairs_df.drop_duplicates(subset=["gnss_station"], keep="first").reset_index(drop=True)

station_meta = {
    row["gnss_station"]: {
        "seismic_station": row["seismic_station"],
        "gnss_lat": float(row["gnss_lat"]),
        "gnss_lon": float(row["gnss_lon"]),
        "pgv": float(row["pgv"]),
        "distance_km": float(row["distance_km"]) if pd.notna(row["distance_km"]) else np.nan,
    }
    for _, row in pairs_df.iterrows()
}

target_stations = set(station_meta.keys())
print("Number of allowed GNSS stations from pairs CSV:", len(target_stations))


# =========================================================
# File loading helpers
# =========================================================
def find_header_line(fp: Path, max_lines: int = 200) -> int:
    """Return the line index where the tabular header starts."""
    with fp.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            if line.startswith("Date/Time"):
                return i
    raise ValueError(f"Header line not found in first {max_lines} lines: {fp.name}")

def load_gnss_tab(fp: Path) -> pd.DataFrame:
    header_idx = find_header_line(fp)
    return pd.read_csv(fp, sep="\t", skiprows=header_idx, engine="python")

def pick_col(df: pd.DataFrame, key: str) -> str:
    """Pick a column matching key (case-insensitive), tolerant to extra header text."""
    key_lower = key.lower()

    for col in df.columns:
        first_word = str(col).lower().split()[0]
        if first_word == key_lower:
            return col

    for col in df.columns:
        if key_lower in str(col).lower():
            return col

    raise KeyError(f"Column for '{key}' not found. Available columns: {list(df.columns)}")


# =========================================================
# Upsampling / normalization / slicing
# =========================================================
def upsampling(df, east_col, north_col, up_col, time_col="Date/Time local", target_fs=100):
    """
    - keep only needed columns
    - convert time / numeric
    - drop NaN
    - set datetime index
    - resample to target_fs
    - cubic interpolate
    - return (T, 3)
    """
    df_processed = df[[time_col, east_col, north_col, up_col]].copy()
    df_processed[time_col] = pd.to_datetime(df_processed[time_col], errors="coerce")

    for col in [east_col, north_col, up_col]:
        df_processed[col] = pd.to_numeric(df_processed[col], errors="coerce")

    df_processed = df_processed.dropna()

    if len(df_processed) < 2:
        return np.empty((0, 3), dtype=np.float32)
    
    df_processed = df_processed.sort_values(time_col)
    df_processed = df_processed.set_index(time_col)
    df_processed = df_processed[~df_processed.index.duplicated(keep="first")]

    resampling = f"{int(1000 / target_fs)}ms"
    df_upsampled = df_processed.resample(resampling).interpolate(method="cubic").dropna()

    if len(df_upsampled) == 0:
        return np.empty((0, 3), dtype=np.float32)
    
    data = df_upsampled[[east_col, north_col, up_col]].to_numpy(dtype=np.float32)
    return data

def normalize_per_station_component(data: np.ndarray) -> np.ndarray:
    """
    Normalize per station, per component.
    data: (T, 3)
    """
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std[std < 1e-8] = 1.0
    return (data - mean) / std

def slice_windows(arr: np.ndarray, win_samples: int, stride_samples: int, fs: float):
    """
    Slice (T, 3) array into windows.

    Returns:
        windows   : (N, win_samples, 3)
        start_sec : (N,)
        end_sec   : (N,)
    """
    n = len(arr)

    if n < win_samples:
        empty_w = np.empty((0, win_samples, arr.shape[1]), dtype=arr.dtype)
        empty_s = np.empty((0,), dtype=np.float32)
        empty_e = np.empty((0,), dtype=np.float32)
        return empty_w, empty_s, empty_e
    
    start_idx = np.arange(0, n - win_samples + 1, stride_samples, dtype=np.int64)
    windows = np.stack([arr[s:s + win_samples] for s in start_idx], axis=0)

    start_sec = (start_idx / fs).astype(np.float32)
    end_sec = ((start_idx + win_samples) / fs).astype(np.float32)

    return windows, start_sec, end_sec


# =========================================================
# Main
# =========================================================
processed = 0
skipped_not_in_csv = 0
skipped_too_short = 0
skipped_wrong_windows = 0
failed_read = 0
failed_cols = 0
failed_empty_after_upsampling = 0

all_X = []
all_gnss_station = []
all_seismic_station = []
all_pgv = []
all_distance_km = []
all_lat = []
all_lon = []
all_start_sec = []
all_end_sec = []

for tab_file in sorted(GNSS_TOHOKU_RAW.glob("*.tab")):
    m = STATION_RE.search(tab_file.name)
    if m is None:
        continue

    gnss_station = m.group(1).upper()

    if gnss_station not in target_stations:
        skipped_not_in_csv += 1
        continue

    meta = station_meta[gnss_station]
    print(f"Processing: {gnss_station}")

    try:
        df = load_gnss_tab(tab_file)
    except Exception as e:
        failed_read += 1
        print("Failed to read file:", tab_file.name, "->", type(e).__name__, e)
        continue

    try:
        east_col = pick_col(df, "East")
        north_col = pick_col(df, "North")
        up_col = pick_col(df, "Up")
    except Exception as e:
        failed_cols += 1
        print("Failed to find required columns:", type(e).__name__, e)
        continue

    # 1) upsample to 100 Hz
    data = upsampling(df, east_col, north_col, up_col, target_fs=TARGET_FS)

    if len(data) == 0:
        failed_empty_after_upsampling += 1
        print(f"Skipped {gnss_station}: empty after upsampling")
        continue

    # 2) normalize per station / per component
    data = normalize_per_station_component(data)

    # 3) slice into 600-sec windows with 300-sec stride
    windows, start_sec, end_sec = slice_windows(
        data,
        win_samples=WIN_SAMPLES,
        stride_samples=STRIDE_SAMPLES,
        fs=TARGET_FS,
    )

    if windows.shape[0] == 0:
        skipped_too_short += 1
        print(f"Skipped {gnss_station}: too short for one {WIN_SEC}-sec window")
        continue

    # sanity check
    if windows.shape[0] != EXPECTED_WINDOWS:
        skipped_wrong_windows += 1
        print(f"Skipped {gnss_station}: expected {EXPECTED_WINDOWS} windows, got {windows.shape[0]}")
        continue

    all_X.append(windows.astype(np.float32))
    all_gnss_station.append(gnss_station)
    all_seismic_station.append(meta["seismic_station"])
    all_pgv.append(meta["pgv"])
    all_distance_km.append(meta["distance_km"])
    all_lat.append(meta["gnss_lat"])
    all_lon.append(meta["gnss_lon"])
    all_start_sec.append(start_sec.astype(np.float32))
    all_end_sec.append(end_sec.astype(np.float32))

    processed += 1

if len(all_X) == 0:
    raise RuntimeError("No valid stations found. Nothing to save.")

X_all = np.stack(all_X, axis=0)  # (num_stations, 8, 60000, 3)
start_sec_all = np.stack(all_start_sec, axis=0)
end_sec_all = np.stack(all_end_sec, axis=0)

gnss_station_all = np.array(all_gnss_station)
seismic_station_all = np.array(all_seismic_station)
pgv_all = np.array(all_pgv, dtype=np.float32)
distance_km_all = np.array(all_distance_km, dtype=np.float32)
lat_all = np.array(all_lat, dtype=np.float32)
lon_all = np.array(all_lon, dtype=np.float32)

OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUT_NPZ,
    X=X_all,
    start_sec=start_sec_all,
    end_sec=end_sec_all,
    fs=np.array([TARGET_FS], dtype=np.float32),
    gnss_station=gnss_station_all,
    seismic_station=seismic_station_all,
    y=pgv_all,
    distance_km=distance_km_all,
    lat=lat_all,
    lon=lon_all,
)

print("\nDone.")
print("Saved to:", OUT_NPZ)
print("Processed stations:", processed)
print("Skipped not in pairs CSV:", skipped_not_in_csv)
print("Skipped too short:", skipped_too_short)
print("Skipped wrong #windows:", skipped_wrong_windows)
print("Failed to read files:", failed_read)
print("Failed to find required columns:", failed_cols)
print("Failed empty after upsampling:", failed_empty_after_upsampling)
print("X shape:", X_all.shape)
print("start_sec shape:", start_sec_all.shape)
print("end_sec shape:", end_sec_all.shape)
print("y shape:", pgv_all.shape)
print(f"Window length: {WIN_SEC} sec ({WIN_SAMPLES} samples at {TARGET_FS} Hz)")
print(f"Stride: {STRIDE_SEC} sec ({STRIDE_SAMPLES} samples at {TARGET_FS} Hz)")