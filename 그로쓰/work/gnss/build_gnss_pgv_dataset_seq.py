from pathlib import Path
import numpy as np
import pandas as pd

from shared.paths import GNSS_TOHOKU_PROC, PGV_TOHOKU
from shared.config import WIN, STRIDE


LABEL_CSV = PGV_TOHOKU / "gnss_pgv_labels_15km.csv"
INPUT_PATH = GNSS_TOHOKU_PROC / f"gnss_station_seq_{WIN}_{STRIDE}.npz"
OUT_NPZ = GNSS_TOHOKU_PROC / "gnss_pgv_dataset_15km_seq.npz"


def normalize_station(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def load_label_map(label_csv: Path):
    df = pd.read_csv(label_csv)

    df["gnss_station"] = normalize_station(df["gnss_station"])
    df["seismic_station"] = normalize_station(df["seismic_station"])
    df["pgv"] = pd.to_numeric(df["pgv"], errors="coerce")

    df = df.dropna(subset=["gnss_station", "seismic_station", "pgv"]).reset_index(drop=True)

    dup_counts = df.groupby("gnss_station").size()
    multi_matched = dup_counts[dup_counts > 1]

    if len(multi_matched) > 0:
        print("[Warning] duplicated gnss_station found:")
        print(multi_matched.head(20))

        df = df.drop_duplicates(
            subset=["gnss_station"],
            keep="first"
        ).reset_index(drop=True)

    label_map = {
        row["gnss_station"]: (row["seismic_station"], float(row["pgv"]))
        for _, row in df.iterrows()
    }

    print("Loaded labels:", len(label_map))
    return label_map


def build_dataset_from_single_npz(label_map, input_path: Path, out_npz: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input NPZ not found: {input_path}")

    data = np.load(input_path, allow_pickle=True)

    required_keys = ["X", "station"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"'{key}' not found in {input_path.name}")

    X = data["X"]                       # (num_stations, 8, 600, 3)
    gnss_station = data["station"]     # (num_stations,)

    num_stations = len(gnss_station)

    if len(X) != num_stations:
        raise ValueError(
            f"Mismatch: len(X)={len(X)} but len(station)={num_stations}"
        )

    # optional metadata
    if "start_sec" in data:
        start_sec = data["start_sec"]  # (num_stations, 8)
        has_start_sec = True
    else:
        start_sec = np.full((num_stations, X.shape[1]), -1, dtype=np.float32)
        has_start_sec = False

    if "end_sec" in data:
        end_sec = data["end_sec"]      # (num_stations, 8)
    else:
        end_sec = np.full((num_stations, X.shape[1]), -1, dtype=np.float32)

    if "fs" in data:
        fs_value = np.asarray(data["fs"]).reshape(-1)[0].astype(np.float32)
        fs = np.full((num_stations,), fs_value, dtype=np.float32)
    else:
        fs = np.full((num_stations,), -1, dtype=np.float32)

    if "lat" in data:
        gnss_lat = np.asarray(data["lat"], dtype=np.float32)
    else:
        gnss_lat = np.full((num_stations,), np.nan, dtype=np.float32)

    if "lon" in data:
        gnss_lon = np.asarray(data["lon"], dtype=np.float32)
    else:
        gnss_lon = np.full((num_stations,), np.nan, dtype=np.float32)

    matched_X = []
    matched_y = []
    matched_gnss_station = []
    matched_seismic_station = []
    matched_start_sec = []
    matched_end_sec = []
    matched_fs = []
    matched_gnss_lat = []
    matched_gnss_lon = []

    skipped_no_label = []

    for i in range(num_stations):
        station = str(gnss_station[i]).strip().upper()

        if station not in label_map:
            skipped_no_label.append(station)
            continue

        seismic_station, pgv_value = label_map[station]

        matched_X.append(X[i])  # (8, 600, 3)
        matched_y.append(pgv_value)
        matched_gnss_station.append(station)
        matched_seismic_station.append(seismic_station)
        matched_start_sec.append(start_sec[i])
        matched_end_sec.append(end_sec[i])
        matched_fs.append(fs[i])
        matched_gnss_lat.append(gnss_lat[i])
        matched_gnss_lon.append(gnss_lon[i])

    if not matched_X:
        raise RuntimeError(
            "No matched samples found. Check label CSV and input NPZ station names."
        )

    X_all = np.stack(matched_X, axis=0).astype(np.float32)  # (matched_stations, 8, 600, 3)
    y_all = np.array(matched_y, dtype=np.float32)           # (matched_stations,)
    gnss_all = np.array(matched_gnss_station)
    seismic_all = np.array(matched_seismic_station)
    start_all = np.stack(matched_start_sec, axis=0).astype(np.float32)
    end_all = np.stack(matched_end_sec, axis=0).astype(np.float32)
    fs_all = np.array(matched_fs, dtype=np.float32)
    lat_all = np.array(matched_gnss_lat, dtype=np.float32)
    lon_all = np.array(matched_gnss_lon, dtype=np.float32)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=X_all,
        y=y_all,
        gnss_station=gnss_all,
        seismic_station=seismic_all,
        start_sec=start_all,
        end_sec=end_all,
        fs=fs_all,
        gnss_lat=lat_all,
        gnss_lon=lon_all,
    )

    print("\n[Done]")
    print("Saved to:", out_npz)
    print("Input NPZ:", input_path)
    print("Total stations in input:", num_stations)
    print("Matched stations:", len(X_all))
    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)

    if skipped_no_label:
        print("\nSkipped stations with no label (first 20):")
        print(sorted(set(skipped_no_label))[:20])

    if not has_start_sec:
        print("\n[Info] start_sec not found in input NPZ. Filled with -1.")


if __name__ == "__main__":
    label_map = load_label_map(LABEL_CSV)

    build_dataset_from_single_npz(
        label_map,
        INPUT_PATH,
        OUT_NPZ,
    )