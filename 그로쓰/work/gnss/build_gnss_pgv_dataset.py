from pathlib import Path
import re
import numpy as np
import pandas as pd
from shared.paths import GNSS_TOHOKU_PROC, PGV_TOHOKU
from shared.config import WIN, STRIDE

LABEL_CSV = PGV_TOHOKU / "gnss_pgv_labels_15km.csv"
GNSS_NPZ_DIR = GNSS_TOHOKU_PROC / f"gnss_windowed_{WIN}_{STRIDE}"
OUT_NPZ = GNSS_TOHOKU_PROC / "gnss_pgv_dataset_15km.npz"

ST_RE = re.compile(r"(GNET\d{4})", re.I)

def normalize_station(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()

def load_label_map(label_csv: Path):
    df = pd.read_csv(label_csv)
    
    df["gnss_station"] = normalize_station(df["gnss_station"])
    df["seismic_station"] = normalize_station(df["seismic_station"])

    df["pgv"] = pd.to_numeric(df["pgv"], errors="coerce")

    df = df.dropna(subset=["pgv"]).reset_index(drop=True)

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
        row["gnss_station"] : (row["seismic_station"], float(row["pgv"]))
        for _, row in df.iterrows()
    }

    print("Loaded labels:", len(label_map))
    return label_map

def extract_station_from_filename(filename: str):
    m = ST_RE.search(filename)

    if m:
        return m.group(1).strip().upper()
    
    return None

def process_one_npz(fp: Path, label_map):
    gnss_station = extract_station_from_filename(fp.name)

    if gnss_station is None:
        return None, None, False
    
    if gnss_station not in label_map:
        return None, gnss_station, False
    
    seismic_station, pgv_value = label_map[gnss_station]

    data = np.load(fp, allow_pickle=True)

    if "X" not in data:
        print(f"[Skip] No 'X' in {fp.name}")
        return None, gnss_station, False
    
    X = data["X"]

    if len(X) == 0:
        return None, gnss_station, False
    
    n = len(X)

    # ---- time metadata ----
    if "start_sec" in data:
        start_sec = data["start_sec"]
        has_start_sec = True
    else:
        start_sec = np.full(n, -1)
        has_start_sec = False

    if "end_sec" in data:
        end_sec = data["end_sec"]
    else:
        end_sec = np.full(n, -1)

    # ---- station/file metadata ----
    if "fs" in data:
        fs = np.full(n, data["fs"], dtype=np.float32)
    else:
        fs = np.full(n, -1, dtype=np.float32)
    
    if "lat" in data:
        gnss_lat = np.full(n, data["lat"], dtype=np.float32)
    else:
        gnss_lat = np.full(n, np.nan, dtype=np.float32)
    
    if "lon" in data:
        gnss_lon = np.full(n, data["lon"], dtype=np.float32)
    else:
        gnss_lon = np.full(n, np.nan, dtype=np.float32)

    item = {
        "X": X,
        "y": np.full(n, pgv_value, dtype=np.float32),
        "gnss_station": np.array([gnss_station] * n),
        "seismic_station": np.array([seismic_station] * n),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "fs": fs,
        "gnss_lat": gnss_lat,
        "gnss_lon": gnss_lon,
    }

    return item, gnss_station, has_start_sec

def build_dataset(label_map, gnss_npz_dir: Path, out_npz: Path):
    chunks = []

    skipped_no_label = []
    skipped_no_start_sec = []

    used_files = 0

    for fp in sorted(gnss_npz_dir.glob("*.npz")):
        item, gnss_station, has_start_sec = process_one_npz(fp, label_map)

        if item is None:
            if gnss_station and gnss_station not in label_map:
                skipped_no_label.append(gnss_station)
            
            continue
        if not has_start_sec:
            skipped_no_start_sec.append(gnss_station)
        
        chunks.append(item)

        used_files += 1

    if not chunks:
        raise RuntimeError(
            "No matched samples found. Check label CSV and GNSS NPZ path."
        )
    
    X_all = np.concatenate([c["X"] for c in chunks], axis=0)
    y_all = np.concatenate([c["y"] for c in chunks], axis=0)
    gnss_all = np.concatenate([c["gnss_station"] for c in chunks], axis=0)
    seismic_all = np.concatenate([c["seismic_station"] for c in chunks], axis=0)
    start_all = np.concatenate([c["start_sec"] for c in chunks], axis=0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        X=X_all,
        y=y_all,
        gnss_station=gnss_all,
        seismic_station=seismic_all,
        start_sec=start_all,
    )

    print("\n[Done]")
    print("Saved to:", out_npz)
    print("Used GNSS files:", used_files)
    print("Total samples:", len(X_all))
    print("X shape:", X_all.shape)
    print("y shape:", y_all.shape)

    if skipped_no_label:
        print("\nSkipped stations with no label (first 20):")
        print(sorted(set(skipped_no_label))[:20])

    if skipped_no_start_sec:
        print("\nStations without start_sec (first 20):")
        print(sorted(set(skipped_no_start_sec))[:20])
    
if __name__ == "__main__":
    label_map = load_label_map(LABEL_CSV)

    build_dataset(
        label_map,
        GNSS_NPZ_DIR,
        OUT_NPZ,
    )