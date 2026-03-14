import math
from pathlib import Path
from shared.paths import GNSS_NOTO_RAW, GNSS_NOTO_CSV
import pandas as pd

input_dir = GNSS_NOTO_RAW 
output_dir = GNSS_NOTO_CSV / "enu"

# =========================
# WGS84 constants
# =========================
A = 6378137.0
F = 1 / 298.257223563
B = A * (1 - F)
E2 = F * (2 - F)
EP2 = (A**2 - B**2) / B**2


# =========================
# ECEF -> Geodetic
# =========================
def ecef_to_geodetic(x, y, z):
    lon = math.atan2(y, x)
    p = math.sqrt(x**2 + y**2)
    theta = math.atan2(z * A, p * B)

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    lat = math.atan2(
        z + EP2 * B * sin_theta**3,
        p - E2 * A * cos_theta**3
    )

    sin_lat = math.sin(lat)
    N = A / math.sqrt(1 - E2 * sin_lat**2)
    h = p / math.cos(lat) - N

    lat_deg = math.degrees(lat)
    lon_deg = math.degrees(lon)

    return lat_deg, lon_deg, h


# =========================
# ECEF -> ENU
# =========================
def ecef_to_enu(x, y, z, x0, y0, z0, lat0_deg, lon0_deg):
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    dx = x - x0
    dy = y - y0
    dz = z - z0

    east = -math.sin(lon0) * dx + math.cos(lon0) * dy

    north = (
        -math.sin(lat0) * math.cos(lon0) * dx
        -math.sin(lat0) * math.sin(lon0) * dy
        + math.cos(lat0) * dz
    )

    up = (
        math.cos(lat0) * math.cos(lon0) * dx
        + math.cos(lat0) * math.sin(lon0) * dy
        + math.sin(lat0) * dz
    )

    return east, north, up


def load_ecef_file(file_path: Path) -> pd.DataFrame:
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            date_str = parts[0]
            time_str = parts[1]

            try:
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
            except ValueError:
                continue

            rows.append([date_str, time_str, x, y, z])
    
    if not rows:
        raise ValueError(f"No valid ECEF rows found in: {file_path}")
    
    df = pd.DataFrame(rows, columns=["date", "time", "X", "Y", "Z"])
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y/%m/%d %H:%M:%S",
        errors="raise"
    )
    return df


def convert_one_station_file(file_path: Path, output_dir: Path, origin_time: pd.Timestamp):
    df = load_ecef_file(file_path)

    #pre-event time interval
    pre_df = df[df["datetime"] < origin_time].copy()
    if len(pre_df) == 0:
        raise ValueError(f"No pre-event rows before origin time in: {file_path.name}")
    
    # reference point = average ECEF of pre-event
    x0 = pre_df["X"].mean()
    y0 = pre_df["Y"].mean()
    z0 = pre_df["Z"].mean()

    # reference lat/lon/h
    lat0, lon0, h0 = ecef_to_geodetic(x0, y0, z0)

    # calculate ENU (cm)
    east_cm_list = []
    north_cm_list = []
    up_cm_list = []

    for _, row in df.iterrows():
        e_m, n_m, u_m = ecef_to_enu(
            row["X"], row["Y"], row["Z"],
            x0, y0, z0,
            lat0, lon0
        )
        east_cm_list.append(e_m * 100.0)
        north_cm_list.append(n_m * 100.0)
        up_cm_list.append(u_m * 100.0)

    out_df = pd.DataFrame({
        "Date/Time": df["datetime"].dt.strftime("%Y/%m/%d %H:%M:%S"),
        "Latitude": lat0,
        "Longitude": lon0,
        "Height [m]": h0,
        "East [cm]": east_cm_list,
        "North [cm]": north_cm_list,
        "Up [cm]": up_cm_list,
    })

    end_time = pd.Timestamp("2024-01-01 07:39:59")
    start_time = end_time - pd.Timedelta(minutes=45)

    mask = (df["datetime"] >= start_time) & (df["datetime"] <= end_time)

    out_df = out_df.loc[mask].reset_index(drop=True)

    station_name = file_path.stem
    output_file = output_dir / f"{station_name}_enu.csv"
    out_df.to_csv(output_file, index=False)

    print(f"[Saved] {output_file.name} | rows={len(out_df)} | lat={lat0:.6f}, lon={lon0:.6f}")


def batch_convert_all(input_dir: Path, output_dir: Path, origin_time_str: str):
    origin_time = pd.Timestamp(origin_time_str)

    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = sorted([
        p for p in input_dir.iterdir()
        if p.is_file() 
    ])

    if not file_list:
        print("No matching files found.")
        return 
    
    print(f"found {len(file_list)} files.\n")

    success = 0
    failed = 0

    for fp in file_list:
        try:
            convert_one_station_file(fp, output_dir, origin_time)
            success += 1
        except Exception as e:
            failed += 1
            print(f"[Failed] {fp.name}: {e}")

    print("\nDone.")
    print(f"Success: {success}")
    print(f"Failed: {failed}")


if __name__ == "__main__":

    # Origin time of the Noto mainshock (UTC)
    origin_time_str = "2024-01-01 07:10:09"

    batch_convert_all(input_dir, output_dir, origin_time_str)