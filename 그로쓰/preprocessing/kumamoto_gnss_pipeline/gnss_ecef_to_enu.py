import math
from pathlib import Path

import pandas as pd
import numpy as np

from shared.paths import GNSS_HOKKAIDO_CSV, GNSS_HOKKAIDO_RAW

# =========================
# paths
# =========================
input_dir = GNSS_HOKKAIDO_RAW / "hokkaido_ecef_txt"
output_dir = GNSS_HOKKAIDO_CSV / "hokkaido_enu_45min"
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# settings
# =========================
LEAP_SECONDS_2016 = 17

# Hokkaido Eastern Iburi mainshock UTC
origin_time = pd.Timestamp("2018-09-05 18:07:59")

# 정확히 origin_time 기준 16분 전 ~ 29분 후
start_time = (origin_time - pd.Timedelta(minutes=16)).floor("min")
end_time = (origin_time + pd.Timedelta(minutes=29)).floor("min")

print("Origin time:", origin_time)
print("Cut start:", start_time)
print("Cut end:", end_time)

# =========================
# WGS84 constants
# =========================
A = 6378137.0
F = 1 / 298.257223563
B = A * (1 - F)
E2 = F * (2 - F)
EP2 = (A**2 - B**2) / B**2


def ecef_to_geodetic(x, y, z):
    lon = math.atan2(y, x)
    p = math.sqrt(x**2 + y**2)
    theta = math.atan2(z * A, p * B)

    lat = math.atan2(
        z + EP2 * B * math.sin(theta) ** 3,
        p - E2 * A * math.cos(theta) ** 3
    )

    sin_lat = math.sin(lat)
    N = A / math.sqrt(1 - E2 * sin_lat**2)
    h = p / math.cos(lat) - N

    return math.degrees(lat), math.degrees(lon), h


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


def robust_smooth_detrend(out_df: pd.DataFrame, window_sec: int = 301) -> pd.DataFrame:
    """
    PPP drift가 큰 경우용.
    전체 45분 ENU에서 rolling median trend를 추정해서 제거한 뒤,
    첫 60초 평균으로 다시 zero-centering.
    """
    cols = ["East [cm]", "North [cm]", "Up [cm]"]

    out_df = out_df.copy()
    dt = pd.to_datetime(out_df["Date/Time"], format="%Y/%m/%d %H:%M:%S")

    for col in cols:
        y = out_df[col].astype(float)

        # 5분 정도의 smooth trend 추정
        trend = (
            y.rolling(window=window_sec, center=True, min_periods=30)
             .median()
             .interpolate(limit_direction="both")
        )

        out_df[col] = y - trend

    # 첫 60초 평균을 0으로 맞춤
    baseline_mask = (
        dt >= dt.iloc[0]
    ) & (
        dt < dt.iloc[0] + pd.Timedelta(seconds=60)
    )

    baseline = out_df.loc[baseline_mask, cols].mean()

    for col in cols:
        out_df[col] = out_df[col] - baseline[col]

    return out_df

def load_ppp_ecef_file(file_path: Path) -> pd.DataFrame:
    rows = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("%"):
                continue

            parts = line.split()

            # gps_week gps_seconds x y z Q ns ...
            if len(parts) < 7:
                continue

            try:
                gps_week = int(parts[0])
                gps_seconds = float(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                q = int(parts[5])
                ns = int(parts[6])
            except ValueError:
                continue

            rows.append([gps_week, gps_seconds, x, y, z, q, ns])

    if not rows:
        raise ValueError(f"No valid PPP ECEF rows found: {file_path}")

    df = pd.DataFrame(
        rows,
        columns=["gps_week", "gps_seconds", "X", "Y", "Z", "Q", "ns"]
    )

    gps_epoch = pd.Timestamp("1980-01-06 00:00:00")

    df["datetime_gpst"] = (
        gps_epoch
        + pd.to_timedelta(df["gps_week"] * 7, unit="D")
        + pd.to_timedelta(df["gps_seconds"], unit="s")
    )

    # 2016년 4월 기준 GPST = UTC + 17s
    df["datetime_utc"] = df["datetime_gpst"] - pd.Timedelta(seconds=LEAP_SECONDS_2016)

    return df


def convert_one_station(file_path: Path):
    df = load_ppp_ecef_file(file_path)

    q_counts = df["Q"].value_counts().sort_index().to_dict()
    print(f"\n[{file_path.name}] Q counts:", q_counts)

    # PPP solution만 사용
    df = df[df["Q"] == 6].copy()

    if df.empty:
        raise ValueError(f"No PPP rows Q=6 in {file_path.name}")

    # 시간 정렬
    df = df.sort_values("datetime_utc").reset_index(drop=True)

    # 45-min window 먼저 자르기
    window_mask = (
        (df["datetime_utc"] >= start_time)
        & (df["datetime_utc"] <= end_time)
    )

    df_win = df.loc[window_mask].copy().reset_index(drop=True)

    if df_win.empty:
        raise ValueError(f"No rows in 45-min window: {file_path.name}")

    # reference = window 시작 후 첫 60초 평균
    ref_start = start_time
    ref_end = start_time + pd.Timedelta(seconds=60)

    ref_df = df_win[
        (df_win["datetime_utc"] >= ref_start)
        & (df_win["datetime_utc"] < ref_end)
    ].copy()

    if ref_df.empty:
        raise ValueError(f"No reference rows in first 60 sec: {file_path.name}")

    x0 = ref_df["X"].mean()
    y0 = ref_df["Y"].mean()
    z0 = ref_df["Z"].mean()

    lat0, lon0, h0 = ecef_to_geodetic(x0, y0, z0)

    east_cm = []
    north_cm = []
    up_cm = []

    for _, row in df_win.iterrows():
        e_m, n_m, u_m = ecef_to_enu(
            row["X"], row["Y"], row["Z"],
            x0, y0, z0,
            lat0, lon0
        )
        east_cm.append(e_m * 100.0)
        north_cm.append(n_m * 100.0)
        up_cm.append(u_m * 100.0)

    out_df = pd.DataFrame({
        "Date/Time": df_win["datetime_utc"].dt.strftime("%Y/%m/%d %H:%M:%S"),
        "Latitude": lat0,
        "Longitude": lon0,
        "Height [m]": h0,
        "East [cm]": east_cm,
        "North [cm]": north_cm,
        "Up [cm]": up_cm,
        "Q": df_win["Q"].values,
        "ns": df_win["ns"].values,
    })

    out_df = robust_smooth_detrend(out_df, window_sec=301)

    station_name = file_path.stem.replace("_ppp_ecef", "")
    output_file = output_dir / f"{station_name}_enu_45min.csv"

    out_df.to_csv(output_file, index=False)

    print(
        f"[Saved] {output_file.name} | rows={len(out_df)} | "
        f"lat={lat0:.6f}, lon={lon0:.6f}"
    )


# 한 폴더 안의 *_ppp_ecef.txt만 처리
file_list = sorted(input_dir.glob("*_ppp_ecef.txt"))

print(f"\nFound {len(file_list)} PPP ECEF files")

success = 0
failed = 0

for fp in file_list:
    try:
        convert_one_station(fp)
        success += 1
    except Exception as e:
        failed += 1
        print(f"[Failed] {fp.name}: {e}")

print("\nDone.")
print("Success:", success)
print("Failed:", failed)