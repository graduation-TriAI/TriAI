import math
from pathlib import Path
import pandas as pd
import numpy as np

from shared.paths import GNSS_KUMAMOTO_CSV, GNSS_KUMAMOTO_RAW

# =========================
# paths
# =========================
input_dir = GNSS_KUMAMOTO_RAW / "ppp_outputs"  # 24개 *_ppp_ecef.txt 있는 폴더
output_dir = GNSS_KUMAMOTO_CSV / "kumamoto_enu_45min"
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# settings
# =========================
LEAP_SECONDS_2016 = 17

# Kumamoto mainshock UTC
origin_time = pd.Timestamp("2016-04-15 16:25:06")
origin_minute = origin_time.floor("min")   # 2016-04-15 16:25:00

# Match Noto-style window: approx. 16 min before, 29 min after
start_time = origin_minute - pd.Timedelta(minutes=16)
end_time = origin_minute + pd.Timedelta(minutes=29)

print("Origin minute:", origin_minute)
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

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    lat = math.atan2(
        z + EP2 * B * sin_theta**3,
        p - E2 * A * cos_theta**3
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

def remove_linear_trend(series: pd.Series) -> pd.Series:
    """
    Linear detrend without scipy.
    y_detrended = y - fitted_linear_trend
    """
    y = series.astype(float).to_numpy()
    x = list(range(len(y)))

    if len(y) < 2:
        return series

    x_mean = sum(x) / len(x)
    y_mean = y.mean()

    denom = sum((xi - x_mean) ** 2 for xi in x)
    if denom == 0:
        return series - y_mean

    slope = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)) / denom
    intercept = y_mean - slope * x_mean

    trend = [slope * xi + intercept for xi in x]
    return pd.Series(y - trend, index=series.index)


def polynomial_pre_event_detrend(out_df: pd.DataFrame, degree: int = 3) -> pd.DataFrame:
    """
    Remove nonlinear PPP convergence drift.
    Fit polynomial trend using pre-event interval only,
    subtract it from the whole 45-min window,
    then re-zero using first 60 seconds.
    """
    cols = ["East [cm]", "North [cm]", "Up [cm]"]

    out_df = out_df.copy()
    dt = pd.to_datetime(out_df["Date/Time"], format="%Y/%m/%d %H:%M:%S")

    # seconds from start of window
    t = (dt - dt.iloc[0]).dt.total_seconds().to_numpy()

    # use only pre-event data for drift fitting
    pre_mask = dt < origin_time

    if pre_mask.sum() < degree + 2:
        raise ValueError("Not enough pre-event rows for polynomial detrend")

    t_pre = t[pre_mask]

    for col in cols:
        y = out_df[col].astype(float).to_numpy()
        y_pre = y[pre_mask]

        coeff = np.polyfit(t_pre, y_pre, deg=degree)
        trend = np.polyval(coeff, t)

        out_df[col] = y - trend

    # re-zero using first 60 sec
    baseline = out_df.loc[:59, cols].mean()
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

            # format:
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

    # GPS epoch: 1980-01-06 00:00:00 GPST
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

    # PPP만 사용
    df = df[df["Q"] == 6].copy()

    if df.empty:
        raise ValueError(f"No PPP rows Q=6 in {file_path.name}")

    # reference = pre-event average
    # reference = cut window 시작 후 1분 평균
    ref_start = start_time
    ref_end = start_time + pd.Timedelta(minutes=1)

    ref_df = df[
        (df["datetime_utc"] >= ref_start) &
        (df["datetime_utc"] < ref_end)
    ].copy()

    if ref_df.empty:
        raise ValueError(f"No reference rows in {file_path.name}")

    x0 = ref_df["X"].mean()
    y0 = ref_df["Y"].mean()
    z0 = ref_df["Z"].mean()

    lat0, lon0, h0 = ecef_to_geodetic(x0, y0, z0)

    east_cm = []
    north_cm = []
    up_cm = []

    for _, row in df.iterrows():
        e_m, n_m, u_m = ecef_to_enu(
            row["X"], row["Y"], row["Z"],
            x0, y0, z0,
            lat0, lon0
        )
        east_cm.append(e_m * 100.0)
        north_cm.append(n_m * 100.0)
        up_cm.append(u_m * 100.0)

    out_df = pd.DataFrame({
        "Date/Time": df["datetime_utc"].dt.strftime("%Y/%m/%d %H:%M:%S"),
        "Latitude": lat0,
        "Longitude": lon0,
        "Height [m]": h0,
        "East [cm]": east_cm,
        "North [cm]": north_cm,
        "Up [cm]": up_cm,
        "Q": df["Q"].values,
        "ns": df["ns"].values,
    })

    # 45-min window, minute-boundary 기준
    mask = (
        (df["datetime_utc"] >= start_time)
        & (df["datetime_utc"] <= end_time)
    )

    out_df = out_df.loc[mask].reset_index(drop=True)

    # Remove PPP drift and re-zero baseline
    out_df = polynomial_pre_event_detrend(out_df, degree=3)

    station_name = file_path.stem.replace("_ppp_ecef", "")
    output_file = output_dir / f"{station_name}_enu_45min.csv"
    out_df.to_csv(output_file, index=False)

    print(
        f"[Saved] {output_file.name} | rows={len(out_df)} | "
        f"lat={lat0:.6f}, lon={lon0:.6f}"
    )


file_list = sorted(input_dir.glob("*_ppp_ecef.txt"))

print(f"Found {len(file_list)} PPP ECEF files")

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