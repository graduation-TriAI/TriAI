import pandas as pd
import numpy as np
from pykrige.ok import OrdinaryKriging

from shared.paths import POST

INPUT_CSV_PATH = POST / "noto_pred_for_kriging_25km_360_180.csv"
OUTPUT_GRID_CSV = POST / "noto_kriging_pred_25km_360_180.csv"

GRID_RES = 0.02
MARGIN = 0.2
VARIOGRAM_MODEL = "spherical"

# 크리깅 입력용 CSV 불러오기
df = pd.read_csv(INPUT_CSV_PATH)

df = df[["seismic_station", "seismic_lat", "seismic_lon", "pgv_pred"]].copy()

df["seismic_station"] = (
    df["seismic_station"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["seismic_lat"] = pd.to_numeric(df["seismic_lat"], errors="coerce")
df["seismic_lon"] = pd.to_numeric(df["seismic_lon"], errors="coerce")
df["pgv_pred"] = pd.to_numeric(df["pgv_pred"], errors="coerce")

df = df.dropna(subset=["seismic_station", "seismic_lat", "seismic_lon", "pgv_pred"])

print("크리깅 입력 행 수:", len(df))

# 크리깅 입력 준비
lons = df["seismic_lon"].values
lats = df["seismic_lat"].values
pgv_pred = df["pgv_pred"].values

# 0.02도 간격 grid 생성
grid_lon = np.arange(
    lons.min() - MARGIN,
    lons.max() + MARGIN + GRID_RES,
    GRID_RES
)

grid_lat = np.arange(
    lats.min() - MARGIN,
    lats.max() + MARGIN + GRID_RES,
    GRID_RES
)

print("grid_lon 개수:", len(grid_lon))
print("grid_lat 개수:", len(grid_lat))
print("격자 해상도:", GRID_RES, "도")

# Ordinary Kriging
OK = OrdinaryKriging(
    lons,
    lats,
    pgv_pred,
    variogram_model=VARIOGRAM_MODEL,
    verbose=False,
    enable_plotting=False
)

z_pred, z_var = OK.execute("grid", grid_lon, grid_lat)

z_pred = np.asarray(z_pred)
z_var = np.asarray(z_var)

#결과 저장
lon_mesh, lat_mesh = np.meshgrid(grid_lon, grid_lat)

grid_df = pd.DataFrame({
    "lat": lat_mesh.ravel(),
    "lon": lon_mesh.ravel(),
    "pgv_kriged_pred": z_pred.ravel(),
    "kriging_var": z_var.ravel()
})

grid_df.to_csv(OUTPUT_GRID_CSV, index=False)
print(f"크리깅 결과 저장 완료: {OUTPUT_GRID_CSV}")