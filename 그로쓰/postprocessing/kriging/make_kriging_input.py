import pandas as pd
from shared.paths import PAIRS_NOTO_CSV, POST

#seismic_station, pgv_pred 가 저장된 csv 파일
PRED_CSV_PATH = POST / "noto_predicted_pgv_25km_360_180.csv"

META_CSV_PATH = PAIRS_NOTO_CSV / "noto_station_pairs.csv"

OUTPUT_CSV_PATH = POST / "noto_pred_for_kriging_25km_360_180.csv"

AGG_FUNC = "mean"

# PRED_CSV_PATH 불러오기
pred_df = pd.read_csv(PRED_CSV_PATH)
pred_df = pred_df[["seismic_station", "pgv_pred"]].copy()

pred_df["seismic_station"] = (
    pred_df["seismic_station"]
    .astype(str)
    .str.strip()
    .str.upper()
)

pred_df["pgv_pred"] = pd.to_numeric(pred_df["pgv_pred"], errors="coerce")
pred_df = pred_df.dropna(subset=["seismic_station", "pgv_pred"])

print("예측 CSV 행 수:", len(pred_df))

# META_CSV_PATH 불러오기
meta_df = pd.read_csv(META_CSV_PATH)
meta_df = meta_df[["seismic_station", "seismic_lat", "seismic_lon"]].copy()

meta_df["seismic_station"] = (
    meta_df["seismic_station"]
    .astype(str)
    .str.strip()
    .str.upper()
)

meta_df["seismic_lat"] = pd.to_numeric(meta_df["seismic_lat"], errors="coerce")
meta_df["seismic_lon"] = pd.to_numeric(meta_df["seismic_lon"], errors="coerce")
meta_df = meta_df.dropna(subset=["seismic_station", "seismic_lat", "seismic_lon"])

print("메타 CSV 행 수:", len(meta_df))

# META_CSV_PATH 중복 제거
meta_unique = (
    meta_df.groupby("seismic_station", as_index=False)
    .agg({
        "seismic_lat": "first",
        "seismic_lon": "first"
    })
)

print("중복 제거 후 메타 관측소 수:", len(meta_unique))

# PRED_CSV_PATH 중복 처리
dup_count = pred_df["seismic_station"].duplicated().sum()
print("예측 CSV 중복 station 수:", dup_count)

if AGG_FUNC == "mean":
    pred_unique = pred_df.groupby("seismic_station", as_index=False).agg({"pgv_pred": "mean"})
elif AGG_FUNC == "first":
    pred_unique = pred_df.groupby("seismic_station", as_index=False).agg({"pgv_pred": "first"})
else:
    raise ValueError("AGG_FUNC must be one of: mean, first")

print("중복 처리 후 예측 관측소 수:", len(pred_unique))

# merge
merged_df = pd.merge(
    pred_unique,
    meta_unique,
    on="seismic_station",
    how="left"
)

print("merge 후 행 수:", len(merged_df))

# 좌표 매칭 실패 확인
missing_coord_df = merged_df[
    merged_df["seismic_lat"].isna() | merged_df["seismic_lon"].isna()
]

print("좌표 매칭 실패 관측소 수:", len(missing_coord_df))

if len(missing_coord_df) > 0:
    print("\n좌표 매칭 실패 예시:")
    print(missing_coord_df.head())

# 좌표 없는 행 제거
merged_df = merged_df.dropna(subset=["seismic_lat", "seismic_lon"]).copy()

# 컬럼 순서 정리
merged_df = merged_df[
    ["seismic_station", "seismic_lat", "seismic_lon", "pgv_pred"]
]

# 저장
merged_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n크리깅 입력용 CSV 저장 완료: {OUTPUT_CSV_PATH}")
print(merged_df.head())