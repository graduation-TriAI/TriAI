import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# 0. 설정

# 경로 설정
PRED_CSV_PATH   = "tohoku_predicted_pgv.csv"               # seismic_station, pgv_pred
META_CSV_PATH   = "tohoku_station_pairs.csv"               # seismic_station, seismic_lat, seismic_lon
VS30_SHP_PATH   = "Z-V4-JAPAN-AMP-VS400_M250-SHAPE.shp"   # J-SHIS 셰이프파일 경로
OUTPUT_CSV_PATH = "tohoku_pgv_vs30_corrected.csv"          # 최종 출력

# 보정 방식 설정
# "arv" : J-SHIS ARV 값 직접 사용
# "avs" : AVS에서 수식으로 증폭계수 계산(ARV 없는 경우)
CORRECTION_MODE = "arv"

# AVS 수식 사용 시 기준 Vs30 (USGS ShakeMap 표준 암반 기준)
VS30_REFERENCE  = 760.0

# AVS=0(정보 없음) 관측소에 적용할 기본값(중앙값)
FALLBACK_MODE   = "median"


# 1. 예측 PGV + 관측소 메타데이터 로드 및 병합
def load_station_data(pred_csv, meta_csv):
    pred_df = pd.read_csv(pred_csv)
    pred_df["seismic_station"] = pred_df["seismic_station"].astype(str).str.strip().str.upper()
    pred_df["pgv_pred"] = pd.to_numeric(pred_df["pgv_pred"], errors="coerce")
    pred_df = pred_df.dropna(subset=["pgv_pred"])

    meta_df = pd.read_csv(meta_csv)
    meta_df["seismic_station"] = meta_df["seismic_station"].astype(str).str.strip().str.upper()
    meta_df = meta_df[["seismic_station", "seismic_lat", "seismic_lon"]].drop_duplicates("seismic_station")

    merged = pd.merge(pred_df, meta_df, on="seismic_station", how="left")
    merged = merged.dropna(subset=["seismic_lat", "seismic_lon"]).copy()
    print(f"[로드] 유효 관측소: {len(merged)}개")
    return merged


# 2. 각 관측소 좌표에 대해 Vs30 폴리곤에서 AVS/ARV 추출
def extract_vs30_for_stations(station_df, shp_path):
    lats = station_df["seismic_lat"].values
    lons = station_df["seismic_lon"].values

    # bbox로 shp 로드 범위 제한 (전체 620만 폴리곤 로드 방지)
    margin = 0.5
    bbox = (lons.min() - margin, lats.min() - margin,
            lons.max() + margin, lats.max() + margin)
    print(f"[Vs30] bbox 범위로 셰이프파일 로드 중: {bbox}")

    vs30_gdf = gpd.read_file(shp_path, bbox=bbox)
    vs30_gdf = vs30_gdf.to_crs("EPSG:4326")
    print(f"[Vs30] 로드된 폴리곤 수: {len(vs30_gdf)}")

    # 관측소 GeoDataFrame 생성
    points_gdf = gpd.GeoDataFrame(
        station_df.copy(),
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326"
    )

    # 공간 조인: 각 관측소 포인트가 포함된 폴리곤의 AVS/ARV 가져오기
    joined = gpd.sjoin(
        points_gdf,
        vs30_gdf[["AVS", "ARV", "geometry"]],
        how="left",
        predicate="within"
    )

    # 중복 제거 (폴리곤 경계선에 걸리는 경우)
    joined = joined.drop_duplicates(subset=["seismic_station"])

    result = station_df.copy()
    result["vs30_avs"] = joined["AVS"].values
    result["vs30_arv"] = joined["ARV"].values

    n_missing = result["vs30_avs"].isna().sum() + (result["vs30_avs"] == 0).sum()
    print(f"[Vs30] 매칭 실패 또는 AVS=0 관측소: {n_missing}개")
    return result


# 3. 증폭계수 계산
def compute_amp_factor(row, mode, vs30_ref):
    """
    mode="arv": J-SHIS ARV 직접 사용
        ARV = "기준 암반 대비 이 지점의 PGV 증폭 배율"
        → pgv_corrected = pgv_pred × ARV

    mode="avs": Campbell & Bozorgnia (2008) 간소화 식
        ln(Amp) = -0.84 × ln(Vs30 / Vs30_ref)
        → Vs30가 낮을수록(연약지반) 증폭계수 커짐
    """
    if mode == "arv":
        arv = row["vs30_arv"]
        if pd.isna(arv) or arv <= 0:
            return np.nan
        return float(arv)
    else:  # "avs"
        avs = row["vs30_avs"]
        if pd.isna(avs) or avs <= 0:
            return np.nan
        avs = np.clip(avs, 100.0, 2000.0)
        return float(np.exp(-0.84 * np.log(avs / vs30_ref)))


# 4. Fallback 처리: Vs30 정보가 없는 관측소
def apply_fallback(df, fallback_mode):
    valid_factors = df.loc[df["amp_factor"].notna(), "amp_factor"]

    if len(valid_factors) == 0:
        print("유효한 증폭계수가 없음. 전체를 1.0으로 설정")
        df["amp_factor"] = df["amp_factor"].fillna(1.0)
        return df

    if fallback_mode == "median":
        fallback_val = float(valid_factors.median())
        print(f"[Fallback] Vs30 없는 관측소 → 중앙값 증폭계수 {fallback_val:.4f} 적용")
    else:
        fallback_val = float(fallback_mode)
        print(f"[Fallback] Vs30 없는 관측소 → 고정값 {fallback_val:.4f} 적용")

    df["amp_factor"] = df["amp_factor"].fillna(fallback_val)
    return df


# 실행
def main():
    # 데이터 로드
    df = load_station_data(PRED_CSV_PATH, META_CSV_PATH)

    # Vs30 폴리곤에서 AVS/ARV 추출
    df = extract_vs30_for_stations(df, VS30_SHP_PATH)

    # 증폭계수 계산
    df["amp_factor"] = df.apply(
        lambda row: compute_amp_factor(row, CORRECTION_MODE, VS30_REFERENCE),
        axis=1
    )

    # Fallback 처리
    df = apply_fallback(df, FALLBACK_MODE)

    # 보정된 PGV 계산 및 음수 방지
    df["pgv_corrected"] = (df["pgv_pred"] * df["amp_factor"]).clip(lower=0)

    # 저장
    out_cols = ["seismic_station", "seismic_lat", "seismic_lon",
                "pgv_pred", "vs30_avs", "vs30_arv", "amp_factor", "pgv_corrected"]
    df[out_cols].to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\n저장: {OUTPUT_CSV_PATH}")
    print(df[["pgv_pred", "amp_factor", "pgv_corrected"]].describe())
    print("\n샘플:")
    print(df[out_cols].head())
    return df


if __name__ == "__main__":
    main()
