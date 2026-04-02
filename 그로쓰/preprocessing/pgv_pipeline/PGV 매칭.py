import pandas as pd

# 1. 데이터 불러오기
df_pgv = pd.read_csv('pgv_data.csv')
df_hinet = pd.read_csv('seismic_stations_sorted_by_distance.csv')


# 2. Hi-net 이름을 KiK-net 이름(타겟명)으로 변환할 딕셔너리 만들기
hinet_to_kiknet = {}
with open('hi-net_data/01_01_20110311.sjis.ch', 'r', encoding='shift_jis', errors='ignore') as f:
    for line in f:
        line = line.strip()
        # 주석 줄에서 관측소 짝꿍 코드 추출 (예: # n.szgh Shizugawa mygh12)
        if line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                hinet_name = parts[1].upper()       # N.SZGH
                kiknet_code = parts[-1].upper()     # MYGH12
                # 매칭을 위해 PGV 데이터 포맷인 KIK.MYGH12 형태로 조립
                hinet_to_kiknet[hinet_name] = f"KIK.{kiknet_code}"

# Hi-net 데이터에 'target_station'이라는 비교용 컬럼 추가
df_hinet['target_station'] = df_hinet['station'].map(hinet_to_kiknet)


# 3. 위도, 경도를 소수점 4자리(약 10m 오차)로 통일
df_hinet['lat_rnd'] = df_hinet['latitude'].round(4)
df_hinet['lon_rnd'] = df_hinet['longitude'].round(4)

df_pgv['lat_rnd'] = df_pgv['latitude'].round(4)
df_pgv['lon_rnd'] = df_pgv['longitude'].round(4)


# 4. 관측소명(타겟명), 위도, 경도 3가지 조건으로 정확히 매칭
def match_exact_pgv(row):
    # 타겟 이름이 없는 경우(변환 실패 시) 건너뜀
    if pd.isna(row['target_station']):
        return None
        
    # 변환된 관측소 이름, 위도, 경도 일치 확인
    match = df_pgv[
        (df_pgv['station'] == row['target_station']) & 
        (df_pgv['lat_rnd'] == row['lat_rnd']) & 
        (df_pgv['lon_rnd'] == row['lon_rnd'])
    ]
    
    # 모두 일치하는 데이터가 있으면 해당 pgv 값 반환
    if not match.empty:
        return match.iloc[0]['pgv']
    
    # 일치하는 데이터가 없으면 빈칸 유지
    return None

# 각 행마다 매칭 함수 적용
df_hinet['pgv'] = df_hinet.apply(match_exact_pgv, axis=1)

# 검사용으로 임시 생성했던 컬럼들 삭제
df_hinet = df_hinet.drop(columns=['target_station', 'lat_rnd', 'lon_rnd'])


# 5. 최종 결과 저장 및 출력

# pgv 값 존재하는 데이터만 저장하는 경우
df_hinet = df_hinet.dropna(subset=['pgv'])

output_filename = 'processed_pgv_data2.csv'
df_hinet.to_csv(output_filename, index=False)

total_count = len(df_hinet)
matched_count = df_hinet['pgv'].notna().sum()

print(f"pgv 매칭 작업 완료")
print(f"총 {total_count}개의 Hi-net 관측소 중, [관측소명, 위도, 경도]가 정확히 일치하는 {matched_count}개의 PGV 발견")