import json
import pandas as pd

# 데이터 불러오기
with open("도호쿠stationlist.json", "r", encoding="utf-8") as f:
    pgv_data = json.load(f)

# 관측소명, 위도, 경도, pgv값 만 추출
extracted_data = []
for feature in pgv_data['features']:
    station = feature['id']
    pgv = feature['properties'].get('pgv', None)

    if 'geometry' in feature and feature['geometry'] is not None:
        lon, lat = feature['geometry']['coordinates']
    else:
        lon, lat = None, None
        
    extracted_data.append({
        'station': station,
        'latitude': lat,
        'longitude': lon,
        'pgv': pgv
    })

# DataFrame 변환, CSV 저장
df_pgv = pd.DataFrame(extracted_data)
df_pgv.to_csv('pgv_data.csv', index=False)

print("pgv 데이터 정리 완료")