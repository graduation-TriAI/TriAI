### tohoku_gnss_pipeline

이 디렉토리는 도호쿠 지진 데이터에 대한 GNSS 전처리 파이프라인의 세부 구현을 포함합니다.

---

#### 주요 단계

1. 관측소 메타데이터 추출
	- 관측소명/위도/경도 정보 추출 (`extract_station_lat_lon.py`)
	
2. 도호쿠 지역 관측소 필터링
	- 관심 지역 내 관측소 선택 (`extract_tohoku_region_stations.py`)

3. pairing 준비
	- 관측소 매칭을 위한 메타데이터 생성 (`extract_station_latlon_for_pairing.py`)

4. 시계열 데이터 슬라이싱
	- 윈도우로 분할 전 Z-score 정규화 수행
	- 3채널(E/N/U) GNSS를 일정 길이의 윈도우로 분할 (`slice_*.py`)

5. 학습용 데이터셋 생성
	- 슬라이싱된 시계열 데이터를 하나로 결합하여 `.npz` 형태로 저장
	- 관측소 이름 및 윈도우 시작 시각 정보 포함
	
---
	
#### (Optional) 거리 기반 필터링

- 진앙 기준 거리 계산 (`compute_distance_to_tohoku.py`)