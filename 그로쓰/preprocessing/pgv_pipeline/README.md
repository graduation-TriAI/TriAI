### pgv_pipeline

이 디렉토리는 PGV(Peak Ground Velocity) 라벨을 관측소별로 매칭하여 학습용 데이터로 생성하는 전처리 파이프라인의 세부 구현을 포함합니다.

---

#### 주요 단계

1. PGV 데이터 정리
	- JSON 형식의 PGV 데이터를 CSV 형태로 변환 (`PGV 정리.py`) 
	
2. 관측소 메타데이터 준비
	- 관측소명, 위도, 경도 정보 추출 (`extract_station_latlon.py`)
	- 진앙으로부터 거리 계산 (`compute_distance_to_tohoku.py`)

3. PGV 매칭
	- 관측소명 및 위치 정보를 기반으로 PGV 값을 Hi-net 관측소에 매칭 (`PGV 매칭.py`)