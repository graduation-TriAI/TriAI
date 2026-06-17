## gnss_pipeline

본 디렉토리는 모델 학습 후 예측 PGV 값을 후처리하는 코드로 구성된 디렉토리입니다.

---

### 전처리 흐름

- 도호쿠 지진
  1. 관측소 메타데이터 추출
    - 관측소명/위도/경도 정보 추출 (`extract_station_latlon.py`)
    
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
  
  * (Optional) 거리 기반 필터링
    - 진앙 기준 거리 계산 (`compute_distance_to_tohoku.py`)

- 쿠마모토 & 홋카이도 지진
  1. 좌표 변환 및 초기 전처리
    - ECEF 좌표를 ENU 좌표로 변환 (`gnss_ecef_to_enu.py`)
    
  2. 관측소 메타데이터 추출
    - 관측소명/위도/경도 정보 추출 (`extract_station_latlon.py`)

  3. 시계열 데이터 슬라이싱
    - 윈도우로 분할 전 Z-score 정규화 수행
    - 3채널(E/N/U) GNSS를 일정 길이의 윈도우로 분할 (`slice_*.py`)

  4. 학습용 데이터셋 생성
    - 슬라이싱된 시계열 데이터를 하나로 결합하여 `.npz` 형태로 저장
    - 관측소 이름 및 윈도우 시작 시각 정보 포함

- 노토 지진
  1. 좌표 변환 및 초기 전처리
    - ECEF 좌표를 ENU 좌표로 변환 (`gnss_ecef_to_enu.py`)
    
  2. 관측소 메타데이터 추출
    - 관측소명/위도/경도 정보 추출 (`extract_station_latlon.py`)

  3. 진앙 기준 거리 계산 및 관측소 필터링
    - 진앙으로부터 거리 계산 (`compute_distance_to_noto.py`)
    - 반경 내 관측소 선택 (`filter_stations_by_epicenter_distance.py`)

  4. pairing 준비
    - 관측소 매칭을 위한 메타데이터 생성 (`extract_station_latlon_for_pairing.py`)

  5. 시계열 데이터 슬라이싱
    - 윈도우로 분할 전 Z-score 정규화 수행
    - 3채널(E/N/U) GNSS를 일정 길이의 윈도우로 분할 (`slice_*.py`)

  6. 학습용 데이터셋 생성
    - 슬라이싱된 시계열 데이터를 하나로 결합하여 `.npz` 형태로 저장
    - 관측소 이름 및 윈도우 시작 시각 정보 포함

---

### 디렉토리 구조

```markdown
.
├── README.md
├── hokkaido_pipeline/
│   ├── extract_station_latlon.py
│   ├── gnss_ecef_to_enu.py
│   └── slice_gnss_station_sequenece.py
├── kumamoto_pipeline/
│   ├── extract_station_latlon.py
│   ├── gnss_ecef_to_enu.py
│   └── slice_gnss_station_sequenece.py
├── noto_pipeline/
│   ├── compute_distance_to_noto.py
│   ├── extract_station_latlon_for_pairing.py
│   ├── extract_station_latlon.py
│   ├── filter_stations_by_epicenter_distance.py
│   ├── gnss_ecef_to_enu.py
│   ├── slice_gnss_station_sequenece_upsample100_from_pairs.py
│   └── slice_gnss_station_sequenece.py
└── tohoku_pipeline/
    ├── __init__.py
    ├── compute_distance_to_tohoku.py
    ├── extract_station_latlon_for_pairing.py
    ├── extract_station_latlon.py
    ├── extract_tohoku_region_stations.py
    ├── slice_gnss_station_sequenece_upsample100_from_pairs.py
    └── slice_gnss_station_sequenece.py

```