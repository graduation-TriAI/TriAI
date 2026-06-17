## preprocessing

본 디렉토리는 지진파형 및 GNSS 데이터를 모델 학습에 사용할 수 있도록 전처리하는 파이프라인을 포함하고 있습니다.

---

### 개요

원시(raw) 지진파형 데이터와 GNSS 데이터를 입력으로 받아 필터링, 정규화, 슬라이딩 윈도우 분할 등의 과정을 거쳐 모델 입력에 적합한 형태의 데이터셋으로 변환합니다. 각 파이프라인 디렉토리에는 데이터셋 생성 과정이 단계별로 구현되어 있으며, 최종적으로 모델 학습에 사용할 수 있는 형태로 데이터를 변환하는 것을 목표로 합니다.

---

### 전처리 흐름 

1. 지진파형 / GNSS 데이터 수집 
	- Hi-net 및 GNSS 관측소 데이터 다운로드
	
2. 관측소 메타데이터 및 관측소 매칭
	- 거리 기반으로 지진파형-GNSS 관측소 매칭

3. 시계열 데이터 전처리
	- 정규화 및 시간 정렬

4. 슬라이딩 윈도우 분할
	- 일정 길이의 시계열 샘플 생성

5. PGV 라벨 매칭 및 정체
	- 관측소별 PGV 값 생성 및 필터링

6. 모델 입력용 데이터셋 생성
	- `.npz` 형태로 최종 학습 데이터 구성

7. 

---

### 디렉토리 구조

```markdown
.
├── README.md
├── seismic_pipeline/
│   ├── README.md
│   ├── hinet_slicing.py
│   └── hinet.py
├── gnss_pipeline/
│   ├── README.md
│   ├── hokkaido_pipeline/
│   │   ├── extract_station_latlon.py
│   │   ├── gnss_ecef_to_enu.py
│   │   └── slice_gnss_station_sequenece.py
│   ├── kumamoto_pipeline/
│   │   ├── extract_station_latlon.py
│   │   ├── gnss_ecef_to_enu.py
│   │   └── slice_gnss_station_sequenece.py
│   ├── noto_pipeline/
│   │   ├── compute_distance_to_noto.py
│   │   ├── extract_station_latlon_for_pairing.py
│   │   ├── extract_station_latlon.py
│   │   ├── filter_stations_by_epicenter_distance.py
│   │   ├── gnss_ecef_to_enu.py
│   │   ├── slice_gnss_station_sequenece_upsample100_from_pairs.py
│   │   └── slice_gnss_station_sequenece.py
│   └── tohoku_pipeline/
│   │   ├── __init__.py
│   │   ├── compute_distance_to_tohoku.py
│   │   ├── extract_station_latlon_for_pairing.py
│   │   ├── extract_station_latlon.py
│   │   ├── extract_tohoku_region_stations.py
│   │   ├── slice_gnss_station_sequenece_upsample100_from_pairs.py
│   │   └── slice_gnss_station_sequenece.py
├── multimodal_pipeline/
│   ├── README.md
│   └── 멀티모달 데이터셋 정렬.ipynb
├── pgv_pipeline/
│   ├── README.md
│   ├── compute_distance_to_tohoku.py
│   ├── extract_station_latlon.py
│   ├── PGV 매칭.py
│   └── PGV 정리.py
├── station_pairs.ipynb
├── upsampling.py
└── 통합 데이터셋 구축.ipynb

```