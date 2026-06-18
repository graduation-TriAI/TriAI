# 실험

## 실험 개요

### 수행 과제

- 대지진 4종류에 대해 **K-Fold Cross Validation**을 활용하여 **PGV 추정**
	- [실험 1] 학습: 대지진 1, 대지진 2, 대지진 3 / 검증 및 테스트: 대지진 4
	- [실험 2] 학습: 대지진 1, 대지진 2, 대지진 4 / 검증 및 테스트: 대지진 3
	- [실험 3] 학습: 대지진 1, 대지진 3, 대지진 4 / 검증 및 테스트: 대지진 2
	- [실험 4] 학습: 대지진 2, 대지진 3, 대지진 4 / 검증 및 테스트: 대지진 1

- 목표
	- 각 실험에서 테스트 데이터의 **PGV 예측 성능을 평가**
    - 예측 PGV 값을 기반으로 한 **ShakeMap 형태의 위험지도 생성**
	- 단일 모달리티 기반 모델과 제안 멀티모달 모델의 성능을 비교하여 **멀티모달 접근 방식의 효과 검증**

### 데이터셋

- 대지진 1
	- 2011.03.11. 도호쿠 대지진
	- 구성
		- 지진파형 데이터: Hi-net
		- GNSS 변위 데이터: PANGAEA

- 대지진 2
	- 2016.04.15. 쿠마모토 대지진
	- 구성
		- 지진파형 데이터: Hi-net
		- GNSS 변위 데이터: GSI

- 대지진 3
	- 2018.09.05. 홋카이도 대지진
	- 구성
		- 지진파형 데이터: Hi-net
		- GNSS 변위 데이터: GSI

- 대지진 4
  	- 2024.01.01. 노토 대지진
  	- 구성
  	  	- 지진파형 데이터: Hi-net
  	  	- GNSS 변위 데이터: GSI

### 로컬 학습 인코더

- 지진파형 인코더: **EQTransformer** 구조 차용
- GNSS 인코더: **EQTransformer** 구조 변형

### 실험 환경

- Google Colab GPU

## 디렉토리 구조

```markdown
.
├── preprocessing/
│   ├── README.md
│   ├── seismic_pipeline/
│   │   ├── README.md
│   │   ├── hinet.py
│   │   ├── hinet_slicing.py
│   │   └── 지진파형_구조_변형.ipynb
│   ├── gnss_pipeline/
│   │   ├── README.md
│   │   ├── hokkaido_pipeline/
│   │   │   ├── extract_station_latlon.py
│   │   │   ├── gnss_ecef_to_enu.py
│   │   │   └── slice_gnss_station_sequenece.py
│   │   ├── kumamoto_pipeline/
│   │   │   ├── extract_station_latlon.py
│   │   │   ├── gnss_ecef_to_enu.py
│   │   │   └── slice_gnss_station_sequenece.py
│   │   ├── noto_pipeline/
│   │   │   ├── compute_distance_to_noto.py
│   │   │   ├── extract_station_latlon_for_pairing.py
│   │   │   ├── extract_station_latlon.py
│   │   │   ├── filter_stations_by_epicenter_distance.py
│   │   │   ├── gnss_ecef_to_enu.py
│   │   │   ├── slice_gnss_station_sequenece_upsample100_from_pairs.py
│   │   │   └── slice_gnss_station_sequenece.py
│   │   └── tohoku_pipeline/
│   │       ├── __init__.py
│   │       ├── compute_distance_to_tohoku.py
│   │       ├── extract_station_latlon_for_pairing.py
│   │       ├── extract_station_latlon.py
│   │       ├── extract_tohoku_region_stations.py
│   │       ├── slice_gnss_station_sequenece_upsample100_from_pairs.py
│   │       └── slice_gnss_station_sequenece.py
│   ├── multimodal_pipeline/
│   │   ├── README.md
│   │   └── 멀티모달 데이터셋 정렬.ipynb
│   ├── pgv_pipeline/
│   │   ├── README.md
│   │   ├── compute_distance_to_tohoku.py
│   │   ├── extract_station_latlon.py
│   │   ├── PGV 매칭.py
│   │   └── PGV 정리.py
│   ├── station_pairs.ipynb
│   ├── upsampling.py
│   └── 통합 데이터셋 구축.ipynb
├── work/
│   ├── README.md
│   ├── gnss/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── baseline_ver2.py
│   │   ├── baseline_weighted_mse.py
│   │   ├── baseline.py
│   │   ├── build_gnss_pgv_dataset_seq.py
│   │   ├── encoder_ver2.py
│   │   ├── encoder.py
│   │   ├── export_metric_npy.py
│   │   ├── final_baseline.py
│   │   ├── inspect_npz.py
│   │   ├── log_export_ver.py
│   │   ├── model_ver2.py
│   │   ├── model.py
│   │   ├── plot_logs.py
│   │   └── plot_test_scatter.py
│   ├── seismic/
│   │   ├── README.md
│   │   ├── seismic_model_ver1.ipynb
│   │   └── seismic_model_ver2.ipynb
│   └── multimodal/
│   	├── README.md
│       └── multimodal model.ipynb
├── postprocessing/
│   ├── README.md
│   ├── kriging/
│   │   ├── make_kriging_input.py
│   │   └── run_kriging.py
│   └── shakemap.ipynb
├── final/
│   ├── README.md
│   ├── seismic.ipynb
│   ├── gnss/
│   │   ├── build_gnss_pgv_dataset_seq.py
│   │   ├── encoder_ver2.py
│   │   ├── final_baseline.py
│   │   ├── model_ver2.py
│   │   ├── plot_logs.py
│   │   └── plot_test_scatter.py
│   ├── multimodal.ipynb
│   └── postprocessing.ipynb
├── docs/
│   ├── 26-TriAI-1차보고서-조혜림.pdf
│   ├── 26-TriAI-2차보고서-조혜림.pdf
│   └── 26-TriAI-포스터 파일-조혜림.pdf
├── shared/
└── .gitignore 
```

## 실험 방법


## 실험 결과

|  | Seismic-only | GNSS-only | Proposed |
| --- | --- | --- | --- |
| 노토 테스트 실험 Test RMSE | 10.96 | 21.38 | 9.92 |
| 홋카이도 테스트 실험 Test RMSE | 8.25 | 8.93 | 7.80 |
| 쿠마모토 테스트 실험 Test RMSE | 4.52 | 14.05 | 3.47 |
| 도호쿠 테스트 실험 Test RMSE | 12.27 | 8.76 | 14.05 |