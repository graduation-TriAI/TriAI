# TriAI

이화여자대학교 컴퓨터공학과 졸업프로젝트 연구 트랙 26팀

---

# 연구 주제

**멀티모달 딥러닝** 기반 지진 **PGV 추정 및 위험지도 시각화**
<br>Multimodal Deep Learning-Based PGV Estimation and Earthquake Hazard Mapping

## 연구 소개

이 연구는 서로 다른 모달리티의 **지진파형과 GNSS 변위 데이터**를 Universal Modality-Independent Space에 투영하고, **EQTransformer 기반 멀티모달 딥러닝 모델**을 활용하여 **PGV 추정 및 지진 위험지도 시각화**를 수행함으로써 지진 발생 시 예상 피해 지역을 신속하게 파악하는 것을 목표로 함.

## 문제상황

- 지진 관측 데이터는 지진파형 데이터(파형), 지표 변위 데이터(위치), 위성 영상 데이터(이미지), 전자기파 변화 데이터(센서), 지하수 및 가스 변화 데이터(화학) 등 **다양한 모달리티**를 가지는 여러 데이터로 구성됨
- 기존의 지진 분석 방식은 **확률 모델이나 물리 기반 지반운동식(GMPE)** 등을 이용해 PGV를 추정하나, 이러한 방법은 **복잡한 지진파 패턴과 다양한 지반 조건**을 충분히 반영하기 어려움
- 최근에는 **지진파형 데이터를 활용한 딥러닝 연구**가 수행되고 있으나, 단일 모달리티 데이터만을 사용해 지진 발생 시 나타나는 **복합적인 지반 변형 특성**을 충분히 반영하기 어려움

## 기대효과

- **PGV 추정 정확도 향상**: 지진파형 데이터와 GNSS 변위 데이터를 함께 활용한 멀티모달 딥러닝 모델을 통해 지진의 복잡한 시공간적 패턴을 학습해, 정확한 PGV 값을 추정할 것임
- **멀티모달 데이터의 통합 분석**: 서로 다른 형태의 지진 관측 데이터를 Universal Modality-Independent Space(UMIS) 기반으로 통합하여, 다양한 모달리티 정보를 동시에 활용하는 통합적 지진 데이터 분석이 가능함
- **지진 위험지도 생성 및 피해 지역 파악**: 공간 보간 기법을 활용해 미계측 지역까지 PGV를 추정할 수 있으며, 격자 기반 지진 위험지도를 생성해 지진 예상 피해 지역을 직관적으로 파악할 수 있음

---

# 실험

## 실험 개요

### 수행 과제

- 대지진 2종류에 대해 **K-Fold Cross Validation**을 활용하여 **PGV 추정**
	- [모델 1] 학습: 대지진 1 / 검증/테스트: 대지진 2
	- [모델 2] 학습: 대지진 2 / 검증/테스트: 대지진 1

- 평가 방식
	- 각 모델에서 검증 데이터의 **PGV 예측 성능을 평가**
	- 2개 Fold의 결과를 종합해 **모델의 일반화 성능**을 분석

### 데이터셋

- 대지진 1
	- 2011.03.11. 도호쿠 대지진
	- 구성
		- 지진파형 데이터: Hi-net
		- GNSS 변위 데이터: PANGAEA

- 대지진 2
  	- 2024.01.01. 노토 대지진
  	- 구성
  	  	- 지진파형 데이터: Hi-net
  	  	- GNSS 변위 데이터: PANGAEA, 

### 로컬 학습 인코더

- 지진파형 인코더: **EQTransformer** 구조 차용
- GNSS 인코더: **EQTransformer** 구조 변형

### 실험 환경

- Google Colab GPU

## 폴더 구조

```markdown
.
├── README.md
├── GroundRule.md
├── Ideation.md
├── Project-Scenario.md
├── 스타트/
│   ├── 데이터/
│   │   ├── gl.csv
│   │   ├── gl_df.csv
│   │   ├── label.csv
│   │   ├── gw_scaler.pkl
│   │   ├── gru_dataset.npz
│   │   ├── merged_seismic_data.h5
│   │   ├── EQT_Training_Dataset.h5
│   │   └── EQT_Multitask_Dataset.h5
│   ├── 데이터 전처리/
│   │   ├── 지진 목록 데이터.xls
│   │   ├── 지하수 데이터 수집.ipynb
│   │   ├── 지진파 데이터 크롤링.ipynb
│   │   └── 데이터 전처리.ipynb
│   └── 실험/
│       ├── 로컬학습/
│       │   ├── 1D_CNN.ipynb
│       │   ├── EQTransformer.ipynb
│       │   └── smallGRU.ipynb
│       └── 연합학습/
│           ├── FL_freeze.ipynb
│           └── FL_freeze_unfreeze.ipynb
└── 그로쓰/ 
		├── preprocessing/
		│   ├── tohoku_gnss_pipeline/
		│   ├── tohoku_seismic_pipeline/
		│   ├── tohoku_pgv/
		│   ├── tohoku_seismic/
		│   ├── station_pairs.ipynb
		│   ├── upsampling.py
		│   └── 카탈로그_데이터_필터링.ipynb
		├── shared/
		├── work/
		│   ├── gnss/
		│   │   └── encoder.py
		│   └── seismic/
		│   │   └── seismic_model.ipynb
        ├── .gitignore 
		└── 26-TriAI-1차보고서-조혜림.pdf
            
```

---

# 팀 소개

| 이름 | 조혜림 | 김민 | 박소영 |
| --- | --- | --- | --- |
| 학번 | 2371062 | 2371012 | 2371031 |
| 역할 | 팀장, 서류 메인 관리, 디코더 설계, 데이터 수집 및 전처리, 레이블 및 기타 데이터 처리, 멀티모달 모델 학습 | 팀원, 깃허브 메인 관리, 데이터 수집 및 전처리, 지진파형 모델 설계 및 학습, 검증 | 팀원, 노션 메인 관리, 데이터 수집 및 전처리, GNSS 모델 설계 및 학습, 검증 |

# 팀 그라운드 룰

[GroundRule.md](https://github.com/graduation-TriAI/TriAI/blob/main/GroundRule.md)
