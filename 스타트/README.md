# 실험

## 실험 개요

### 수행 과제

- 지진 **발생 여부** 판별 **(이진 분류)**
    - 목표: **지진(1) vs 노이즈(0) 판별**
    - 손실 함수: Binary Cross-Entropy
- 지진 **규모** 예측 **(회귀)**
    - 목표: **리히터 규모 실수값 예측**
    - 손실 함수: MSE (Mean Squared Error)

### 데이터셋

- 범위: **2017년 포항 지진** 및 전후 데이터
- 구성
    - **지진 목록 데이터:** StandardScaler 스케일링, Sliding Window 적용하여 전처리
    - **지하수 수위 데이터**: 이상치 제거 및 선형 보간, 일 단위 평균 및 정규를 통해 전처리
    - **지진파 데이터**: 지진 이벤트(P/S파)+노이즈 데이터 1:1 비율로 생성하여 전처리
- Train set : Test set = 80 : 20

### 로컬 학습 인코더

- 지진 목록 데이터: **1D-CNN**
- 지하수 수위 데이터: **small GRU**
- 지진파 데이터: **EQTransformer**

### UMIS 통합 아키텍처

1. **Local Adapter**: 서로 다른 로컬 학습 모델의 특징 벡터를 중간 공유 차원(=64)으로 확장 및 통일
2. **Batch Normalization**: 특징 벡터 분포 정규화를 통한 학습 안정성 확보
3. **UMIS Projection**: 정규화된 특징 벡터를 공통 임베딩 공간(=32)으로 투영
4. **Prediction Head**: 최종 예측(이진 분류/회귀) 수행

### 연합학습 알고리즘

- **FedAvg (Federaged Averaging):** 각 클라이언트의 데이터 샘플 수에 비례하여 가중치 평균 계산
- **손실 함수**
    - 이진 분류: **Binary Cross-Entropy**
    - 회귀: **MSE (Mean Squared Error)**
- **평가 지표:**
    - 이진 분류: **Accuracy**
    - 회귀: **MAE (Mean Absolute Error)**

### **2단계 학습 전략**

- **Stage 1: 인코더 Freeze (Round 1~10)**
    - 방법: 모든 로컬 인코더 가중치 고정(Freeze), UMIS Projection Layer만 학습
    - 목적: 사전 학습된 인코더가 추출한 특징 표현이 연합학습 환경에서도 안정적으로 활용 가능한지 검증
- **Stage 2: 인코더  Unfreeze (Round 11~15)**
    - 방법: 인코더 고정 해제(Unfreeze) 후 추가 학습
    - 목적: 정렬된 공통 임베딩 공간을 바탕으로, 글로벌 모델의 가중치를 로컬 인코더까지 역전파(Backpropagation)하여 미세 조정 및 최종 성능 개선

### 실험 환경

- Google Colab GPU

## 폴더 구조

```markdown
.
├── README.md
├── GroundRule.md
├── Ideation.md
├── Project-Scenario.md
└── 스타트
    ├── 데이터
    │   ├── gl.csv
    │   ├── gl_df.csv
    │   ├── label.csv
    │   ├── gw_scaler.pkl
    │   ├── gru_dataset.npz
    │   ├── merged_seismic_data.h5
    │   ├── EQT_Training_Dataset.h5
    │   └── EQT_Multitask_Dataset.h5
    ├── 데이터 전처리
    │   ├── 지진 목록 데이터.xls
    │   ├── 지하수 데이터 수집.ipynb
    │   ├── 지진파 데이터 크롤링.ipynb
    │   └── 데이터 전처리.ipynb
    └── 실험
        ├── 로컬학습
        │   ├── 1D_CNN.ipynb
        │   ├── EQTransformer.ipynb
        │   └── smallGRU.ipynb
        └── 연합학습
            ├── FL_freeze.ipynb
            └── FL_freeze_unfreeze.ipynb
```

## 실험 방법

### 데이터 수집 및 전처리

1. 스타트/데이터 전처리 폴더의 파일들을 다운 받는다.
2. 지하수 데이터 수집.ipynb, 지진파 데이터 크롤링.ipynb 파일을 실행한다.
3. 데이터 전처리.ipynb 파일을 실행한 후 각 데이터들이 저장되었는지 확인한다.

### 로컬학습

1. 스타트/실험/로컬학습 폴더의 1D_CNN.ipynb, EQTransformer.ipynb, smallGRU.ipynb 파일을 다운 받아 실행한다.
2. 결과를 확인한다.

### 연합학습

1. 스타트/실험/연합학습 폴더의 FL_freeze.ipynb, FL_freeze_unfreeze.ipynb 파일을 다운 받아 실행한다.
2. 결과를 확인한다.

## 실험 결과

### 로컬 학습

| 인코더 | 1D-CNN | small GRU | EQTransformer |
| --- | --- | --- | --- |
| 이진 분류 (Accuracy) | 86.30 % | 47.95 % | 100.00 % |
| 회귀 (MSE) | 0.3067 | 0.4067 | 0.0001 |

### 연합학습 Stage 1: 인코더 Freeze

| 인코더 | 1D-CNN | small GRU | EQTransformer |
| --- | --- | --- | --- |
| 이진 분류 (Accuracy) | 86.30 % | 47.95 % | 100.00 % |
| 회귀 (MAE) | 0.3107 | 0.8682 | 0.0139 |

### 연합학습 Stage 2: 인코더  Unfreeze (회귀 추가 진행)

| 인코더 | 1D-CNN | small GRU | EQTransformer |
| --- | --- | --- | --- |
| 회귀 (MAE) | 0.3112 | 0.8597 | 0.0068 |
