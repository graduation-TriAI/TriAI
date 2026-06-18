## final

본 디렉토리는 지진파형, GNSS, 멀티모달 모델의 최종 학습 코드 및 최종 후처리 코드로 구성된 디렉토리입니다. 

---

### 지진파형 모델

지진파형 데이터를 입력받아 PGV를 예측하는 지진파형 모델의 학습 및 검증 코드 (`seismic.ipynb`)

- 주요 단계
    - 지진파형 데이터셋 로드 및 전처리
    - 인코더를 통한 특징 추출
    - Vs30 데이터로 지반 증폭 보정 후 디코더 기반 PGV 회귀 예측
    - 로그 스케일링을 적용한 PGV 예측 및 RMSE 평가

---

### GNSS 모델

GNSS 데이터를 입력받아 PGV를 예측하는 GNSS 모델의 학습 및 검증 코드 (`gnss/`)

- 코드 설명

    1. build_gnss_pgv_dataset_seq.py
    미리 생성된 GNSS 시계열 시퀀스 데이터에 station pair CSV의 PGV label을 매칭하여, 모델 학습에 사용할 GNSS-PGV 데이터셋을 생성합니다.

    2. encoder_ver2.py
    GNSS 시계열 데이터로부터 특징을 추출하는 Encoder 구조를 정의합니다.

    3. model_ver2.py
    PGV 추정을 위해 디코더를 포함한 전체 딥러닝 모델 아키텍처를 정의합니다.

    4. final_baseline.py
    모델 학습 및 테스트를 수행하는 메인 실행 파일입니다.

    5. plot_logs.py
    학습 과정에서 기록된 로그를 시각화합니다.

    6. plot_test_scatter.py
    테스트 데이터에 대한 예측값과 실제값을 비교하는 Scatter Plot을 생성합니다.

- How to Build

    별도의 Build 과정은 필요하지 않습니다.
    필요한 Python 패키지를 설치한 후 각 스크립트를 바로 실행할 수 있습니다.

- How to Install

    필요한 패키지를 설치합니다.

    ```bash
    pip install numpy pandas matplotlib scikit-learn torch
    ```

- How to Test

    모델 학습 및 평가를 수행합니다.

    ```bash
    python -m work.final.gnss.final_baseline
    ```

    학습 로그를 시각화합니다.

    ```bash
    python -m work.final.gnss.plot_logs
    ```

    예측 결과 Scatter Plot을 생성합니다.

    ```bash
    python -m work.final.gnss.plot_test_scatter
    ```

---

### 멀티모달 모델

지진파형 및 GNSS 데이터를 입력받아 UMIS(Universal Modality-Independent Space) 레이어로 투영하고, 이를 바탕으로 PGV를 예측하는 멀티모달 모델의 학습 및 검증 코드 (`multimodal.ipynb`)

- 주요 단계
    - 지진파형/ GNSS 데이터셋 동시 로드 및 전처리
    - 각 모달리티별 인코더를 통한 특징 추출
    - UMIS 레이어를 통한 공통 표현 공간 투영 및 특징 융합
    - Vs30 데이터로 지반 증폭 보정 후 디코더 기반 PGV 회귀 예측
    - 로그 스케일링을 적용한 PGV 예측 및 RMSE 평가

---

### 후처리

예측된 PGV 값을 후처리하는 과정을 통합한 코드 (`postprocessing.ipynb`)

- 주요 단계
    - 크리깅
    - ShakeMap 생성

---

### 디렉토리 구조

```markdown
.
├── README.md
├── seismic.ipynb
├── gnss/
│   ├── build_gnss_pgv_dataset_seq.py
│   ├── encoder_ver2.py
│   ├── final_baseline.py
│   ├── model_ver2.py
│   ├── plot_logs.py
│   └── plot_test_scatter.py
├── multimodal.ipynb
└── postprocessing.ipynb

```