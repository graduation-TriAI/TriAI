### seismic

이 디렉토리는 지진파형 데이터를 입력으로 PGV를 예측하기 위한 단일 모달 모델의 구현 및 학습 코드를 포함합니다.

---

#### 주요 구성

- `seismic_model_ver2.ipynb`  
  관측소 단위의 시계열 데이터((관측소 수, 윈도우 개수, 시간, 채널))를 입력으로 받아 PGV를 예측하는 모델의 학습 및 검증 코드

  - Data Augmentation (amplitude scaling, channel dropout, noise 추가)
  - EQTransformer 기반 인코더 (CNN + ResNet + BiLSTM + Transformer)
  - Attention pooling 기반 디코더 및 회귀 모델
  - 로그 스케일링을 적용한 PGV 예측 및 RMSE 평가

- `seismic_model_ver1.ipynb`  
  데이터셋 구조 변경 이전((윈도우 개수, 시간, 채널)) 기반의 초기 실험 코드
