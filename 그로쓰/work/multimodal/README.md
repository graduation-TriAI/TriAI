### multimodal

이 디렉토리는 지진파형 데이터와 GNSS 데이터를 함께 입력으로 사용하여 PGV를 예측하기 위한 멀티모달 모델의 구현 및 학습 코드를 포함합니다.

---

#### 주요 구성

- `multimodal model.ipynb`  
  지진파형 및 GNSS 시계열 데이터를 입력받아 UMIS(Universal Modality-Independent Space) 레이어로 투영하고, 이를 바탕으로 PGV를 예측하는 멀티모달 모델의 학습 및 검증 코드
  
  - 지진파형/ GNSS 데이터셋 동시 로드 및 전처리
  - 각 모달리티별 인코더를 통한 특징 추출
  - UMIS 레이어를 통한 공통 표현 공간 투영 및 특징 융합
  - 디코더 기반 PGV 회귀 예측
  - 로그 스케일링을 적용한 PGV 예측 및 RMSE 평가
