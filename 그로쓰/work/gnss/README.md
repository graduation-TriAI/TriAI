### gnss

이 디렉토리는 GNSS 데이터를 입력으로 PGV를 예측하기 위한 단일 모달 모델의 구현 및 학습 코드를 포함합니다.

---

#### 주요 구성

- `build_gnss_pgv_dataset_seq.py`
	기존에 생성된 GNSS npz 데이터를 기반으로, 지진파 관측소와의 거리 기준에 따라 필터링하여 모델 학습용 데이터셋(.npz)을 생성하는 코드
	
- `encoder.py`
	GNSS 시계열 데이터를 입력받아 특징을 추출하는 EQTransformer 기반 인코더 모델 정의(CNN + ResNet + BiLSTM + Transformer)
	
- `model.py`
	인코더를 기반으로 PGV를 예측하는 전체 모델 구조 정의
	
- `baseline.py`
	GNSS 데이터를 입력으로 PGV를 예측하는 기본(baseline) 모델의 학습 및 검증(train/val) 코드
	
- `plot_logs.py`
	학습 과정에서의 loss 및 RMSE 등의 로그를 시각화하는 코드

---

#### (Optional) npz 확인 코드

- `inspect_npz.py`
	생성된 npz 데이터셋의 구조 및 내용을 확인하기 위한 디버깅용 코드
