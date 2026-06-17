## postprocessing

본 디렉토리는 모델 학습 후 예측 PGV 값을 후처리하는 코드로 구성된 디렉토리입니다.

---

### 후처리 흐름

1. 크리깅
	- 관측소의 위도 및 경도 정보를 기반으로 일정 해상도의 격자 생성 후, 각 격자 지점에서의 PGV 값 추정
	
2. ShakeMap 생성
	- 크리깅을 통해 생성된 격자 기반 PGV 분포를 활용하여 ShakeMap 생성

---

### 디렉토리 구조

```markdown
.
├── README.md
├── kriging/
│   ├── make_kriging_input.py
│   └── run_kriging.py
└── shakemap.ipynb

```