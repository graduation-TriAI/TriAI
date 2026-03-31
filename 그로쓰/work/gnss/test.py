import numpy as np
from shared.paths import GNSS_TOHOKU_PROC

npz_path = GNSS_TOHOKU_PROC / "tohoku_gnss_pgv_dataset_30km_seq.npz"   # npz 파일 경로

data = np.load(npz_path, allow_pickle=True)

# npz 내부 key 확인
print("Keys in NPZ:")
print(data.files)

print("\n==============================")

for key in data.files:
    arr = data[key]

    print(f"\n[{key}]")
    print("shape:", arr.shape)
    print("dtype:", arr.dtype)

    # 앞 20개만 출력
    if arr.ndim == 1:
        print(arr[:20])

    elif arr.ndim == 2:
        print(arr[:20])

    elif arr.ndim == 3:
        print("First sample first 20 rows:")
        print(arr[0][:20])