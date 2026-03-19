import numpy as np
import pandas as pd
from pathlib import Path
from shared.paths import GNSS_TOHOKU_PROC, GNSS_TOHOKU_CSV

# ===== 경로 설정 =====
npz_path = GNSS_TOHOKU_PROC / "tohoku_gnss_station_seq_600_300.npz"
out_csv = GNSS_TOHOKU_CSV / "gnss_station_info_for_pairing.csv"

# ===== npz 로드 =====
data = np.load(npz_path, allow_pickle=True)

station = data["station"]
lat = data["lat"]
lon = data["lon"]

# ===== pandas DataFrame 생성 =====
df = pd.DataFrame({
    "station": station,
    "latitude": lat,
    "longitude": lon
})

# ===== CSV 저장 =====
df.to_csv(out_csv, index=False)

print(f"Saved to {out_csv}")