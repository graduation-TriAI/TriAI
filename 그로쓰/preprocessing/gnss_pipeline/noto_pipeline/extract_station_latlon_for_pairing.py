import numpy as np
import pandas as pd
from pathlib import Path
from shared.paths import GNSS_NOTO_PROC, GNSS_NOTO_CSV
from shared.config import WIN, STRIDE

# ===== 경로 설정 =====
base_dir = GNSS_NOTO_PROC / f"{WIN}_{STRIDE}" 
npz_path = base_dir / f"noto_gnss_station_seq_{WIN}_{STRIDE}.npz"
out_csv = GNSS_NOTO_CSV / "gnss_station_info_for_pairing_260km.csv"

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