
"""
Export the first GNSS window from an NPZ file to CSV
for inspection and visualization.
"""

import numpy as np
import pandas as pd
from shared.paths import CSV_SAMPLES, GNSS_TOHOKU_PROC
from shared.config import WIN, STRIDE

GNSS_NPZ_DIR = GNSS_TOHOKU_PROC / f"gnss_windowed_{WIN}_{STRIDE}"
NPZ_FILE = GNSS_NPZ_DIR / "GNET0023_600s_300s.npz"
OUT_CSV = CSV_SAMPLES / "GNET0023_window_sample.csv"

data = np.load(NPZ_FILE)
print(data.files)

#X shape: (num_windows, 600 samples, 3 components: E/N/U)
X = data["X"] 


#Select the first window as a sample
window = X[0]

df = pd.DataFrame(window, columns=["East_cm", "North_cm", "Up_cm"])
df.insert(0, "t_sec", range(len(df)))

df.to_csv(OUT_CSV, index=False)
print("Sample CSV file saved:", OUT_CSV)
print("Station:", data["station"])
print("Latitude:", data["lat"])
print("Longitude:", data["lon"])
print("Sampling rate:", data["fs"])
print("Start time of first window:", data["start_sec"][0])
print("End time of first window:", data["end_sec"][0])