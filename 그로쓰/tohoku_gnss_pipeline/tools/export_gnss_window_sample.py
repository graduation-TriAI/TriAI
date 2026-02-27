
"""
Export the first GNSS window from an NPZ file to CSV
for inspection and visualization.
"""

import numpy as np
import pandas as pd
from scripts.common import SAMP_DIR, NPZ_DIR

NPZ_FILE = NPZ_DIR / "GNET0027_600s_300s.npz"
OUT_CSV = SAMP_DIR / "GNET0027_window_sample.csv"

data = np.load(NPZ_FILE)

#X shape: (num_windows, 600 samples, 3 components: E/N/U)
X = data["X"] 

#Select the first window as a sample
window = X[0]

df = pd.DataFrame(window, columns=["East_cm", "North_cm", "Up_cm"])
df.insert(0, "t_sec", range(len(df)))

df.to_csv(OUT_CSV, index=False)
print("Sample CSV file saved:", OUT_CSV)