
"""
Extract station IDs from generated 600s window NPZ files
and save the unique station list to CSV.
"""

from shared.paths import GNSS_CSV_DIR, GNSS_NPZ_DIR
import re
import pandas as pd

OUT = GNSS_CSV_DIR / "stations_tohoku_valid_600s.csv"

#Extract station ID (e.g., GNET0001) from filename
ST_RE = re.compile(r"(GNET\d{4})", re.I)

stations = []
for fp in GNSS_NPZ_DIR.glob("*.npz"):
    m = ST_RE.search(fp.name)
    if m:
        stations.append(m.group(1).upper())

df = pd.DataFrame({"station": sorted(set(stations))})
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print("saved:", OUT, "count:", len(df))
