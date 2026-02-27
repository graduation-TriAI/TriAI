
"""
Compare stations withing 300km against generated GNSS window files
and export the list of missing stations.
"""

from scripts.common import CSV_DIR, NPZ_DIR
import re
import pandas as pd

IN_300KM_CSV = CSV_DIR / "stations_within_300km.csv"
OUT_MISSING_CSV = CSV_DIR / "missing_stations_300km_vs_windows.csv"

#Extract station ID (e.g., GNET0001) from filename
ST_RE = re.compile(r"(GNET\d{4})", re.I)

df = pd.read_csv(IN_300KM_CSV)
stations_300 = set(df["station"].astype(str).str.upper())

made = set()
for fp in NPZ_DIR.glob("*.npz"):
    m = ST_RE.search(fp.name)
    if m:
        made.add(m.group(1).upper())

missing = sorted(stations_300 - made)

out = pd.DataFrame({"station": missing})
out.to_csv(OUT_MISSING_CSV, index=False, encoding="utf-8-sig")

print("Number of stations within 300 km:", len(stations_300))
print("Number of stations with generated windows (npz):", len(made))
print("Number of missing stations:", len(missing))
print("Saved to:", OUT_MISSING_CSV)
print("Missing stations (up to 20 examples):", missing[:20])