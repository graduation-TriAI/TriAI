
"""
Extract stations within the Tohoku region using a rough latitude/longitude bounding box.
"""

from pathlib import Path
import re
import pandas as pd
from scripts.common import CSV_DIR

IN_CSV  = CSV_DIR / "stations_latlon_1221.csv"
OUT_CSV = CSV_DIR / "stations_tohoku_bbox.csv"

# -------------------------
# Bounding box for Tohoku (approximate)
# -------------------------
LAT_MIN, LAT_MAX = 36.5, 41.7
LON_MIN, LON_MAX = 138.8, 142.9

def pick_col(df: pd.DataFrame, candidates):
    """Return the first matching column name (case/whitespace-insensitive)."""
    norm = {c: re.sub(r"\s+", "", str(c)).lower() for c in df.columns}
    for cand in candidates:
        cand_n = re.sub(r"\s+", "", cand).lower()
        for orig, n in norm.items():
            if n == cand_n or cand_n in n:
                return orig
    raise KeyError(f"Column not found. candidates={candidates}, available={list(df.columns)}")

df = pd.read_csv(IN_CSV)

lat_col = pick_col(df, ["lat", "latitude"])
lon_col = pick_col(df, ["lon", "longitude"])

df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
df = df.dropna(subset=[lat_col, lon_col]).copy()

# Filter stations by the bounding box
out = df[
    (df[lat_col] >= LAT_MIN) & (df[lat_col] <= LAT_MAX) &
    (df[lon_col] >= LON_MIN) & (df[lon_col] <= LON_MAX)
].copy()

out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print(f"[OK] saved: {OUT_CSV} (n={len(out)})")
print(out.head(10))