
"""
Compute great-circle distances(km) from the 2011 Tohoku earthquake epicenter
to each GNSS station, then sort stations by distance and save the result.
"""

from scripts.common import CSV_DIR
import pandas as pd
import numpy as np

INPUT_CSV = CSV_DIR / "stations_latlon_1221.csv"
OUTPUT_CSV = CSV_DIR / "gnss_stations_sorted_by_distance.csv"

#Epicenter coordinates (lat, lon) for the 2011 Tohoku earthquake
TOHOKU_LAT = 38.297
TOHOKU_LON = 142.373

def haversine(lat1, lon1, lat2, lon2):
    """Return great-circle distance in km between (lat1, lon1) and (lat2, lon2)."""
    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c

df = pd.read_csv(INPUT_CSV)

df["distance_km"] = haversine(
    df["latitude"],
    df["longitude"],
    TOHOKU_LAT,
    TOHOKU_LON
)

df_sorted = df.sort_values("distance_km").reset_index(drop=True)

df_sorted.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("File saved:", OUTPUT_CSV)
print("\nTop 10 nearest stations:")
print(df_sorted.head(10).to_string(index=False))
