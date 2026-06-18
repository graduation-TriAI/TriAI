from shared.paths import GNSS_HOKKAIDO_CSV
import pandas as pd
import numpy as np

INPUT_CSV = GNSS_HOKKAIDO_CSV / "stations_latlon.csv"
OUTPUT_CSV = GNSS_HOKKAIDO_CSV / "gnss_stations_sorted_by_distance.csv"

# Epicenter coordinates (lat, lon) for the 2018 Hokkaido Eastern Iburi earthquake
HOKKAIDO_LAT = 42.690
HOKKAIDO_LON = 142.007

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
    HOKKAIDO_LAT,
    HOKKAIDO_LON
)

df_sorted = df.sort_values("distance_km").reset_index(drop=True)

df_sorted.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("File saved:", OUTPUT_CSV)
print("\nTop 10 nearest stations:")
print(df_sorted.head(10).to_string(index=False))