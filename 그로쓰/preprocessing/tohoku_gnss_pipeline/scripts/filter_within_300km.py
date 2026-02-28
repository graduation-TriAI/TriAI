
"""
Filter stations located within 300km of the Tohoku epicenter
based on precomputed distance values.
"""

from scripts.common import CSV_DIR
import pandas as pd

INPUT_CSV = CSV_DIR / "gnss_stations_sorted_by_distance.csv"
OUTPUT_CSV = CSV_DIR / "stations_within_300km.csv"

df = pd.read_csv(INPUT_CSV)

df_300 = df[df["distance_km"] <= 300].copy()

df_300 = df_300.sort_values("distance_km").reset_index(drop=True)

df_300.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Saved successfully:", OUTPUT_CSV)
print("Number of stations within 300 km:", len(df_300))
print("\nTop 10 nearest stations:")
print(df_300.head(10).to_string(index=False))
