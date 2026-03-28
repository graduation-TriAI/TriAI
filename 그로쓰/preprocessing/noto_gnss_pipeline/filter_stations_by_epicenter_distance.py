from shared.paths import GNSS_NOTO_CSV
import pandas as pd

INPUT_CSV = GNSS_NOTO_CSV / "gnss_stations_sorted_by_distance.csv"
OUTPUT_CSV = GNSS_NOTO_CSV / "stations_within_260km.csv"

df = pd.read_csv(INPUT_CSV)

df_out = df[df["distance_km"] <= 260].copy()

df_out = df_out.sort_values("distance_km").reset_index(drop=True)

df_out = df_out.drop(columns=["distance_km"])

df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("Saved successfully:", OUTPUT_CSV)
print("Number of stations within 250 km:", len(df_out))
print("\nTop 10 nearest stations:")
print(df_out.head(10).to_string(index=False))