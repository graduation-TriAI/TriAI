import pandas as pd
from shared.paths import PAIRS_TOHOKU_CSV, PGV_TOHOKU

PAIR_CSV = PAIRS_TOHOKU_CSV / "station_pairs_ver_15km.csv"
PGV_CSV = PGV_TOHOKU / "processed_pgv_data2.csv"
OUT_CSV = PGV_TOHOKU / "gnss_pgv_labels_15km.csv"

pairs_df = pd.read_csv(PAIR_CSV)
pgv_df = pd.read_csv(PGV_CSV)

print("pairs_df columns:", pairs_df.columns.tolist())
print("pgv_df columns:", pgv_df.columns.tolist())

pairs_df = pairs_df.rename(columns={
    "gnss_station" : "gnss_station",
    "seismic_station" : "seismic_station"
})

pgv_df = pgv_df.rename(columns={
    "station" : "seismic_station",
    "pgv" : "pgv"
})

pairs_df["gnss_station"] = pairs_df["gnss_station"].astype(str).str.strip().str.upper()
pairs_df["seismic_station"] = pairs_df["seismic_station"].astype(str).str.strip().str.upper()
pgv_df["seismic_station"] = pgv_df["seismic_station"].astype(str).str.strip().str.upper()

if "pgv" in pairs_df.columns:
    pairs_df = pairs_df.drop(columns=["pgv"])

pgv_small = pgv_df[["seismic_station", "pgv"]].copy()

merged = pairs_df.merge(
    pgv_small,
    on="seismic_station",
    how="left"
)

merged = merged.dropna(subset=["pgv"])

merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

print("\n[Done]")
print("Saved to:", OUT_CSV)
print("Rows:", len(merged))
print("Matched PGV rows:", merged["pgv"].notna().sum())
print("Missing PGV rows:", merged["pgv"].isna().sum())

print("\nSample:")
print(merged.head())