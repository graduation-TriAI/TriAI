from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] 

#top-level folders
DATA = ROOT / "data"
PREPROCESSING = ROOT / "preprocessing"
WORK = ROOT / "work"

#data base
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
CSV = DATA / "csv"

#modalities
RAW_GNSS = RAW / "gnss"
RAW_SEIS = RAW / "seismic"

PROC_GNSS = PROCESSED / "gnss"
PROC_SEIS = PROCESSED / "seismic"
PROC_PGV = PROCESSED / "pgv"

CSV_GNSS = CSV / "gnss"
CSV_SEIS = CSV / "seismic"
CSV_SAMPLES = CSV / "samples"
CSV_PAIRS = CSV / "station_pairs"

#events
TOHOKU = "tohoku"

# ---- GNSS: tohoku ----
GNSS_TOHOKU_RAW = RAW_GNSS / TOHOKU
GNSS_TOHOKU_PROC = PROC_GNSS / TOHOKU
GNSS_TOHOKU_CSV = CSV_GNSS / TOHOKU

# ---- SEISMIC: tohoku ----
SEIS_TOHOKU_RAW = RAW_SEIS / TOHOKU
SEIS_TOHOKU_PROC = PROC_SEIS / TOHOKU
SEIS_TOHOKU_CSV = CSV_SEIS / TOHOKU

# ---- Station pairs ----
PAIRS_TOHOKU_CSV = CSV_PAIRS / TOHOKU

# ---- PGV ----
PGV_TOHOKU = PROC_PGV / TOHOKU