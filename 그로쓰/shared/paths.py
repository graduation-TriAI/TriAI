from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREPROCESSING = ROOT / "preprocessing"
WORK = ROOT / "work"

TOHOKU_GNSS = PREPROCESSING / "tohoku_gnss_pipeline"
TOHOKU_SEIS = PREPROCESSING / "tohoku_seismic_pipeline"

GNSS_DATA = TOHOKU_GNSS / "data"
SEIS_DATA = TOHOKU_SEIS / "data"

GNSS_NPZ_DIR = GNSS_DATA / "npz" / "gnss_windowed_600_300"
GNSS_CSV_DIR = GNSS_DATA / "csv"
GNSS_RAW_DIR = GNSS_DATA / "raw"
GNSS_SAMP_DIR = GNSS_DATA / "sample"
GNSS_DATASET_DIR = GNSS_RAW_DIR / "datasets"

SEIS_CSV_DIR = SEIS_DATA / "csv"
SEIS_RAW_DIR = SEIS_DATA / "raw"