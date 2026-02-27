
"""
Common path configuration for the project

This module defines base directories used across the scritps.
"""

from pathlib import Path

#Project root directory (two levels above this file)
BASE_DIR = Path(__file__).resolve().parent.parent

#Data directories
CSV_DIR = BASE_DIR / "data" / "csv"
NPZ_DIR = BASE_DIR / "data" / "npz" / "gnss_windowed_600_300"
RAW_DIR = BASE_DIR / "data" / "raw"
DATASET_DIR = RAW_DIR / "datasets"
SAMP_DIR = BASE_DIR / "data" / "sample"