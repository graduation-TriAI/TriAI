import numpy as np
import pandas as pd
from pathlib import Path

from shared.paths import GNSS_CROSS
from shared.config import WIN, STRIDE

EXPERIMENT = "cross_event_npy_weighted_mse_alpha=0.1_tohoku_train_2026-04-12"
DIST_KM = "25km"

LOG_CSV_PATH = GNSS_CROSS / f"{WIN}_{STRIDE}" / "logs" / EXPERIMENT / f"{DIST_KM}_1hz.csv"
NPY_SAVE_PATH = GNSS_CROSS / f"{WIN}_{STRIDE}" / "logs" / EXPERIMENT / f"{DIST_KM}_metric.npy"

def main():
    df = pd.read_csv(LOG_CSV_PATH)

    metric = {
        "epoch": df["epoch"].to_numpy(),
        "train_loss": df["train_loss"].to_numpy(),
        "test_loss": df["test_loss"].to_numpy(),
        "test_rmse": df["test_rmse_orig"].to_numpy(),
        "lr": df["lr"].to_numpy(),
    }

    np.save(NPY_SAVE_PATH, metric, allow_pickle=True)
    print(f"Metric npy saved to: {NPY_SAVE_PATH}")

if __name__ == "__main__":
    main()