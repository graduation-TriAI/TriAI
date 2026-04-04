import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from shared.paths import GNSS_CROSS
from shared.config import WIN, STRIDE

EXPERIMENT = "tuning_weighted_loss_tohoku_train_2026-04-04"
DIST_KM = "25km"

LOG_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "logs" / EXPERIMENT
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"{DIST_KM}_1hz.csv"

PLOT_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "plots" / EXPERIMENT
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LOG_PATH)

# 1. loss graph
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Weighted Loss (log+z-score space)")
plt.title(f"{DIST_KM} | WIN={WIN}, STRIDE={STRIDE} | Weighted Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{DIST_KM}_loss.png")
plt.show()

# 2. RMSE graph
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_rmse"], label="Train RMSE (unweighted, log+z)")
plt.plot(df["epoch"], df["val_rmse"], label="Val RMSE (unweighted, log+z)")
plt.plot(df["epoch"], df["val_rmse_orig"], label="Val RMSE (unweighted, original PGV)")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title(f"{DIST_KM} | WIN={WIN}, STRIDE={STRIDE} | RMSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{DIST_KM}_rmse.png")
plt.show()