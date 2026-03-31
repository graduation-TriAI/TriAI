import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from shared.paths import GNSS_TOHOKU_PROC
from shared.config import WIN, STRIDE

EXPERIMENT = "baseline_noto_2026-03-31"
DIST_KM = "10km"

LOG_DIR = GNSS_TOHOKU_PROC / f"{WIN}_{STRIDE}" / "logs" / EXPERIMENT
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"{DIST_KM}_1hz.csv"

PLOT_DIR = GNSS_TOHOKU_PROC / f"{WIN}_{STRIDE}" / "plots" / EXPERIMENT
PLOT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(LOG_PATH)

# 1. loss graph
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"{DIST_KM} | WIN={WIN}, STRIDE={STRIDE} | Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{DIST_KM}_loss.png")
plt.show()

# 2. RMSE graph
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_rmse"], label="Train RMSE")
plt.plot(df["epoch"], df["val_rmse"], label="Val RMSE")
plt.plot(df["epoch"], df["val_rmse_orig"], label="Val Original PGV RMSE")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.title(f"{DIST_KM} | WIN={WIN}, STRIDE={STRIDE} | RMSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{DIST_KM}_rmse.png")
plt.show()