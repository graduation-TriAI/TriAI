import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from shared.paths import GNSS_CROSS
from shared.config import WIN, STRIDE

EXPERIMENT = "tuning_weighted_loss2_noto_train_2026-04-04"
DIST_KM = "25km"

LOG_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "logs" / EXPERIMENT
PLOT_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "plots" / EXPERIMENT
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = LOG_DIR / f"{DIST_KM}_test_predictions.csv"

df = pd.read_csv(PRED_PATH)

y_true = df["y_true"].values
y_pred = df["y_pred"].values

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

min_val = min(y_true.min(), y_pred.min())
max_val = max(y_true.max(), y_pred.max())

# 예: ±20% 범위
lower_20 = y_true * 0.8
upper_20 = y_true * 1.2
within_20 = ((y_pred >= lower_20) & (y_pred <= upper_20)).mean() * 100

high_mask = y_true >= 20

if high_mask.sum() > 0:
    high_rmse = np.sqrt(np.mean((y_true[high_mask] - y_pred[high_mask]) ** 2))
    high_within_20 = (
        ((y_pred[high_mask] >= y_true[high_mask] * 0.8) &
         (y_pred[high_mask] <= y_true[high_mask] * 1.2)).mean() * 100
    )
else:
    high_rmse = np.nan
    high_within_20 = np.nan

plt.figure(figsize=(7, 7))
plt.scatter(y_true, y_pred, alpha=0.5, s=10)

# y=x
plt.plot([min_val, max_val], [min_val, max_val])

# ±20% lines
plt.plot([min_val, max_val], [min_val * 1.2, max_val * 1.2], linestyle="--")
plt.plot([min_val, max_val], [min_val * 0.8, max_val * 0.8], linestyle="--")

plt.xlim(min_val, max_val)
plt.ylim(min_val, max_val)
plt.gca().set_aspect('equal', adjustable='box')

plt.xlabel("True PGV")
plt.ylabel("Predicted PGV")
plt.title(
    f"{DIST_KM} Test Scatter\n"
    f"RMSE: {rmse:.2f}, Within ±20%: {within_20:.2f}%\n"
    f"PGV≥20 RMSE: {high_rmse:.2f}, PGV≥20 Within ±20%: {high_within_20:.2f}%")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{DIST_KM}_test_scatter.png")
plt.show()