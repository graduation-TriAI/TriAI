import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from shared.paths import GNSS_CROSS
from shared.config import WIN, STRIDE

EXPERIMENT = "cross_event_tohoku_train_noto_test_2026-05-06"
DIST_KM = "25km"

LOG_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "logs" / EXPERIMENT
PLOT_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "plots" / EXPERIMENT
PLOT_DIR.mkdir(parents=True, exist_ok=True)

PRED_PATH = LOG_DIR / f"{DIST_KM}_test_predictions.csv"
SAVE_PATH = PLOT_DIR / f"{DIST_KM}_test_scatter.png"

df = pd.read_csv(PRED_PATH)

y_true = df["y_true"].to_numpy()
y_pred = df["y_pred"].to_numpy()

rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

# 예시 코드 느낌 맞추기: 축 범위 고정
max_val = 100

plt.figure(figsize=(10, 10))
plt.scatter(
    y_true,
    y_pred,
    alpha=0.5,
    color="dodgerblue",
    edgecolor="k",
    s=50,
)

# Perfect prediction line
plt.plot(
    [0, max_val],
    [0, max_val],
    color="red",
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction (y=x)",
)

# ±10 error lines
plt.plot(
    [0, max_val],
    [10, max_val + 10],
    color="gray",
    linestyle=":",
    alpha=0.7,
    label="+10 Error",
)
plt.plot(
    [0, max_val],
    [-10, max_val - 10],
    color="gray",
    linestyle=":",
    alpha=0.7,
    label="-10 Error",
)

plt.title(
    f"[{WIN}_{STRIDE}] Best Model: Actual vs Predicted (Original Scale)\nRMSE: {rmse:.4f}",
    fontsize=16,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Actual PGV", fontsize=14)
plt.ylabel("Predicted PGV", fontsize=14)

plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.gca().set_aspect("equal", adjustable="box")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig(SAVE_PATH, dpi=300)
plt.show()

print(f"Saved to: {SAVE_PATH}")

print(y_true[:5], y_pred[:5])
print(df["y_pred"].describe())
print(df.sort_values("y_pred", ascending=False).head(10))