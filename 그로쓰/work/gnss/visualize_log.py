"""
Visualization utilities for GNSS log(PGV) prediction
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from shared.paths import GNSS_TOHOKU_PROC
from work.gnss.model import GNSSModel
from 그로쓰.work.gnss.train_y_log import GNSSPGVDataset   # log(PGV) dataset import

DATA_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_dataset_15km.npz"
MODEL_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_log_best_15km.pt"

BATCH_SIZE = 64
TRAIN_RATIO = 0.8
SEED = 42


def plot_pgv_distribution(y):
    plt.figure()
    plt.hist(y, bins=50)
    plt.title("PGV Distribution")
    plt.xlabel("PGV")
    plt.ylabel("Count")
    plt.savefig("pgv_hist_log_model.png")
    plt.close()


def plot_log_pgv_distribution(y):
    plt.figure()
    plt.hist(np.log(y), bins=50)
    plt.title("log(PGV) Distribution")
    plt.xlabel("log(PGV)")
    plt.ylabel("Count")
    plt.savefig("log_pgv_hist_log_model.png")
    plt.close()


def prediction_scatter(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)

    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--")

    plt.xlabel("True PGV")
    plt.ylabel("Predicted PGV")
    plt.title("Prediction vs Ground Truth (log model)")
    plt.savefig("prediction_scatter_log_model.png")
    plt.close()


def error_vs_true(y_true, y_pred):
    abs_error = np.abs(y_pred - y_true)

    plt.figure()
    plt.scatter(y_true, abs_error, alpha=0.5)
    plt.xlabel("True PGV")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs True PGV (log model)")
    plt.savefig("error_vs_true_log_model.png")
    plt.close()


def residual_vs_true(y_true, y_pred):
    residual = y_pred - y_true

    plt.figure()
    plt.scatter(y_true, residual, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("True PGV")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residual vs True PGV (log model)")
    plt.savefig("residual_vs_true_log_model.png")
    plt.close()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # log(PGV) dataset class를 쓰더라도,
    # 원래 PGV 분포 시각화를 위해 raw npz를 직접 읽음
    raw_data = np.load(DATA_PATH)
    y_all_orig = raw_data["y"]

    plot_pgv_distribution(y_all_orig)
    plot_log_pgv_distribution(y_all_orig)

    # train_log.py와 동일한 dataset / split 재현
    dataset = GNSSPGVDataset(DATA_PATH)
    print("Total samples:", len(dataset))

    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # y는 이미 log(PGV) 상태
    y_train = dataset.y[train_indices]
    y_mean = y_train.mean()
    y_std = y_train.std()

    if y_std < 1e-8:
        y_std = torch.tensor(1.0)

    print(f"log y_mean: {y_mean.item():.6f}, log y_std: {y_std.item():.6f}")

    model = GNSSModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)

            pred_norm = model(X).cpu().squeeze(1)

            # normalized log(PGV) -> log(PGV)
            pred_log = pred_norm * y_std + y_mean
            y_log = y  # val_dataset에서 나온 y는 이미 log(PGV)

            # log(PGV) -> original PGV
            pred_orig = torch.exp(pred_log)
            y_orig = torch.exp(y_log)

            y_pred_list.append(pred_orig.numpy())
            y_true_list.append(y_orig.numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    print("True PGV range :", y_true.min(), "~", y_true.max())
    print("Pred PGV range :", y_pred.min(), "~", y_pred.max())

    prediction_scatter(y_true, y_pred)
    error_vs_true(y_true, y_pred)
    residual_vs_true(y_true, y_pred)

    print(
        "Saved: "
        "pgv_hist_log_model.png, "
        "log_pgv_hist_log_model.png, "
        "prediction_scatter_log_model.png, "
        "error_vs_true_log_model.png, "
        "residual_vs_true_log_model.png"
    )


if __name__ == "__main__":
    main()