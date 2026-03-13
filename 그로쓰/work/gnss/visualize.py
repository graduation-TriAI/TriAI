"""
Visualization utilities for GNSS PGV prediction
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split

from shared.paths import GNSS_TOHOKU_PROC
from work.gnss.model import GNSSModel

DATA_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_dataset_15km.npz"
MODEL_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_best_15km_y_only_MSE_lr=5e-4.pt"
OUT_DIR = GNSS_TOHOKU_PROC

BATCH_SIZE = 64
TRAIN_RATIO = 0.8
SEED = 42


class GNSSPGVDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y


def plot_pgv_distribution(y):
    plt.figure()
    plt.hist(y, bins=50)
    plt.title("PGV Distribution")
    plt.xlabel("PGV")
    plt.ylabel("Count")
    plt.savefig(OUT_DIR / "pgv_hist.png")
    plt.close()


def plot_log_pgv_distribution(y):
    plt.figure()
    plt.hist(np.log1p(y), bins=50)
    plt.title("log1p(PGV) Distribution")
    plt.xlabel("log1p(PGV)")
    plt.ylabel("Count")
    plt.savefig(OUT_DIR / "log_pgv_hist.png")
    plt.close()


def prediction_scatter(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5, s=20)

    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--")

    plt.xlabel("True PGV")
    plt.ylabel("Predicted PGV")
    plt.title("Prediction vs Ground Truth")
    plt.savefig(OUT_DIR / "prediction_scatter.png")
    plt.close()


def error_vs_true(y_true, y_pred):
    abs_error = np.abs(y_pred - y_true)

    plt.figure()
    plt.scatter(y_true, abs_error, alpha=0.5, s=20)
    plt.xlabel("True PGV")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs True PGV")
    plt.savefig(OUT_DIR / "error_vs_true.png")
    plt.close()


def residual_vs_true(y_true, y_pred):
    residual = y_pred - y_true

    plt.figure()
    plt.scatter(y_true, residual, alpha=0.5, s=20)
    plt.axhline(0, linestyle="--", alpha=0.7)
    plt.xlabel("True PGV")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residual vs True PGV")
    plt.savefig(OUT_DIR / "residual_vs_true.png")
    plt.close()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = GNSSPGVDataset(DATA_PATH)
    y_all = dataset.y.numpy()

    # 전체 분포
    plot_pgv_distribution(y_all)
    plot_log_pgv_distribution(y_all)

    # train/val split 재현
    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_indices = train_dataset.indices

    # train 기준 정규화 통계 계산
    y_train = dataset.y[train_indices]
    y_mean = y_train.mean()
    y_std = y_train.std()

    if y_std < 1e-8:
        y_std = torch.tensor(1.0)

    print(f"y_mean: {y_mean.item():.6f}, y_std: {y_std.item():.6f}")

    # 모델 로드
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
            pred_orig = pred_norm * y_std + y_mean

            y_pred_list.append(pred_orig.numpy())
            y_true_list.append(y.numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))

    print("True PGV range :", y_true.min(), "~", y_true.max())
    print("Pred PGV range :", y_pred.min(), "~", y_pred.max())
    print(f"Val RMSE: {rmse:.6f}")
    print(f"Val MAE : {mae:.6f}")

    prediction_scatter(y_true, y_pred)
    error_vs_true(y_true, y_pred)
    residual_vs_true(y_true, y_pred)

    print("Saved:")
    print(OUT_DIR / "pgv_hist.png")
    print(OUT_DIR / "log_pgv_hist.png")
    print(OUT_DIR / "prediction_scatter.png")
    print(OUT_DIR / "error_vs_true.png")
    print(OUT_DIR / "residual_vs_true.png")


if __name__ == "__main__":
    main()