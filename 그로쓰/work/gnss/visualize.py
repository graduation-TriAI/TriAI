"""
Visualization utilities for GNSS PGV prediction
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from shared.paths import GNSS_TOHOKU_PROC
from work.gnss.model import GNSSModel
from work.gnss.train import GNSSPGVDataset

DATA_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_dataset_15km.npz"
MODEL_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_best_15km.pt"

BATCH_SIZE = 64
TRAIN_RATIO = 0.8
SEED = 42


def plot_pgv_distribution(y):
    plt.figure()
    plt.hist(y, bins=50)
    plt.title("PGV Distribution")
    plt.xlabel("PGV")
    plt.ylabel("Count")
    plt.savefig("pgv_hist.png")
    plt.close()


def plot_log_pgv_distribution(y):
    plt.figure()
    plt.hist(np.log(y), bins=50)
    plt.title("log(PGV) Distribution")
    plt.xlabel("log(PGV)")
    plt.ylabel("Count")
    plt.savefig("log_pgv_hist.png")
    plt.close()


def prediction_scatter(y_true, y_pred):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)

    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--")

    plt.xlabel("True PGV")
    plt.ylabel("Predicted PGV")
    plt.title("Prediction vs Ground Truth")
    plt.savefig("prediction_scatter.png")
    plt.close()

def error_vs_true(y_true, y_pred):
    """Plot absolute error vs true PGV"""
    abs_error = np.abs(y_pred - y_true)

    plt.figure()
    plt.scatter(y_true, abs_error, alpha=0.5)
    plt.xlabel("True PGV")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs True PGV")
    plt.savefig("error_vs_true.png")
    plt.close()


def residual_vs_true(y_true, y_pred):
    """Plot residual (pred - true) vs true PGV"""
    residual = y_pred - y_true

    plt.figure()
    plt.scatter(y_true, residual, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("True PGV")
    plt.ylabel("Residual (Pred - True)")
    plt.title("Residual vs True PGV")
    plt.savefig("residual_vs_true.png")
    plt.close()

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = GNSSPGVDataset(DATA_PATH)
    y_all = dataset.y.numpy()

    # 1) 전체 데이터 분포 시각화
    plot_pgv_distribution(y_all)
    plot_log_pgv_distribution(y_all)

    # 2) train/val split을 train.py와 동일하게 재현
    train_size = int(len(dataset) * TRAIN_RATIO)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_indices = train_dataset.indices
    val_indices = val_dataset.indices

    # 3) train set 기준 정규화 통계 다시 계산
    y_train = dataset.y[train_indices]
    y_mean = y_train.mean()
    y_std = y_train.std()

    if y_std < 1e-8:
        y_std = torch.tensor(1.0)

    print(f"y_mean: {y_mean.item():.6f}, y_std: {y_std.item():.6f}")

    # 4) 모델 로드
    model = GNSSModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 5) val set만 평가
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true_list = []
    y_pred_list = []

    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)

            pred_norm = model(X).cpu().squeeze(1)

            # 정규화된 예측값 -> 원래 PGV 스케일로 복원
            pred_orig = pred_norm * y_std + y_mean

            y_pred_list.append(pred_orig.numpy())
            y_true_list.append(y.numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)

    print("True PGV range :", y_true.min(), "~", y_true.max())
    print("Pred PGV range :", y_pred.min(), "~", y_pred.max())

    prediction_scatter(y_true, y_pred)
    error_vs_true(y_true, y_pred)
    residual_vs_true(y_true, y_pred)

    print("Saved: pgv_hist.png, log_pgv_hist.png, prediction_scatter.png, error_vs_true.png, residual_vs_true.png")



if __name__ == "__main__":
    main()