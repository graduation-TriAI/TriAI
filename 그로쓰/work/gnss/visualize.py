"""
Visualization utilities for GNSS PGV prediction

This script:
1. Plots PGV distribution
2. Plots log(PGV) distribution
3. Plots prediction vs ground truth scatter
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from shared.paths import GNSS_TOHOKU_PROC
from work.gnss.model import GNSSModel
from work.gnss.train import GNSSPGVDataset


DATA_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_dataset_15km.npz"
MODEL_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_best_15km.pt"

BATCH_SIZE = 64


def plot_pgv_distribution(y):
    """Plot raw PGV distribution"""
    plt.figure()
    plt.hist(y, bins=50)
    plt.title("PGV Distribution")
    plt.xlabel("PGV")
    plt.savefig("pgv_hist.png")
    plt.close()


def plot_log_pgv_distribution(y):
    """Plot log(PGV) distribution"""
    plt.figure()
    plt.hist(np.log(y), bins=50)
    plt.title("log(PGV) Distribution")
    plt.xlabel("log(PGV)")
    plt.ylabel("Count")
    plt.savefig("log_pgv_hist.png")
    plt.close()


def prediction_scatter(y_true, y_pred):
    """Plot prediction vs ground truth"""
    plt.figure()

    plt.scatter(y_true, y_pred, alpha=0.5)

    max_val = max(y_true.max(), y_pred.max())
    plt.plot([0, max_val], [0, max_val], "r--")

    plt.xlabel("True PGV")
    plt.ylabel("Predicted PGV")
    plt.savefig("prediction_scatter.png")
    plt.close()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = GNSSPGVDataset(DATA_PATH)

    y = dataset.y.numpy()

    # -------------------------------
    # 1. PGV distribution
    # -------------------------------
    plot_pgv_distribution(y)

    # -------------------------------
    # 2. log(PGV) distribution
    # -------------------------------
    plot_log_pgv_distribution(y)

    # -------------------------------
    # 3. Load model
    # -------------------------------
    model = GNSSModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)

            pred = model(X).cpu().numpy().flatten()

            y_pred.append(pred)
            y_true.append(y.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # -------------------------------
    # 4. Prediction scatter
    # -------------------------------
    prediction_scatter(y_true, y_pred)


if __name__ == "__main__":
    main()