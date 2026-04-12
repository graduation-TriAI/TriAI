import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import pandas as pd

from shared.paths import GNSS_NOTO_PROC, GNSS_TOHOKU_PROC, GNSS_CROSS
from work.gnss.model import GNSSModel
from shared.config import WIN, STRIDE

EXPERIMENT = "cross_event_npy_weighted_mse_alpha=0.1_noto_train_2026-04-12"
DIST_KM = "25km"

TARGET_DATA_PATH = GNSS_TOHOKU_PROC / f"{WIN}_{STRIDE}" / "1hz" / f"tohoku_gnss_pgv_dataset_{DIST_KM}_seq.npz"
TRAIN_DATA_PATH = GNSS_NOTO_PROC / f"{WIN}_{STRIDE}" / "1hz" / f"noto_gnss_pgv_dataset_{DIST_KM}_seq.npz"

MODEL_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "models" / EXPERIMENT / DIST_KM
LOG_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "logs" / EXPERIMENT

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_DIR / "best_1hz.pt"
LOG_SAVE_PATH = LOG_DIR / f"{DIST_KM}_1hz.csv"

BATCH_SIZE = 8
EPOCHS = 150 #우선은 30으로 하고 나중에 100으로 늘리기! 100/150
LR = 1e-3   #1e-3, 5e-4, 3e-4, 1e-4
VAL_RATIO = 0.2
SEED = 42

DROP_LAST = True #이후 True로 바꿔서 실험해 보기

class GNSSPGVDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32)  # (N, W, T, 3)
        self.y = torch.tensor(data["y"], dtype=torch.float32)  # (N,)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        return x, y
    
class NormalizedSubset(Dataset):
    def __init__(self, base_dataset, indices, y_mean, y_std):
        self.base_dataset = base_dataset
        self.indices = indices
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y = self.base_dataset[real_idx]
        y_log = torch.log1p(y)
        y_norm = (y_log - self.y_mean) / self.y_std
        return x, y_norm

def weighted_mse_loss(pred, target, y_mean, y_std, alpha=0.1):
    """
    Weighted MSE where larger original PGV gets a larger weight.
    pred and target are in normalized space.
    """
    y_mean_dev = y_mean.to(target.device)
    y_std_dev = y_std.to(target.device)

    target_log = target * y_std_dev + y_mean_dev
    target_orig = torch.expm1(target_log)

    weights = 1.0 + alpha * target_orig
    loss = weights * (pred - target) ** 2
    return loss.mean()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_sq_error = 0.0
    total_count = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()

        pred = model(X)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        batch_size = X.size(0)

        total_loss += loss.item() * batch_size
        total_sq_error += torch.sum((pred - y) ** 2).item()
        total_count += batch_size

    avg_loss = total_loss / total_count
    rmse = (total_sq_error / total_count) ** 0.5

    return avg_loss, rmse

def evaluate(model, loader, criterion, device, y_mean=None, y_std=None):
    model.eval()

    total_loss = 0.0
    total_sq_error = 0.0
    total_sq_error_orig = 0.0 
    total_count = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)

            pred = model(X)
            loss = criterion(pred, y)

            batch_size = X.size(0)

            total_loss += loss.item() * batch_size
            total_sq_error += torch.sum((pred - y) ** 2).item()

            if y_mean is not None and y_std is not None:
                pred_log = pred * y_std.to(device) + y_mean.to(device)
                y_log = y * y_std.to(device) + y_mean.to(device)

                pred_orig = torch.expm1(pred_log)
                y_orig = torch.expm1(y_log)
                total_sq_error_orig += torch.sum((pred_orig - y_orig) ** 2).item()

            total_count += batch_size

    avg_loss = total_loss / total_count
    rmse = (total_sq_error / total_count) ** 0.5

    if y_mean is not None and y_std is not None:
        rmse_orig = (total_sq_error_orig / total_count) ** 0.5
        return avg_loss, rmse, rmse_orig

    return avg_loss, rmse, None

def predict(model, loader, device, y_mean, y_std):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)

            pred = model(X)

            y_mean_dev = y_mean.to(device)
            y_std_dev = y_std.to(device)

            pred_log = pred * y_std_dev + y_mean_dev
            y_log = y * y_std_dev + y_mean_dev

            print("pred min/max:", pred.min().item(), pred.max().item())
            print("pred_log min/max:", pred_log.min().item(), pred_log.max().item())

            pred_orig = torch.expm1(pred_log)
            y_orig = torch.expm1(y_log)

            all_true.append(y_orig.cpu().numpy())
            all_pred.append(pred_orig.cpu().numpy())

    all_true = np.concatenate(all_true, axis=0).reshape(-1)
    all_pred = np.concatenate(all_pred, axis=0).reshape(-1)

    return all_true, all_pred

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_base_dataset = GNSSPGVDataset(TRAIN_DATA_PATH)
    target_base_dataset = GNSSPGVDataset(TARGET_DATA_PATH)
    
    print("Train-event total samples:", len(train_base_dataset))
    print("Target-event total samples:", len(target_base_dataset))

    y_train_log = torch.log1p(train_base_dataset.y)
    y_mean = y_train_log.mean()
    y_std = y_train_log.std()

    if y_std < 1e-8:
        y_std = torch.tensor(1.0)

    print(f"Train y_mean: {y_mean.item():.6f}, y_std: {y_std.item():.6f}")

    train_indices = list(range(len(train_base_dataset)))

    target_size = len(target_base_dataset)
    val_size = max(1, int(target_size * VAL_RATIO))
    test_size = target_size - val_size

    val_subset_raw, test_subset_raw = random_split(
        target_base_dataset,
        [val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    val_indices = val_subset_raw.indices
    test_indices = test_subset_raw.indices

    train_dataset = NormalizedSubset(train_base_dataset, train_indices, y_mean, y_std)
    val_dataset = NormalizedSubset(target_base_dataset, val_indices, y_mean, y_std)
    test_dataset = NormalizedSubset(target_base_dataset, test_indices, y_mean, y_std)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=DROP_LAST
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False
    )

    model = GNSSModel().to(device)

    criterion = lambda pred, target: weighted_mse_loss(
        pred, target, y_mean, y_std, alpha=0.1
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[40, 80, 110],    #milestones=[20, 40, 60, 90] / milestones=[30, 60, 90, 120],
        gamma=0.1
    )

    log_epochs = []
    log_lrs = []
    log_train_loss = []
    log_train_rmse = []
    log_val_loss = []
    log_val_rmse = []
    log_val_rmse_orig = []
    log_test_loss = []
    log_test_rmse = []
    log_test_rmse_orig = []

    best_val_rmse_orig = float("inf")
    best_model_weights = None

    print("\n ============================ Train ===============================")

    #Training Loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_rmse = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_rmse, val_rmse_orig = evaluate(
            model, val_loader, criterion, device, y_mean, y_std
        )

        test_loss, test_rmse, test_rmse_orig = evaluate(
            model, test_loader, criterion, device, y_mean, y_std
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if val_rmse_orig < best_val_rmse_orig:
            best_val_rmse_orig = val_rmse_orig
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, MODEL_SAVE_PATH)
            print("   -> Best model saved")
        
        log_epochs.append(epoch)
        log_lrs.append(current_lr)
        log_train_loss.append(train_loss)
        log_train_rmse.append(train_rmse)
        log_val_loss.append(val_loss)
        log_val_rmse.append(val_rmse)
        log_val_rmse_orig.append(val_rmse_orig)
        log_test_loss.append(test_loss)
        log_test_rmse.append(test_rmse)
        log_test_rmse_orig.append(test_rmse_orig)

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.6f} | Train RMSE: {train_rmse:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:.6f} "
            f"(Original PGV RMSE: {val_rmse_orig:.6f})"
        )

        scheduler.step()

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    print("\n============================ Test ============================")

    y_true_orig, y_pred_orig = predict(
        model, test_loader, device, y_mean, y_std
    )

    pred_df = pd.DataFrame({
        "y_true": y_true_orig,
        "y_pred": y_pred_orig,
    })
    pred_save_path = LOG_DIR / f"{DIST_KM}_test_predictions.csv"
    pred_df.to_csv(pred_save_path, index=False)

    print(
        f"Test Loss: {test_loss:.6f} | "
        f"Test RMSE: {test_rmse:.6f} | "
        f"Test Original PGV RMSE: {test_rmse_orig:.6f}"
    )

    print("Test predictions saved to:", pred_save_path)

    log_df = pd.DataFrame({
        "epoch": log_epochs,
        "lr": log_lrs,
        "train_loss": log_train_loss,
        "train_rmse": log_train_rmse,
        "val_loss": log_val_loss,
        "val_rmse": log_val_rmse,
        "val_rmse_orig": log_val_rmse_orig,
        "test_loss": log_test_loss,
        "test_rmse": log_test_rmse,
        "test_rmse_orig": log_test_rmse_orig,
    })
    log_df.to_csv(LOG_SAVE_PATH, index=False)

    print("\nTraining finished.")
    print("Best Original PGV RMSE:", best_val_rmse_orig)
    print("Test Original PGV RMSE:", test_rmse_orig)
    print("Best model saved to:", MODEL_SAVE_PATH)

    return model
    

if __name__ == "__main__":
    best_model = main()