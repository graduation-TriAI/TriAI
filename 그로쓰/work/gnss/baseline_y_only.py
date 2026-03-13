import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from shared.paths import GNSS_TOHOKU_PROC
from work.gnss.model import GNSSModel

DATA_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_dataset_15km.npz"
MODEL_SAVE_PATH = GNSS_TOHOKU_PROC / "gnss_pgv_best_15km_y_only_SmoothL1Loss_lr=5e-4.pt"

BATCH_SIZE = 32
EPOCHS = 100 #우선은 30으로 하고 나중에 100으로 늘리기!
LR = 5e-4   #1e-3, 5e-4, 3e-4, 1e-4
TRAIN_RATIO = 0.8
SEED = 42

DROP_LAST = False #이후 True로 바꿔서 실험해 보기

class GNSSPGVDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32)  # (N, 600, 3)
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
        y_norm = (y - self.y_mean) / self.y_std
        return x, y_norm

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
                pred_orig = pred * y_std.to(device) + y_mean.to(device)
                y_orig = y * y_std.to(device) + y_mean.to(device)
                total_sq_error_orig += torch.sum((pred_orig - y_orig) ** 2).item()

            total_count += batch_size

    avg_loss = total_loss / total_count
    rmse = (total_sq_error / total_count) ** 0.5

    if y_mean is not None and y_std is not None:
        rmse_orig = (total_sq_error_orig / total_count) ** 0.5
        return avg_loss, rmse, rmse_orig

    return avg_loss, rmse, None

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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

    y_train = dataset.y[train_indices]
    y_mean = y_train.mean()
    y_std = y_train.std()

    if y_std < 1e-8:
        y_std = torch.tensor(1.0)

    print(f"y_mean: {y_mean.item():.6f}, y_std: {y_std.item():.6f}")

    train_dataset = NormalizedSubset(dataset, train_indices, y_mean, y_std)
    val_dataset = NormalizedSubset(dataset, val_indices, y_mean, y_std)

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

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

    model = GNSSModel().to(device)

    criterion = nn.SmoothL1Loss()    #MSELoss() -> SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 40, 60, 90],
        gamma=0.1
    )

    best_val_loss = float("inf")
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

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, MODEL_SAVE_PATH)
            print("   -> Best model saved")
        
        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"LR: {current_lr:.8f} | "
            f"Train Loss: {train_loss:.6f} | Train RMSE: {train_rmse:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:.6f} "
            f"(Original PGV RMSE: {val_rmse_orig:.6f})"
        )

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        
    print("\nTraining finished.")
    print("Best val loss:", best_val_loss)
    print("Best model saved to:", MODEL_SAVE_PATH)

    return model
    

if __name__ == "__main__":
    best_model = main()