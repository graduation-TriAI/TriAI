import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import torch.optim as optim
from pathlib import Path
import pandas as pd
from torch.cuda.amp import autocast, GradScaler
import geopandas as gpd
from shapely.geometry import Point

from shared.paths import GNSS_NOTO_PROC, GNSS_TOHOKU_PROC, GNSS_KUMAMOTO_PROC, GNSS_HOKKAIDO_PROC, GNSS_CROSS, POST
from work.gnss.model_ver2 import GNSSModel
from shared.config import WIN, STRIDE

EXPERIMENT = "final_test_noto_2026-05-17"
DIST_KM = "25km"

TRAIN_DATA_PATHS = [
    GNSS_HOKKAIDO_PROC / f"{WIN}_{STRIDE}" / "1hz" / f"hokkaido_gnss_pgv_dataset_{DIST_KM}_seq.npz",
    GNSS_KUMAMOTO_PROC / f"{WIN}_{STRIDE}" / "1hz" / f"kumamoto_gnss_pgv_dataset_{DIST_KM}_seq.npz",
    GNSS_TOHOKU_PROC / f"{WIN}_{STRIDE}" / "1hz" / f"tohoku_gnss_pgv_dataset_{DIST_KM}_seq.npz",
]

TARGET_DATA_PATH = GNSS_NOTO_PROC / f"{WIN}_{STRIDE}" / "1hz" / f"noto_gnss_pgv_dataset_{DIST_KM}_seq.npz"

VS30_SHP_PATH = POST / "vs30 데이터" / "Z-V4-JAPAN-AMP-VS400_M250-SHAPE.shp"

MODEL_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "models" / EXPERIMENT / DIST_KM
LOG_DIR = GNSS_CROSS / f"{WIN}_{STRIDE}" / "logs" / EXPERIMENT

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_DIR / "best_1hz.pt"
LOG_SAVE_PATH = LOG_DIR / f"{DIST_KM}_1hz.csv"

BATCH_SIZE = 16
EPOCHS = 100 
LR = 3e-4   
VAL_RATIO = 0.2
SEED = 42

DROP_LAST = True 
PATIENCE = 40

class GNSSPGVDataset(Dataset):
    def __init__(self, npz_path, vs30_shp_path=VS30_SHP_PATH):
        data = np.load(npz_path, allow_pickle=True)

        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)

        print("NPZ keys:", data.files)

        if "lat" in data.files and "lon" in data.files:
            lats = data["lat"]
            lons = data["lon"]
        elif "latitude" in data.files and "longitude" in data.files:
            lats = data["latitude"]
            lons = data["longitude"]
        elif "gnss_lat" in data.files and "gnss_lon" in data.files:
            lats = data["gnss_lat"]
            lons = data["gnss_lon"]
        else:
            raise KeyError(
                f"{npz_path} 안에 위도/경도 키가 없음. "
                f"현재 keys={data.files}. SHP에서 VS30 매칭하려면 lat/lon이 필요함."
            )

        margin = 0.5
        bbox = (
            float(np.min(lons)) - margin,
            float(np.min(lats)) - margin,
            float(np.max(lons)) + margin,
            float(np.max(lats)) + margin,
        )

        vs30_gdf = gpd.read_file(vs30_shp_path, bbox=bbox).to_crs("EPSG:4326")

        pts_gdf = gpd.GeoDataFrame(
            {"idx": np.arange(len(lats))},
            geometry=[Point(float(lo), float(la)) for lo, la in zip(lons, lats)],
            crs="EPSG:4326"
        )

        joined = gpd.sjoin(
            pts_gdf,
            vs30_gdf[["AVS", "ARV", "geometry"]],
            how="left",
            predicate="within"
        ).sort_values("idx")

        vs30 = joined["ARV"].values

        if np.isnan(vs30).any():
            fallback = np.nanmedian(vs30)
            vs30 = np.where(np.isnan(vs30), fallback, vs30)

        vs30 = np.asarray(vs30, dtype=np.float32)

        # 멀티모달처럼 log + z-score
        log_vs30 = np.log(np.clip(vs30, 1e-3, None))
        mu = log_vs30.mean()
        sigma = log_vs30.std()
        if sigma < 1e-8:
            sigma = 1.0

        vs30_feat = (log_vs30 - mu) / sigma
        self.vs30 = torch.tensor(vs30_feat, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.vs30[idx]
    
class NormalizedDataset(Dataset):
    def __init__(self, base_dataset, y_mean, y_std, indices=None):
        self.base_dataset = base_dataset
        self.indices = list(range(len(base_dataset))) if indices is None else indices
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x, y, vs30 = self.base_dataset[real_idx]

        y_log = torch.log1p(y)
        y_norm = (y_log - self.y_mean) / self.y_std

        return x, y_norm, vs30

def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.05):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class HuberMSELoss(nn.Module):
    def __init__(self, alpha=0.5, delta=1.0):
        super().__init__()
        self.alpha = alpha
        self.huber = nn.HuberLoss(delta=delta)
        self.mse   = nn.MSELoss()

    def forward(self, pred, target):
        return self.alpha * self.mse(pred, target) + (1 - self.alpha) * self.huber(pred, target)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_sq_error = 0.0
    total_count = 0

    for X, y, vs30 in loader:
        X = X.to(device)
        y = y.to(device).unsqueeze(1)
        vs30 = vs30.to(device)

        optimizer.zero_grad()

        pred = model(X, vs30)
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
        for X, y, vs30 in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)
            vs30 = vs30.to(device)

            pred = model(X, vs30)
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
        for X, y, vs30 in loader:
            X = X.to(device)
            y = y.to(device).unsqueeze(1)
            vs30 = vs30.to(device)

            pred = model(X, vs30)

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

    train_base_datasets = [GNSSPGVDataset(path) for path in TRAIN_DATA_PATHS]
    target_base_dataset = GNSSPGVDataset(TARGET_DATA_PATH)

    for path, dataset in zip(TRAIN_DATA_PATHS, train_base_datasets):
        print(f"Train event: {path.name} | samples: {len(dataset)}")

    print("Target-event total samples:", len(target_base_dataset))

    # 중요: y_mean, y_std는 train 3개 이벤트에서만 계산
    y_train_log_all = torch.cat([
        torch.log1p(dataset.y) for dataset in train_base_datasets
    ], dim=0)

    y_mean = y_train_log_all.mean()
    y_std = y_train_log_all.std()

    if y_std < 1e-8:
        y_std = torch.tensor(1.0)

    print(f"Train y_mean: {y_mean.item():.6f}, y_std: {y_std.item():.6f}")

    # train 3개 이벤트 합치기
    train_datasets = [
        NormalizedDataset(dataset, y_mean, y_std)
        for dataset in train_base_datasets
    ]

    train_dataset = ConcatDataset(train_datasets)

    # target event에서 val/test split
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

    val_dataset = NormalizedDataset(target_base_dataset, y_mean, y_std, val_indices)
    test_dataset = NormalizedDataset(target_base_dataset, y_mean, y_std, test_indices)

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

    criterion = HuberMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    scheduler = get_cosine_warmup_scheduler(optimizer, warmup_epochs=5, total_epochs=EPOCHS)

    log_epochs = []
    log_lrs = []
    log_train_loss = []
    log_train_rmse = []
    log_val_loss = []
    log_val_rmse = []
    log_val_rmse_orig = []

    best_val_rmse_orig = float("inf")
    best_model_weights = None
    patience_counter = 0

    print("\n ============================ Train ===============================")

    #Training Loop
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_rmse = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_rmse, val_rmse_orig = evaluate(
            model, val_loader, criterion, device, y_mean, y_std
        )

        current_lr = optimizer.param_groups[0]["lr"]

        if val_rmse_orig < best_val_rmse_orig:
            best_val_rmse_orig = val_rmse_orig
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save(best_model_weights, MODEL_SAVE_PATH)
            patience_counter = 0
            print("   -> Best model saved")

        else:
            patience_counter += 1
            print(f"   -> No improvement: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
        
        log_epochs.append(epoch)
        log_lrs.append(current_lr)
        log_train_loss.append(train_loss)
        log_train_rmse.append(train_rmse)
        log_val_loss.append(val_loss)
        log_val_rmse.append(val_rmse)
        log_val_rmse_orig.append(val_rmse_orig)

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
    test_loss, test_rmse, test_rmse_orig = evaluate(
        model, test_loader, criterion, device, y_mean, y_std
    )

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
    })
    log_df.to_csv(LOG_SAVE_PATH, index=False)

    print("\nTraining finished.")
    print("Best Original PGV RMSE:", best_val_rmse_orig)
    print("Test Original PGV RMSE:", test_rmse_orig)
    print("Best model saved to:", MODEL_SAVE_PATH)

    return model
    

if __name__ == "__main__":
    best_model = main()