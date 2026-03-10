import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.paths import GNSS_TOHOKU_PROC
from shared.config import WIN, STRIDE
import numpy as np

GNSS_NPZ_DIR = GNSS_TOHOKU_PROC / f"gnss_windowed_{WIN}_{STRIDE}"

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, dropout=0.0):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
    
    def forward(self, x):
        shortcut = x
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        if self.proj is not None:
            shortcut = self.proj(shortcut)
        return F.relu(x + shortcut)

class TransformerBlock(nn.Module):
    """Attn + FFN with residuals (EQT-style)"""
    def __init__(self, d_model, num_heads=2, dropout=0.1, ff_mult=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, return_attn=False):
        attn_out, attn_w = self.mha(x, x, x, need_weights=return_attn)
        x = self.ln1(x + attn_out)

        ff = self.ffn(x)
        x = self.ln2(x + ff)

        if return_attn:
            return x, attn_w
        return x
    
class GNSSFeatMapEncoder(nn.Module):
    """
    GNSS 1Hz ENU displacement encoder (featmap only)
    Input : (B, T, 3)
    Output : (B, T', F)

    Downsample options:
        - downsample="none" (default): T' = T
        - downsample="pool":           T' ~ floor(T/2)
        - downsample="conv":           T' ~ floor((T+1)/2) (stride-2 conv)
    
    Conditional downsample:
        - auto_downsample=True: apply downsample only if T > threshold_T
    """
    def __init__(
            self,
            in_ch=3,
            base_filters=32,
            res_depth=2,
            lstm_units=64,
            num_heads=2,
            dropout=0.15,
            k=7,
            ff_mult=2,
            downsample="none",      # "none" | "pool" | "conv"
            auto_downsample=False,
            threshold_T=512,
    ):
        super().__init__()
        self.auto_downsample = auto_downsample
        self.threshold_T = threshold_T

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_filters, k, padding=k//2, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
        )
        self.res = nn.Sequential(*[
            ResBlock1D(base_filters, base_filters, k=k, dropout=dropout)
            for _ in range(res_depth)
        ])

        #Downsample module (default: Identity)
        if downsample == "none":
            self.ds = nn.Identity()
        elif downsample == "pool":
            self.ds = nn.MaxPool1d(kernel_size=2, stride=2)
        elif downsample == "conv":
            self.ds = nn.Sequential(
                nn.Conv1d(base_filters, base_filters, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"downsample must be one of ['none', 'pool', 'conv'], got: {downsample}")
        
        self.bilstm = nn.LSTM(
            input_size=base_filters,
            hidden_size=lstm_units,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.align = nn.Identity()

        self.tx = TransformerBlock(
            d_model=2 * lstm_units, num_heads=num_heads, dropout=dropout, ff_mult=ff_mult
        )

    def forward(self, x, return_attn=False):
        #x: (B, T, 3)
        B, T, C = x.shape

        x = x.transpose(1, 2)   # (B, 3, T)
        x = self.stem(x)        # (B, F, T)
        x = self.res(x)         # (B, F, T)

        #downsample (optional)
        if (self.auto_downsample) and (T > self.threshold_T):
            x = self.ds(x)      # (B, F, T')    (Identity/pool/stride-conv)
        
        #BiLSTM + align
        x = x.transpose(1, 2)   # (B, T', F)
        x, _ = self.bilstm(x)   # (B, T', 2*lstm_units)

        x = x.transpose(1, 2)   # (B, 2*lstm_units, T')
        x = self.align(x)       # (B, 2*lstm_units, T')
        x = x.transpose(1, 2)   # (B, T', 2*lstm_units)

        #Transformer (EQT-style)
        if return_attn:
            x, w = self.tx(x, return_attn=True)
            return x, w
        return self.tx(x, return_attn=False)
    
if __name__ == "__main__":
    npz_dir = GNSS_NPZ_DIR   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ==== 인코더 생성 ====
    enc = GNSSFeatMapEncoder(downsample="none").to(device)
    enc.eval()

    print("Start feature extraction...\n")

    with torch.no_grad():
        for fp in sorted(npz_dir.glob("*_600s_300s.npz")):

            # -------- station 이름 추출 --------
            station = fp.name.split("_")[0]

            print(f"Station: {station}")
            print(f"File   : {fp.name}")

            # -------- 파일 로드 --------
            data = np.load(fp)

            X = data["X"]              # (N,600,3)
            start_sec = data.get("start_sec")
            fs = data.get("fs") 

            print("Input shape :", X.shape)

            if start_sec is not None:
                print("Start times (sec, first 5):", start_sec[:5])
            else:
                print("No 'start_sec' metadata found.")

            if fs is not None:
                fs_val = fs.item() if hasattr(fs, "item") else fs
                print("Sampling rate:", fs_val, "Hz")

            x = torch.from_numpy(X).float().to(device, non_blocking=True)

            feat = enc(x)

            print("Feature shape:", feat.shape)
            print("-" * 50)

    print("\nDone.")