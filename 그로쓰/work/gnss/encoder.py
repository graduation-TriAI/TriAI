import torch
import torch.nn as nn
import torch.nn.functional as F
from shared.paths import GNSS_TOHOKU_PROC
import numpy as np

GNSS_NPZ = GNSS_TOHOKU_PROC / "tohoku_gnss_pgv_dataset_25km_seq.npz"

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
    """Attn + FFN with residual connections."""
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
    Window-level GNSS encoder

    Input : (B, T, 3)
    Output: (B, T', F)

    B = batch of windows
    T = time length per window
    F = feature dimension (= 2 * lstm_units)
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
        self.downsample_mode = downsample

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
        """
        x: (B, T, 3)
        """
        _, T, _ = x.shape

        x = x.transpose(1, 2)   # (B, 3, T)
        x = self.stem(x)        # (B, base_filters, T)
        x = self.res(x)         # (B, base_filters, T)

        #downsample 
        if self.downsample_mode != "none":
            if self.auto_downsample:
                if T > self.threshold_T:
                    x = self.ds(x)
            else:
                x = self.ds(x)
        
        #BiLSTM + align
        x = x.transpose(1, 2)   # (B, T', base_filters)
        x, _ = self.bilstm(x)   # (B, T', 2*lstm_units)

        x = x.transpose(1, 2)   # (B, 2*lstm_units, T')
        x = self.align(x)       
        x = x.transpose(1, 2)   # (B, T', 2*lstm_units)

        if return_attn:
            x, w = self.tx(x, return_attn=True)
            return x, w
        return self.tx(x, return_attn=False)
    
if __name__ == "__main__":
    npz_path = GNSS_NPZ

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    enc = GNSSFeatMapEncoder(downsample="none").to(device)
    enc.eval()

    print("Load NPZ:", npz_path)
    data = np.load(npz_path)

    X = data["X"]   # expected: (N_station, W, T, 3)
    print("Input shape:", X.shape)

    # encoder.py에서는 윈도우 단위 encoder만 확인
    # station/window 축을 펼쳐서 윈도우 배치처럼 넣어봄
    N_station, W, T, C = X.shape
    x = torch.from_numpy(X).float().to(device)
    x = x.reshape(N_station * W, T, C)

    with torch.no_grad():
        feat = enc(x)

    print("Encoded feature map shape:", feat.shape)  # (N_station*W, T', F)
    print("Done.")