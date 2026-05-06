import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, dropout=0.2):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, k, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.proj  = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        ) if in_ch != out_ch else None

    def forward(self, x):
        shortcut = self.proj(x) if self.proj else x
        out = self.drop(F.relu(self.bn1(self.conv1(x))))
        return F.relu(self.bn2(self.conv2(out)) + shortcut)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads=2, dropout=0.1, ff_mult=2):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * ff_mult, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.mha(x, x, x)
        x = self.ln1(x + attn_out)
        return self.ln2(x + self.ffn(x))


class GNSSEncoder(nn.Module):
    """
    Input : (B, T_g, 3)  — 1Hz GNSS
    Output: (B, T_g, 128)
    """
    def __init__(self, in_ch=3, base_filters=48, res_depth=3,
                 lstm_units=64, num_heads=2, dropout=0.15, k=7):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base_filters, k, padding=k//2, bias=False),
            nn.BatchNorm1d(base_filters), nn.ReLU()
        )
        self.res = nn.Sequential(*[
            ResBlock1D(base_filters, base_filters, k=k, dropout=dropout)
            for _ in range(res_depth)
        ])
        self.bilstm = nn.LSTM(base_filters, lstm_units, batch_first=True, bidirectional=True)
        self.tx = TransformerBlock(2 * lstm_units, num_heads=num_heads, dropout=dropout)

    def forward(self, x):           # (B, T_g, 3)
        x = self.stem(x.transpose(1, 2))   # (B, base_filters, T_g)
        x = self.res(x).transpose(1, 2)    # (B, T_g, base_filters)
        x, _ = self.bilstm(x)              # (B, T_g, 128)
        return self.tx(x)
