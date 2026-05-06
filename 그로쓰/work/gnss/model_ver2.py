import torch
import torch.nn as nn
import torch.nn.functional as F
from work.gnss.encoder_ver2 import GNSSEncoder

class Decoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, output_dim=1, dropout=0.3):
        super().__init__()
        self.attn_pool = nn.Linear(input_dim, 1)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.reg = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):           # (B, T, D)
        w   = torch.softmax(self.attn_pool(x).squeeze(-1), dim=1).unsqueeze(-1)
        ctx = (x * w).sum(dim=1)    # (B, D)
        return self.reg(self.mlp(ctx))
	
class GNSSModel(nn.Module):
    """
    Station-level GNSS PGV model

    Input:
        x: (B, W, T, 3)
            B = number of stations in a batch
            W = number of windows per station
            T = time length of each window
            3 = ENU channels

    Flow:
        1) flatten windows        : (B, W, T, 3) -> (B*W, T, 3)
        2) window encoder         : (B*W, T, 3) -> (B*W, T', F)
        3) time pooling           : (B*W, T', F) -> (B*W, F)
        4) restore station groups : (B*W, F) -> (B, W, F)
        5) window-level LSTM      : (B, W, F) -> (B, W, H)
        6) decoder(attention)     : (B, W, H) -> (B, 1)
    """
    def __init__(
        self,
        encoder_kwargs=None,
        station_lstm_hidden=128,
        station_lstm_layers=1,
        decoder_hidden_dim=64,
        output_dim=1,
        use_last_hidden=False,
    ):
        super().__init__()

        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.encoder = GNSSEncoder(**encoder_kwargs)

        # encoder output dim = 2 * lstm_units
        encoder_lstm_units = encoder_kwargs.get("lstm_units", 64)
        self.window_feat_dim = 2 * encoder_lstm_units

        self.window_lstm = nn.LSTM(
            input_size=self.window_feat_dim,
            hidden_size=station_lstm_hidden,
            num_layers=station_lstm_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.use_last_hidden = use_last_hidden

        if self.use_last_hidden:
            self.final_head = nn.Linear(station_lstm_hidden, output_dim)
        else:
            self.decoder = Decoder(
                input_dim=station_lstm_hidden,
                hidden_dim=decoder_hidden_dim,
                output_dim=output_dim,
                dropout=0.3,
            )

    def forward(self, x):
        """
        x: (B, W, T, 3)
        """
        B, W, T, C = x.shape

        # 1) flatten windows
        x = x.reshape(B * W, T, C)                         # (B*W, T, 3)

        # 2) window-level encoding
        x = self.encoder(x)                                # (B*W, T', F)

        # 3) time pooling inside each window
        x = x.mean(dim=1)                                  # (B*W, F)

        # 4) restore station grouping
        x = x.reshape(B, W, self.window_feat_dim)          # (B, W, F)

        # 5) model window order
        x, _ = self.window_lstm(x)                         # (B, W, H)

        if self.use_last_hidden:
            x = x[:, -1, :]                                # (B, H)
            x = self.final_head(x)                         # (B, 1)
        else:
            x = self.decoder(x)                            # (B, 1)

        return x