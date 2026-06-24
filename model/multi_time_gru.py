"""Multi-window GRU with attention fusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pipeline.config import GRU_HIDDEN_SIZE, TIME_WINDOWS, N_PRED_TARGETS


class FusedGRULinear(nn.Module):
    def __init__(self, dim: int, layers: list):
        super().__init__()
        ops = []
        for d in layers:
            ops.append(nn.Linear(dim, d))
            ops.append(nn.ReLU())
            dim = d
        self.fc = nn.Sequential(*ops)

    def forward(self, x):
        return self.fc(x)


class MultiTimeAttention(nn.Module):
    """Compute attention weights over n_windows."""
    def __init__(self, hidden_size: int, n_windows: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, n_windows)

    def forward(self, feat):
        return F.softmax(self.attn(feat), dim=-1)


class MultiTimeGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = GRU_HIDDEN_SIZE,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_windows = len(TIME_WINDOWS)
        self.hidden_size = hidden_size

        # One GRU per window, each takes raw features as input
        self.grus = nn.ModuleList([
            nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False) for _ in TIME_WINDOWS
        ])

        # Fusion pipeline
        self.mlp_fuse = FusedGRULinear(hidden_size * self.n_windows,
                                       [hidden_size * 2, hidden_size * 4, hidden_size * self.n_windows])
        self.att = MultiTimeAttention(hidden_size * self.n_windows, self.n_windows)
        self.fusion = FusedGRULinear(hidden_size,
                                      [hidden_size * 4, hidden_size * 2, hidden_size])
        self.lin_out = nn.Linear(hidden_size, N_PRED_TARGETS)

    def forward(self, x_list: list) -> tuple:
        """
        x_list: list of tensors, each (batch, window_size, input_size).
        Returns: pred (B, N_PRED_TARGETS), feat (B, hidden), attention (B, n_windows).
        """
        batch = x_list[0].size(0)
        hidden_list = []

        # Multi-window GRU: each GRU processes its own time window
        for gru, x in zip(self.grus, x_list):
            out, _ = gru(x)                      # out: (B, W_i, hidden)
            hidden_list.append(out[:, -1, :])     # last hidden: (B, hidden)

        # Stack across windows: (B, n_windows, hidden)
        last_hs = torch.stack(hidden_list, dim=1)

        # Flatten for MLP
        feat_flat = last_hs.view(batch, -1)     # (B, hidden * n_windows)

        # MLP fusion + attention
        feat_mlp = self.mlp_fuse(feat_flat)     # (B, hidden * n_windows)
        attention = self.att(feat_mlp)               # (B, n_windows)

        # Weighted fusion
        fused = torch.sum(last_hs * attention.unsqueeze(-1), dim=1)  # (B, hidden * n_windows)
        fused_latent = self.fusion(fused)       # (B, hidden)

        pred = self.lin_out(fused_latent)        # (B, N_HORIZONS)

        return pred, feat_flat, attention
