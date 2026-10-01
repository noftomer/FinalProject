"""Sequence model that maps a window of features to the next-day return."""
import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """Stacked LSTM followed by a small regression head that outputs one value per sequence."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        """Create the LSTM encoder and the LayerNorm -> Dropout -> Linear output head."""
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size), nn.Dropout(dropout),
                                  nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of windows (batch, window, n_features) to predictions (batch,) from the last time step."""
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


def build_model(n_features: int, hidden_size: int, num_layers: int, dropout: float) -> nn.Module:
    """Construct the LSTM regressor with the given architecture hyper-parameters."""
    return LSTMRegressor(n_features, hidden_size, num_layers, dropout)
