"""LSTM that maps a window of daily returns to the next-day return."""
import numpy as np
import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM over the input window; the last hidden state feeds a linear head.

    The head predicts a standardized return. ``y_mean``/``y_std`` (set from the training targets)
    are stored with the weights, so ``predict`` returns real log returns.
    """

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_std", torch.ones(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, n_features) -> (batch,) standardized next-day return."""
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Next-day log returns for windows ``X`` shaped (n, seq_len, n_features)."""
        self.eval()
        z = self(torch.tensor(X, dtype=torch.float32, device=self.y_mean.device))
        return (z * self.y_std + self.y_mean).cpu().numpy()


def build_model(n_features: int, hidden_size: int, num_layers: int, dropout: float) -> LSTMRegressor:
    """A freshly initialized LSTMRegressor."""
    return LSTMRegressor(n_features, hidden_size, num_layers, dropout)
