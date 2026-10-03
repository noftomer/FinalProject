"""Train the LSTM with Adam, early-stopping on validation loss."""
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.models import LSTMRegressor, build_model


def fit(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
        cfg: Config) -> tuple[LSTMRegressor, dict]:
    """Return the model from the epoch with the lowest validation MSE, plus the training history."""
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(X_train.shape[-1], cfg.hidden_size, cfg.num_layers, cfg.dropout).to(device)
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    model.y_mean.fill_(y_mean)
    model.y_std.fill_(y_std)

    def tensors(X, y):
        return (torch.tensor(X, dtype=torch.float32, device=device),
                torch.as_tensor((y - y_mean) / y_std, dtype=torch.float32, device=device))

    Xt, yt = tensors(X_train, y_train)
    Xv, yv = tensors(X_val, y_val)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(cfg.seed))
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_state, best_val, best_epoch = None, float("inf"), 0
    history = {"train_loss": [], "val_loss": []}
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xv), yv).item()
        history["train_loss"].append(total / len(Xt))
        history["val_loss"].append(val)
        print(f"epoch {epoch:>3} | train loss {history['train_loss'][-1]:.4f} | val loss {val:.4f}")

        if val < best_val:
            best_state, best_val, best_epoch = copy.deepcopy(model.state_dict()), val, epoch
        elif epoch - best_epoch >= cfg.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch} (val loss {best_val:.4f})")
    history["best_epoch"] = best_epoch
    return model.eval(), history
