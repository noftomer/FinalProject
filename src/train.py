"""Training loop with early stopping on validation loss."""
import copy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs and enable deterministic algorithms for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device,
               optimizer: torch.optim.Optimizer | None = None) -> float:
    """Run one pass over ``loader`` and return the mean loss; trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)
    return total / count


def fit(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
        epochs: int, lr: float, weight_decay: float, patience: int,
        device: torch.device) -> dict:
    """Train with AdamW and early stopping, restore the best validation weights, and return the loss history."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    loss_fn = nn.HuberLoss(delta=1.0)  # robust to fat-tailed return outliers

    history = {"train_loss": [], "val_loss": []}
    best_loss, best_state, bad_epochs = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        tr = _run_epoch(model, train_loader, loss_fn, device, optimizer)
        va = _run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(va)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        print(f"epoch {epoch:3d} | train {tr:.4f} | val {va:.4f}")

        if va < best_loss - 1e-5:
            best_loss=va
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs=0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (best val {best_loss:.4f}).")
                break

    model.load_state_dict(best_state)
    history["best_val_loss"] = best_loss
    return history


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, y_scale: float, device: torch.device) -> np.ndarray:
    """Predict next-day log returns for windows ``X``, rescaled back to real units by ``y_scale``."""
    model.eval()
    out = model(torch.from_numpy(X).to(device)).cpu().numpy()
    return out * y_scale
