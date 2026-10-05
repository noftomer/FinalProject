"""Metrics, a direction significance test, and plots."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BLUE, LIGHT_BLUE, GRAY = "#1f5fbf", "#7fb2ff", "#8a8a8a"


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, close: np.ndarray) -> dict:
    """Metrics on returns and on reconstructed next-day prices."""
    price_true = close * np.exp(y_true)
    price_pred = close * np.exp(y_pred)
    nonzero = y_true != 0
    has_direction = np.any(y_pred != 0)  # a zero forecast makes no directional call
    return {
        "return_rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "return_mae": float(np.mean(np.abs(y_true - y_pred))),
        "price_rmse": float(np.sqrt(np.mean((price_true - price_pred) ** 2))),
        "price_mape_%": float(np.mean(np.abs(price_true - price_pred) / price_true) * 100),
        "directional_accuracy_%": float(np.mean(np.sign(y_true[nonzero]) == np.sign(y_pred[nonzero])) * 100)
                                  if has_direction else float("nan"),
    }


def baseline_metrics(y_true: np.ndarray, close: np.ndarray, y_train: np.ndarray) -> dict:
    """Metrics of trivial forecasters; the model must beat these to claim it learned anything."""
    n = len(y_true)
    forecasts = {
        "ZERO (tomorrow = today)": np.zeros(n),
        "TRAIN_MEAN": np.full(n, float(y_train.mean())),
        "ALWAYS_UP": np.full(n, 1e-12),
    }
    return {name: regression_metrics(y_true, pred, close) for name, pred in forecasts.items()}


def volatility_metrics(y_true: np.ndarray, y_pred: np.ndarray, trailing: np.ndarray, y_train: np.ndarray) -> dict:
    """Metrics (log-volatility space) for the model and baselines.

    ``r2_vs_train_mean`` > 0 means the forecast beats predicting the training mean; the
    "TRAILING_5D" baseline assumes the next 5 days are as volatile as the last 5.
    """
    def row(pred: np.ndarray) -> dict:
        sse = np.sum((y_true - pred) ** 2)
        sst = np.sum((y_true - y_train.mean()) ** 2)
        return {"rmse": float(np.sqrt(np.mean((y_true - pred) ** 2))),
                "mae": float(np.mean(np.abs(y_true - pred))),
                "r2_vs_train_mean": float(1 - sse / sst),
                "corr": float(np.corrcoef(y_true, pred)[0, 1]) if np.std(pred) > 0 else float("nan")}
    return {"MODEL": row(y_pred), "TRAIN_MEAN": row(np.full(len(y_true), y_train.mean())),
            "TRAILING_5D": row(trailing)}


def direction_significance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binomial test: is directional accuracy better than always guessing the majority class?"""
    nz = y_true != 0
    hits = int(np.sum(np.sign(y_true[nz]) == np.sign(y_pred[nz])))
    n = int(nz.sum())
    majority = max(np.mean(y_true[nz] > 0), np.mean(y_true[nz] < 0))
    p = stats.binomtest(hits, n, majority, alternative="greater").pvalue
    return {"hits": hits, "n": n, "majority_rate_%": float(majority * 100), "p_value": float(p)}


def plot_all(out_dir: Path, dates: pd.DatetimeIndex, y_true: np.ndarray,
             y_pred: np.ndarray, close: np.ndarray, history: dict, title: str) -> None:
    """Save the loss curves, predicted-vs-actual price, and return scatter plots to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(out_dir, history, title)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, close * np.exp(y_true), color=GRAY, lw=1.2, label="actual next-day close")
    ax.plot(dates, close * np.exp(y_pred), color=BLUE, lw=1.0, label="predicted next-day close")
    ax.set(title=f"Test set: predicted vs actual price — {title}", ylabel="price",
           xlabel="forecast date (value shown = following trading day's close)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "price_prediction.png", dpi=150); plt.close(fig)

    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(11, 5.5))
    ax.scatter(y_true * 100, y_pred * 100, s=6, alpha=0.5, color=BLUE)
    lim = np.abs(y_true).max() * 100
    ax.plot([-lim, lim], [-lim, lim], color=GRAY, ls="--", lw=1)
    ax.axhline(0, color=GRAY, lw=0.5); ax.axvline(0, color=GRAY, lw=0.5)
    ax.set(title="Predicted vs actual return (%)", xlabel="actual", ylabel="predicted")

    # Zoomed view: y-axis autoscaled with many decimals, so near-zero predictions are visibly not exactly 0
    ax_zoom.scatter(y_true * 100, y_pred * 100, s=6, alpha=0.5, color=BLUE)
    ax_zoom.axhline(0, color=GRAY, lw=0.5)
    ax_zoom.yaxis.set_major_formatter(plt.FormatStrFormatter("%.5f"))
    ax_zoom.set(title="Zoom on predictions (y-axis autoscaled)", xlabel="actual", ylabel="predicted (%)")
    ax_zoom.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "return_scatter.png", dpi=150); plt.close(fig)


def plot_volatility(out_dir: Path, dates: pd.DatetimeIndex, y_true: np.ndarray,
                    y_pred: np.ndarray, history: dict, title: str) -> None:
    """Save the loss curves and predicted-vs-actual 5-day volatility (daily %) to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss(out_dir, history, title)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, np.exp(y_true) * 100, color=GRAY, lw=1.2, label="actual next-5-day volatility")
    ax.plot(dates, np.exp(y_pred) * 100, color=BLUE, lw=1.0, label="predicted")
    ax.set(title=f"Test set: 5-day realized volatility — {title}", ylabel="daily volatility (%)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "volatility_prediction.png", dpi=150); plt.close(fig)


def plot_loss(out_dir: Path, history: dict, title: str) -> None:
    """Save the train/validation loss curves to ``out_dir/loss_curve.png``."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=BLUE, label="train loss")
    ax.plot(epochs, history["val_loss"], color=LIGHT_BLUE, label="validation loss")
    ax.axvline(history["best_epoch"], color=GRAY, ls="--", lw=1, label="best epoch")
    if "baseline_val_loss" in history:
        ax.axhline(history["baseline_val_loss"], color=GRAY, ls=":", lw=1, label="baseline (predict train mean)")
    ax.set(title=f"Training and validation loss (MSE) — {title}", xlabel="epoch", ylabel="loss")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150); plt.close(fig)
