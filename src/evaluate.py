"""Metrics, baselines, a simple trading backtest, and plots."""
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


def direction_significance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Binomial test: is directional accuracy better than always guessing the majority class?"""
    nz = y_true != 0
    hits = int(np.sum(np.sign(y_true[nz]) == np.sign(y_pred[nz])))
    n = int(nz.sum())
    majority = max(np.mean(y_true[nz] > 0), np.mean(y_true[nz] < 0))
    p = stats.binomtest(hits, n, majority, alternative="greater").pvalue
    return {"hits": hits, "n": n, "majority_rate_%": float(majority * 100), "p_value": float(p)}


def baselines(y_true: np.ndarray, prev_ret: np.ndarray, train_mean: float, close: np.ndarray) -> dict:
    """Naive forecasters every model must beat to be meaningful."""
    random_walk=regression_metrics(y_true, np.zeros_like(y_true), close)
    train_mean_drift=regression_metrics(y_true, np.full_like(y_true, train_mean), close)
    momentum=regression_metrics(y_true, prev_ret, close)
    return {
        "random_walk (ret=0)": random_walk,
        "train_mean_drift": train_mean_drift,
        "momentum (ret=yesterday)": momentum
    }


def summary(r: np.ndarray) -> dict:
    """Equity growth, annualized Sharpe, and max drawdown for a return series."""
    equity = np.cumprod(1 + r)
    peak = np.maximum.accumulate(equity)
    sharpe = np.sqrt(252) * r.mean() / r.std() if r.std() > 0 else 0.0
    return {"total_return_%": float((equity[-1] - 1) * 100),
            "sharpe": float(sharpe),
            "max_drawdown_%": float(((equity / peak) - 1).min() * 100)}


def backtest(y_true: np.ndarray, y_pred: np.ndarray, cost_bps: float = 1.0) -> dict:
    """Long when predicted return > 0, flat otherwise. Includes transaction costs."""
    position = (y_pred > 0).astype(float)
    trades = np.abs(np.diff(np.concatenate([[0.0], position])))
    simple = np.expm1(y_true)
    strat = position * simple - trades * cost_bps / 1e4

    return {"strategy": summary(strat), "buy_and_hold": summary(simple),
            "exposure_%": float(position.mean() * 100), "n_trades": int(trades.sum()),
            "_equity": (np.cumprod(1 + strat), np.cumprod(1 + simple))}


def plot_all(out_dir: Path, dates: pd.DatetimeIndex, y_true: np.ndarray,
             y_pred: np.ndarray, close: np.ndarray, equity: tuple[np.ndarray, np.ndarray],
             title: str) -> None:
    """Save the predicted-vs-actual price, return scatter, and backtest equity plots to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

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

    strat_eq, bh_eq = equity
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, bh_eq, color=GRAY, label="buy & hold")
    ax.plot(dates, strat_eq, color=BLUE, label="model long/flat strategy")
    ax.set(title="Backtest equity curve (test period)", ylabel="growth of $1")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "backtest.png", dpi=150); plt.close(fig)
