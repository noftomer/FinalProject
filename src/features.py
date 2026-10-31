"""Features: daily return, volume change, and technical indicators. Every value at row t uses data up to the close of day t.

The target at row t is the log return from close t to close t+1. The LSTM sees a window of
the last ``seq_len`` rows of these features (built in dataset.py).
"""
import numpy as np
import pandas as pd

FEATURES = ["log_ret", "vol_chg", "rsi", "macd_hist", "volatility_10", "hl_range", "sma_gap_20",
            "sma_gap_10", "sma_gap_50", "open_gap", "dow_sin", "dow_cos",
            "abs_ret", "log_parkinson", "log_vol_22"]
MARKET_FEATURES = ["spy_ret", "vix_level", "vix_chg"]   # added only when market data is supplied


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing), scaled to 0..1."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return gain / (gain + loss)


VOL_HORIZON = 5  # trading days over which realized volatility is measured


def market_features(spy: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Market-wide features indexed by date: SPY log return, log VIX level, and its daily change."""
    vix_level = np.log(vix["Close"])
    return pd.DataFrame({"spy_ret": np.log(spy["Close"] / spy["Close"].shift(1)),
                         "vix_level": vix_level, "vix_chg": vix_level.diff()})


def feature_cols(df: pd.DataFrame) -> list[str]:
    """The model's input columns: ``FEATURES``, plus ``MARKET_FEATURES`` when the frame has them."""
    return FEATURES + MARKET_FEATURES if all(c in df.columns for c in MARKET_FEATURES) else FEATURES


def add_features(df: pd.DataFrame, target: str = "return", market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add the feature columns in ``FEATURES`` and the ``target`` column to an OHLCV DataFrame.

    target="return":     next-day log return.
    target="volatility": log of realized volatility (RMS of daily log returns) over the next
                         ``VOL_HORIZON`` days; ``vol_trailing`` is the same quantity over the past days.
    market:              optional output of ``market_features``; aligned to ``df`` dates (forward-filled over
                         holidays that differ between markets) and used as extra model inputs.
    """
    out = df.copy()
    close = out["Close"]
    out["log_ret"] = np.log(close / close.shift(1))
    out["vol_chg"] = np.log1p(out["Volume"]).diff()                  # change in trading volume
    out["rsi"] = _rsi(close)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    out["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close   # price-normalized
    out["volatility_10"] = out["log_ret"].rolling(10).std()
    out["hl_range"] = (out["High"] - out["Low"]) / close             # intraday range
    out["sma_gap_20"] = close / close.rolling(20).mean() - 1         # distance from 20-day average
    out["sma_gap_10"] = close / close.rolling(10).mean() - 1
    out["sma_gap_50"] = close / close.rolling(50).mean() - 1
    out["open_gap"] = np.log(out["Open"] / close.shift(1))            # overnight gap vs previous close
    dow = 2 * np.pi * out.index.dayofweek / 5                        # day of week, cyclical
    out["dow_sin"], out["dow_cos"] = np.sin(dow), np.cos(dow)
    # Targets look ahead (negative shift) -> labels only, never features
    sq = out["log_ret"] ** 2
    out["abs_ret"] = out["log_ret"].abs()
    out["log_parkinson"] = np.log(np.log(out["High"] / out["Low"]) / (2 * np.sqrt(np.log(2))) + 1e-6)  # Parkinson daily vol
    out["log_vol_22"] = 0.5 * np.log(sq.rolling(22).mean() + 1e-12)
    if market is not None:
        out[MARKET_FEATURES] = market[MARKET_FEATURES].reindex(out.index, method="ffill")
    out["vol_trailing"] = 0.5 * np.log(sq.rolling(VOL_HORIZON).mean() + 1e-12)
    if target == "return":
        out["target"] = out["log_ret"].shift(-1)
    elif target == "volatility":
        out["target"] = 0.5 * np.log(sq.rolling(VOL_HORIZON).mean().shift(-VOL_HORIZON) + 1e-12)
    else:
        raise ValueError(f"Unknown target '{target}' (expected 'return' or 'volatility').")
    return out
