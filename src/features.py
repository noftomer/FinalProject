"""Features: the last few daily log returns. Every feature at row t uses data up to the close of day t.

The target at row t is the log return from close t to close t+1.
"""
import numpy as np
import pandas as pd


def feature_names(n_lags: int) -> list[str]:
    """Column names of the lagged-return features: ret_lag0 is today's return, ret_lag1 yesterday's, ..."""
    return [f"ret_lag{k}" for k in range(n_lags)]


def add_features(df: pd.DataFrame, n_lags: int) -> pd.DataFrame:
    """Add the lagged-return feature columns, ``log_ret``, and the next-day ``target`` column to an OHLCV DataFrame."""
    out = df.copy()
    out["log_ret"] = np.log(out["Close"] / out["Close"].shift(1))
    for k, name in enumerate(feature_names(n_lags)):
        out[name] = out["log_ret"].shift(k)
    # Target: next-day log return (shift -1 looks one day ahead -> label only)
    out["target"] = out["log_ret"].shift(-1)
    return out
