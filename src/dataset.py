"""Chronological train/validation/test split of return windows, scaled with training data only."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import StandardScaler

from src.features import FEATURES


@dataclass
class Split:
    X: np.ndarray            # (n, seq_len, n_features) windows of scaled features
    y: np.ndarray            # (n,) next-day log return
    dates: pd.DatetimeIndex  # date of the last observed day (prediction made at its close)
    close: np.ndarray        # close price on that date (to rebuild predicted price)


@dataclass
class Prepared:
    train: Split
    val: Split
    test: Split
    x_scaler: StandardScaler  # fitted on the training rows


def _windows(x: np.ndarray, seq_len: int) -> np.ndarray:
    """All consecutive windows of ``seq_len`` rows, shaped (n - seq_len + 1, seq_len, n_features)."""
    return sliding_window_view(x, seq_len, axis=0).transpose(0, 2, 1)


def prepare(df: pd.DataFrame, seq_len: int, val_ratio: float, test_ratio: float) -> Prepared:
    """Split labeled windows chronologically into train/val/test; the scaler never sees val or test rows.

    A window may reach back into an earlier split (past data only), but its label never does.
    """
    df = df.dropna(subset=FEATURES)
    ends = df.iloc[seq_len - 1:]                       # rows where a full window ends
    ends = ends[ends["target"].notna()]                # drop the latest day (no label yet)
    n = len(ends)
    n_test, n_val = int(n * test_ratio), int(n * val_ratio)
    n_train = n - n_val - n_test

    x = df[FEATURES].values
    x_scaler = StandardScaler().fit(x[:seq_len - 1 + n_train])
    X = _windows(x_scaler.transform(x), seq_len)[:n]

    def split(lo: int, hi: int) -> Split:
        rows = ends.iloc[lo:hi]
        return Split(X[lo:hi], rows["target"].values, rows.index, rows["Close"].values)

    return Prepared(split(0, n_train), split(n_train, n_train + n_val), split(n_train + n_val, n), x_scaler)


def last_window(df: pd.DataFrame, seq_len: int, x_scaler: StandardScaler) -> np.ndarray:
    """The most recent ``seq_len`` rows (which have no label yet), scaled, shaped (1, seq_len, n_features)."""
    x = df.dropna(subset=FEATURES)[FEATURES].values[-seq_len:]
    return x_scaler.transform(x)[None]
