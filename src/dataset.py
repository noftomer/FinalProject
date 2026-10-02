"""Chronological train/validation/test split with a scaler fitted on training data only."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Split:
    X: np.ndarray            # (n, n_features) lagged returns, scaled
    y: np.ndarray            # (n,) next-day log return
    dates: pd.DatetimeIndex  # date of the last observed day (prediction made at its close)
    close: np.ndarray        # close price on that date (to rebuild predicted price)


@dataclass
class Prepared:
    train: Split
    val: Split
    test: Split
    x_scaler: StandardScaler  # fitted on the training split


def _split(df: pd.DataFrame, features: list[str], x_scaler: StandardScaler) -> Split:
    """Turn a slice of the feature table into a Split."""
    return Split(x_scaler.transform(df[features].values), df["target"].values,
                 df.index, df["Close"].values)


def prepare(df: pd.DataFrame, features: list[str], val_ratio: float, test_ratio: float) -> Prepared:
    """Split labeled rows chronologically into train/val/test; the scaler never sees val or test data."""
    labeled = df.dropna(subset=features + ["target"])
    n = len(labeled)
    n_test, n_val = int(n * test_ratio), int(n * val_ratio)
    n_train = n - n_val - n_test

    x_scaler = StandardScaler().fit(labeled[features].values[:n_train])
    return Prepared(_split(labeled.iloc[:n_train], features, x_scaler),
                    _split(labeled.iloc[n_train:n_train + n_val], features, x_scaler),
                    _split(labeled.iloc[n_train + n_val:], features, x_scaler),
                    x_scaler)


def last_row(df: pd.DataFrame, features: list[str], x_scaler: StandardScaler) -> np.ndarray:
    """Features of the most recent day (which has no label yet), scaled and shaped (1, n_features)."""
    return x_scaler.transform(df.dropna(subset=features)[features].values[-1:])
