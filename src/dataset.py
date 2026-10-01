"""Turn the feature table into leak-free sliding-window tensors."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class Split:
    X: np.ndarray          # (n, window, n_features)
    y: np.ndarray          # (n,) next-day log return
    dates: pd.DatetimeIndex  # date of the last observed day (prediction made at its close)
    close: np.ndarray      # close price on that date (to rebuild predicted price)


@dataclass
class Prepared:
    train: Split              # training split (windows, labels, dates, close)
    val: Split                # validation split
    test: Split               # test split
    x_scaler: StandardScaler  # fitted feature scaler (to transform new data the same way)
    y_scale: float            # target std used to rescale labels/predictions


def _windows(feats: np.ndarray, window: int, idx: np.ndarray) -> np.ndarray:
    """Build sliding windows: for each index in ``idx``, stack the ``window`` rows ending at that index."""
    return np.stack([feats[i - window + 1: i + 1] for i in idx])


def _make(feats: np.ndarray, window: int, rows: np.ndarray, labeled: pd.DataFrame, sl: slice) -> Split:
    """Assemble one Split (windows, targets, dates, close prices) for the slice ``sl`` of the data."""
    r = rows[sl]
    lab = labeled.iloc[sl]    
    windows=_windows(feats, window, r)
    target=lab["target"].values.astype(np.float32)
    indexes= lab.index
    close=lab["Close"].values
    return Split(windows,target ,
                indexes, close)


def prepare(df: pd.DataFrame, features: list[str], window: int,val_ratio: float, test_ratio: float) -> Prepared:
    """Split the feature table chronologically into train/val/test windows, fitting scalers on train only to avoid leakage."""
    df = df.dropna(subset=features)
    labeled = df.dropna(subset=["target"])   
    n = len(labeled)                         
    n_test = int(n * test_ratio)             
    n_val = int(n * val_ratio)               
    n_train = n - n_val - n_test             
    train_end = labeled.index[n_train - 1]   

    x_scaler = StandardScaler()
    x_scaler.fit(df.loc[:train_end, features].values)
    feats = x_scaler.transform(df[features].values).astype(np.float32)   
    y_scale = float(labeled["target"].iloc[:n_train].std())  

    pos = {d: i for i, d in enumerate(df.index)}      
    rows = np.array([pos[d] for d in labeled.index]) 
    valid = rows >= window - 1                        
    rows = rows[valid]                                
    labeled = labeled[valid]                          
    n_train -= int((~valid).sum())                   

    train = _make(feats, window, rows, labeled, slice(0, n_train))                    
    val = _make(feats, window, rows, labeled, slice(n_train, n_train + n_val))        
    test = _make(feats, window, rows, labeled, slice(n_train + n_val, None))          
    return Prepared(train, val, test, x_scaler, y_scale)


def last_window(df: pd.DataFrame, features: list[str], window: int,
                x_scaler: StandardScaler) -> np.ndarray:
    """Most recent window (including the latest day, which has no label yet)."""
    df = df.dropna(subset=features) 
    feats = x_scaler.transform(df[features].values[-window:]).astype(np.float32) 
    return feats[None]# adds a new axis at the front of the array


def loader(split: Split, y_scale: float, batch_size: int, shuffle: bool) -> DataLoader:
    """Wrap a Split into a torch DataLoader, converting arrays to tensors and normalizing targets by ``y_scale``."""
    ds = TensorDataset(torch.from_numpy(split.X), torch.from_numpy(split.y / y_scale))  
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)  
