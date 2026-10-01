"""Download and cache daily OHLCV price data."""
from pathlib import Path

import pandas as pd


def load_prices(ticker: str, start: str, end: str | None, data_dir: Path,
                refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame indexed by date with Open, High, Low, Close, Volume.

    Data is fetched from Yahoo Finance once and cached as CSV so experiments
    are repeatable offline.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    safe = ticker.replace("^", "").replace("/", "_")
    cache = data_dir / f"{safe}_{start}_{end or 'latest'}.csv"

    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"No data returned for ticker '{ticker}'.")
        if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
            # flatten MultiIndex columns to just the field name (drop the ticker level)
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.to_csv(cache)

    df = df.sort_index()
    df = df[~df.index.duplicated()].dropna()
    return df
