"""Shared fixtures: tests run offline, on the cached AAPL CSV or on synthetic prices."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / "data" / "AAPL_2010-01-01_latest.csv"


@pytest.fixture(scope="session")
def aapl() -> pd.DataFrame:
    """Cached AAPL prices (skips the test when the cache is missing)."""
    if not CACHE.exists():
        pytest.skip("AAPL cache not found; run `python main.py train` once")
    return pd.read_csv(CACHE, index_col=0, parse_dates=True)


@pytest.fixture
def synthetic() -> pd.DataFrame:
    """A random-walk OHLCV frame with 600 business days."""
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600)))
    idx = pd.bdate_range("2020-01-01", periods=600)
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close,
                         "Volume": rng.integers(1_000_000, 5_000_000, 600)}, index=idx)
