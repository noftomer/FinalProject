"""Experiment configuration. Every hyper-parameter lives here so runs are reproducible."""
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Data
    ticker: str = "^GSPC"            # S&P 500 index by default
    start: str = "2010-01-01"        # first date of historical data to download (YYYY-MM-DD)
    end: str | None = None           # None = up to today
    data_dir: Path = ROOT / "data"   # where raw price data is cached
    output_dir: Path = ROOT / "outputs"  # where the model, plots, and metrics are written

    # Chronological splits (no shuffling across time)
    val_ratio: float = 0.15          # fraction of data reserved for validation
    test_ratio: float = 0.15         # fraction of data reserved for testing

    # Model: Ridge regression on the last `n_lags` daily log returns
    model: str = "ridge"             # used to name the output directory
    n_lags: int = 5                  # how many past daily returns the model sees
    alphas: list[float] = field(default_factory=lambda: [0.1, 1, 10, 100, 1000, 10000, 100000])
    # regularization strengths to try; the best one on validation data is kept

    def to_dict(self) -> dict:
        """Serialize the config to a plain dict (e.g. for logging or saving to JSON)."""
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["output_dir"] = str(self.output_dir)
        return d
