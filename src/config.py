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
    data_dir: Path = ROOT / "data"   # where raw/processed data files are cached
    output_dir: Path = ROOT / "outputs"  # where models, plots, and metrics are written

    # Windowing / splits (chronological, no shuffling across time)
    window: int = 30                 # trading days of history per sample
    val_ratio: float = 0.15          # fraction of data reserved for validation
    test_ratio: float = 0.15         # fraction of data reserved for testing

    # Model
    model: str = "lstm"              # the only model; used to name the output directory
    hidden_size: int = 64            # number of hidden units per LSTM layer
    num_layers: int = 2              # number of stacked LSTM layers
    dropout: float = 0.2             # dropout probability applied between layers for regularization

    # Training
    epochs: int = 100                # maximum number of training passes over the dataset
    batch_size: int = 64             # number of samples processed per gradient update
    lr: float = 1e-3                 # optimizer learning rate
    weight_decay: float = 1e-5       # L2 regularization strength for the optimizer
    patience: int = 15               # early stopping on validation loss
    seed: int = 42                   # random seed for reproducible runs

    # Input features fed to the model (one column per name, computed from raw OHLCV data)
    # Uses field(default_factory=...) because a list is mutable and can't be a direct default
    features: list[str] = field(default_factory=lambda: [
        "log_ret", "log_ret_5", "log_ret_20",
        "sma_10_ratio", "sma_50_ratio",
        "ema_12_ratio", "ema_26_ratio",
        "rsi_14", "macd", "macd_signal",
        "volatility_20", "bb_position",
        "hl_range", "oc_change", "volume_change",
        "day_of_week",
    ])

    # Serialize the config to a plain dict (e.g. for logging or saving to JSON)
    def to_dict(self) -> dict:
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["output_dir"] = str(self.output_dir)
        return d
