"""Experiment configuration. Every hyper-parameter lives here so runs are reproducible."""
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # Data
    ticker: str = "AAPL"             # Apple stock by default
    start: str = "2010-01-01"        # first date of historical data to download (YYYY-MM-DD)
    end: str | None = None           # None = up to today
    data_dir: Path = ROOT / "data"   # where raw price data is cached
    output_dir: Path = ROOT / "outputs"  # where the model, plots, and metrics are written

    # Chronological splits (no shuffling across time)
    val_ratio: float = 0.15          # fraction of data reserved for validation
    test_ratio: float = 0.15         # fraction of data reserved for testing

    # What to predict: "return" = next-day log return; "volatility" = log realized volatility of the next 5 days
    target: str = "return"

    # Model: LSTM (PyTorch) over the last `seq_len` days of features (see features.py)
    model: str = "lstm"              # used to name the output directory
    seq_len: int = 40                # length of the input window (trading days)
    hidden_size: int = 32            # LSTM hidden units per layer (small: daily returns are mostly noise)
    num_layers: int = 1              # stacked LSTM layers
    dropout: float = 0.2             # dropout before the output layer (and between layers if stacked)

    # Training
    lr: float = 3e-4                 # Adam learning rate
    weight_decay: float = 1e-2       # L2 penalty
    huber_delta: float = 1.0         # Huber loss threshold (in std units); robust to extreme days
    batch_size: int = 64
    epochs: int = 100                # upper bound; early stopping usually ends sooner
    patience: int = 10               # stop after this many epochs without validation improvement
    seed: int = 42                   # for reproducible weight initialization and batch order

    def to_dict(self) -> dict:
        """Serialize the config to a plain dict (e.g. for logging or saving to JSON)."""
        d = asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["output_dir"] = str(self.output_dir)
        return d
