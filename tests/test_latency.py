"""Response time of the prediction path. Run with ``pytest tests/test_latency.py -s`` to see the numbers.

The bounds are deliberately loose (they catch a regression of an order of magnitude, not noise).
"""
import statistics
import sys
import time
import types

import pytest
import torch

import main
from src.config import Config
from src.dataset import last_window
from src.features import add_features
from src.models import build_model

CHECKPOINTS = {"return": "AAPL_lstm", "volatility": "AAPL_lstm_vol"}


def _checkpoint(target: str):
    path = main.Config().output_dir / CHECKPOINTS[target] / "model.pt"
    if not path.exists():
        pytest.skip(f"no trained model at {path}")
    return torch.load(path, weights_only=False)


@pytest.mark.parametrize("target", ["return", "volatility"])
def test_single_window_inference_latency(aapl, target):
    """Forward pass for one window (model only), median over 200 calls."""
    ckpt = _checkpoint(target)
    saved = ckpt["config"]
    model = build_model(ckpt["x_scaler"].n_features_in_, saved["hidden_size"], saved["num_layers"], saved["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    X = last_window(add_features(aapl, target), saved["seq_len"], ckpt["x_scaler"])
    model.predict(X)  # warm-up
    times = []
    for _ in range(200):
        t0 = time.perf_counter()
        model.predict(X)
        times.append((time.perf_counter() - t0) * 1000)
    median, p95 = statistics.median(times), sorted(times)[int(0.95 * len(times))]
    print(f"\n[{target}] model inference: median {median:.2f} ms, p95 {p95:.2f} ms")
    assert median < 100


@pytest.mark.parametrize("target", ["return", "volatility"])
def test_end_to_end_predict_latency_offline(monkeypatch, aapl, capsys, target):
    """``predict`` as the user runs it (load checkpoint, features, scaling, forward pass, print).

    The Yahoo download is replaced by the cached frame, so the network time is excluded.
    """
    _checkpoint(target)
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=lambda *a, **k: aapl))
    cfg = Config(target=target, data_dir=main.Config().data_dir / "_latency_tmp")
    t0 = time.perf_counter()
    try:
        main.predict_next_day(cfg)
    finally:
        for f in cfg.data_dir.glob("*"):
            f.unlink()
        cfg.data_dir.rmdir()
    elapsed = time.perf_counter() - t0
    capsys.readouterr()
    with capsys.disabled():
        print(f"\n[{target}] end-to-end predict (no network): {elapsed * 1000:.0f} ms")
    assert elapsed < 5
