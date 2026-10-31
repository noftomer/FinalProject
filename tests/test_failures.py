"""Behavior in failure situations: bad input, missing/corrupt model, external service failure."""
import re
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

import main
from src.config import Config
from src.data import load_prices
from src.dataset import last_window, prepare
from src.features import FEATURES, MARKET_FEATURES, add_features
from src.models import build_model


def _fake_yfinance(monkeypatch, download):
    """Replace the ``yfinance`` module that ``load_prices`` imports lazily."""
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=download))


def _no_network(*args, **kwargs):
    raise ConnectionError("network is unreachable")


def _cache_prices(data_dir, aapl) -> None:
    """Write AAPL, SPY and VIX price caches (the VIX/SPY ones are copies of AAPL) into ``data_dir``."""
    for name in ("AAPL", "SPY", "VIX"):
        aapl.to_csv(data_dir / f"{name}_2010-01-01_latest.csv")


def _tiny_checkpoint(cfg: Config) -> None:
    """Save an untrained model in the layout ``main.train_and_evaluate`` writes."""
    n = len(FEATURES) + len(MARKET_FEATURES)
    scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(50, n)))
    model = build_model(n, cfg.hidden_size, cfg.num_layers, cfg.dropout)
    out = main.run_dir(cfg)
    out.mkdir(parents=True)
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(), "x_scaler": scaler}, out / "model.pt")


# --- invalid input ---------------------------------------------------------------------------

def test_unknown_ticker_gives_clear_error(tmp_path, monkeypatch):
    _fake_yfinance(monkeypatch, lambda *a, **k: pd.DataFrame())  # what Yahoo returns for an unknown symbol
    with pytest.raises(RuntimeError, match="No data returned for ticker 'ZZZZ'"):
        load_prices("ZZZZ", "2010-01-01", None, tmp_path)
    assert not list(tmp_path.glob("*.csv")), "a failed download must not leave a cache file behind"


def test_unknown_target_rejected(synthetic):
    with pytest.raises(ValueError, match="Unknown target"):
        add_features(synthetic, "price")


@pytest.mark.parametrize("argv", [["main.py", "train", "--target", "bogus"], ["main.py", "explode"]])
def test_cli_rejects_invalid_arguments(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code == 2


@pytest.mark.parametrize("n_rows", [10, 45, 60])
def test_too_short_history_gives_clear_error(aapl, n_rows):
    with pytest.raises(ValueError, match="Not enough"):
        prepare(add_features(aapl.iloc[:n_rows], "return"), 40, 0.15, 0.15)


def test_constant_prices_give_clear_error(aapl):
    flat = aapl.iloc[:400].copy()
    flat[["Open", "High", "Low", "Close"]] = 100.0   # RSI is undefined, so every row is dropped
    with pytest.raises(ValueError, match="Not enough"):
        prepare(add_features(flat, "return"), 40, 0.15, 0.15)


def test_last_window_needs_full_window(synthetic):
    df = add_features(synthetic, "return")
    scaler = prepare(df, 40, 0.15, 0.15).x_scaler
    with pytest.raises(ValueError, match="needs a window"):
        last_window(df.iloc[:30], 40, scaler)


def test_missing_values_do_not_reach_the_model(synthetic):
    synthetic.loc[synthetic.index[100:103], "Close"] = np.nan
    data = prepare(add_features(synthetic.dropna(), "return"), 40, 0.15, 0.15)
    for split in (data.train, data.val, data.test):
        assert np.isfinite(split.X).all() and np.isfinite(split.y).all()


# --- external service failure ----------------------------------------------------------------

def test_network_failure_is_reported_as_runtime_error(tmp_path, monkeypatch):
    _fake_yfinance(monkeypatch, _no_network)
    with pytest.raises(RuntimeError, match="Could not download data.*network is unreachable"):
        load_prices("AAPL", "2010-01-01", None, tmp_path)


def test_cache_is_used_without_network(tmp_path, monkeypatch, aapl):
    aapl.to_csv(tmp_path / "AAPL_2010-01-01_latest.csv")
    _fake_yfinance(monkeypatch, _no_network)
    assert len(load_prices("AAPL", "2010-01-01", None, tmp_path)) == len(aapl)


def test_predict_falls_back_to_cache_when_download_fails(tmp_path, monkeypatch, aapl, capsys):
    cfg = Config(output_dir=tmp_path, data_dir=tmp_path)
    _tiny_checkpoint(cfg)
    _cache_prices(tmp_path, aapl)
    _fake_yfinance(monkeypatch, _no_network)
    main.predict_next_day(cfg)
    out = capsys.readouterr().out
    assert "Warning" in out and "cached data" in out and "Predicted close" in out


def test_predict_without_network_and_without_cache_fails_clearly(tmp_path, monkeypatch):
    cfg = Config(output_dir=tmp_path, data_dir=tmp_path)
    _tiny_checkpoint(cfg)
    _fake_yfinance(monkeypatch, _no_network)
    with pytest.raises(RuntimeError, match="Could not download data"):
        main.predict_next_day(cfg)


# --- model failures --------------------------------------------------------------------------

def test_predict_without_model_exits_with_instructions(tmp_path):
    with pytest.raises(SystemExit, match="Run `python main.py train` first"):
        main.predict_next_day(Config(output_dir=tmp_path))


def test_predict_with_corrupt_checkpoint_exits_cleanly(tmp_path):
    cfg = Config(output_dir=tmp_path)
    out = main.run_dir(cfg)
    out.mkdir(parents=True)
    (out / "model.pt").write_bytes(b"this is not a checkpoint")
    with pytest.raises(SystemExit, match="Could not load the model"):
        main.predict_next_day(cfg)


def test_untrained_model_still_prints_a_finite_prediction(tmp_path, monkeypatch, aapl, capsys):
    """A garbage model must still give a number the CLI can print (no NaN / crash)."""
    cfg = Config(output_dir=tmp_path, data_dir=tmp_path)
    _tiny_checkpoint(cfg)
    _fake_yfinance(monkeypatch, lambda *a, **k: aapl)   # a "successful" download
    main.predict_next_day(cfg)
    assert not re.search(r"nan|inf", capsys.readouterr().out.lower())
