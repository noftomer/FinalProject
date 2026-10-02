"""Next-day stock market prediction — command-line entry point.

Examples
--------
    python main.py                      # train with all defaults from src/config.py
    python main.py train --ticker ^GSPC
    python main.py predict --ticker ^GSPC
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import Config
from src.data import load_prices
from src.dataset import last_row, prepare
from src.evaluate import (backtest, baselines, direction_significance, plot_all,
                          regression_metrics)
from src.features import add_features, feature_names
from src.train import fit


def run_dir(cfg: Config) -> Path:
    """Output directory for this ticker/model combination (e.g. outputs/GSPC_ridge)."""
    return cfg.output_dir / f"{cfg.ticker.replace('^', '')}_{cfg.model}"


def train_and_evaluate(cfg: Config, refresh: bool = False) -> dict:
    """Train the model, evaluate it on the test split against baselines, and save plots, metrics and the checkpoint."""
    features = feature_names(cfg.n_lags)
    df = add_features(load_prices(cfg.ticker, cfg.start, cfg.end, cfg.data_dir, refresh), cfg.n_lags)
    data = prepare(df, features, cfg.val_ratio, cfg.test_ratio)
    print(f"[{cfg.ticker} | {cfg.model}] samples: train={len(data.train.y)} "
          f"val={len(data.val.y)} test={len(data.test.y)} | "
          f"test period {data.test.dates[0].date()} -> {data.test.dates[-1].date()}")

    model, alpha = fit(data.train.X, data.train.y, data.val.X, data.val.y, cfg.alphas)

    t = data.test
    y_pred = model.predict(t.X)
    prev_ret = df.loc[t.dates, "log_ret"].values
    bt = backtest(t.y, y_pred)
    equity = bt.pop("_equity")

    config = cfg.to_dict()
    model_metrics = regression_metrics(t.y, y_pred, t.close)
    direction_test = direction_significance(t.y, y_pred)
    baseline_metrics = baselines(t.y, prev_ret, float(data.train.y.mean()), t.close)

    results = {
        "config": config,
        "model": model_metrics,
        "direction_test": direction_test,
        "baselines": baseline_metrics,
        "backtest": bt,
        "alpha": alpha,
        "coefficients": dict(zip(features, model.coef_.tolist())),
    }

    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    plot_all(out, t.dates, t.y, y_pred, t.close, equity, f"{cfg.ticker} {cfg.model.upper()}")
    pd.DataFrame({"date": t.dates, "close": t.close, "actual_next_ret": t.y,
                  "pred_next_ret": y_pred}).to_csv(out / "test_predictions.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    joblib.dump({"model": model, "config": cfg.to_dict(), "x_scaler": data.x_scaler}, out / "model.joblib")

    print_report(results)
    print(f"\nArtifacts saved to {out}")
    return results


def print_report(r: dict) -> None:
    """Print test-set metrics, the direction significance test, and backtest results to the console."""
    rows = {"MODEL": r["model"], **r["baselines"]}
    table = pd.DataFrame(rows).T[["return_rmse", "return_mae", "price_rmse",
                                  "price_mape_%", "directional_accuracy_%"]]
    print("\n=== Test-set metrics ===")
    print(table.to_string(float_format=lambda v: f"{v:.5f}"))
    d = r["direction_test"]
    print(f"\nDirection: {d['hits']}/{d['n']} correct; majority-class rate "
          f"{d['majority_rate_%']:.1f}%; binomial p-value = {d['p_value']:.4f}")
    b = r["backtest"]
    print("\n=== Backtest (long/flat, 1 bp cost) ===")
    print(pd.DataFrame({k: b[k] for k in ("strategy", "buy_and_hold")}).T
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"exposure {b['exposure_%']:.1f}% | trades {b['n_trades']}")


def predict_next_day(cfg: Config) -> None:
    """Load the saved model and forecast the next trading day's return, close price and direction."""
    ckpt_path = run_dir(cfg) / "model.joblib"
    if not ckpt_path.exists():
        raise SystemExit(f"No trained model at {ckpt_path}. Run `python main.py train` first.")
    ckpt = joblib.load(ckpt_path)
    saved = ckpt["config"]

    # Always pull fresh data up to today for a live prediction.
    df = add_features(load_prices(cfg.ticker, saved["start"], None, cfg.data_dir, refresh=True), saved["n_lags"])
    X = last_row(df, feature_names(saved["n_lags"]), ckpt["x_scaler"])
    ret = float(ckpt["model"].predict(X)[0])
    last_date, last_close = df.index[-1], float(df["Close"].iloc[-1])
    next_date = last_date + pd.offsets.BDay(1)

    print(f"\nTicker:            {cfg.ticker}")
    print(f"Last close ({last_date.date()}): {last_close:,.2f}")
    print(f"Predicted return for {next_date.date()}: {np.expm1(ret) * 100:+.3f}%")
    print(f"Predicted close:   {last_close * np.exp(ret):,.2f}")
    print(f"Predicted direction: {'UP' if ret > 0 else 'DOWN'}")
    print("\nAcademic use only — not financial advice.")


def main() -> None:
    """Parse command-line arguments, build the Config, and dispatch to train or predict."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", nargs="?", default="train", choices=["train", "predict"],
                   help="what to run (default: train)")
    p.add_argument("--ticker", default=Config.ticker)
    p.add_argument("--start", default=Config.start)
    p.add_argument("--end", default=None)
    p.add_argument("--lags", type=int, default=Config.n_lags, help="past daily returns used as input")
    p.add_argument("--refresh", action="store_true", help="re-download data instead of using cache")
    a = p.parse_args()

    cfg = Config(ticker=a.ticker, start=a.start, end=a.end, n_lags=a.lags)

    if a.command == "train":
        train_and_evaluate(cfg, a.refresh)
    else:
        predict_next_day(cfg)


if __name__ == "__main__":
    main()
