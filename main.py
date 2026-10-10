"""Stock forecasting with a PyTorch LSTM (next-day return/price and 5-day volatility) — command-line entry point.

Examples
--------
    python main.py predict              # forecast with every trained target (return + volatility)
    python main.py train --ticker AAPL  # train both targets
    python main.py train --target return
    python main.py predict --ticker AAPL
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data import load_prices
from src.dataset import last_window, prepare
from src.evaluate import (baseline_metrics, direction_significance, plot_all, plot_volatility,
                          regression_metrics, volatility_metrics)
from src.features import add_features
from src.models import build_model
from src.train import fit

TARGETS = ["return", "volatility"]


def run_dir(cfg: Config) -> Path:
    """Output directory for this ticker/model combination (e.g. outputs/AAPL_lstm)."""
    suffix = "_vol" if cfg.target == "volatility" else ""
    return cfg.output_dir / f"{cfg.ticker.replace('^', '')}_{cfg.model}{suffix}"


def train_and_evaluate(cfg: Config, refresh: bool = False) -> dict:
    """Train the model, evaluate it on the test split, and save plots, metrics and the checkpoint."""
    df = add_features(load_prices(cfg.ticker, cfg.start, cfg.end, cfg.data_dir, refresh), cfg.target)
    data = prepare(df, cfg.seq_len, cfg.val_ratio, cfg.test_ratio)
    print(f"[{cfg.ticker} | {cfg.model}] samples: train={len(data.train.y)} "
          f"val={len(data.val.y)} test={len(data.test.y)} | "
          f"test period {data.test.dates[0].date()} -> {data.test.dates[-1].date()}")

    model, history = fit(data.train.X, data.train.y, data.val.X, data.val.y, cfg)

    t = data.test
    y_pred = model.predict(t.X)

    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    title = f"{cfg.ticker} {cfg.model.upper()}"
    if cfg.target == "volatility":
        results = {"config": cfg.to_dict(),
                   "metrics": volatility_metrics(t.y, y_pred, t.trailing_vol, data.train.y),
                   "history": history}
        plot_volatility(out, t.dates, t.y, y_pred, history, title)
        pd.DataFrame({"date": t.dates, "actual_log_vol": t.y, "pred_log_vol": y_pred,
                      "trailing_log_vol": t.trailing_vol}).to_csv(out / "test_predictions.csv", index=False)
    else:
        results = {"config": cfg.to_dict(),
                   "model": regression_metrics(t.y, y_pred, t.close),
                   "baselines": baseline_metrics(t.y, t.close, data.train.y),
                   "direction_test": direction_significance(t.y, y_pred),
                   "history": history}
        plot_all(out, t.dates, t.y, y_pred, t.close, history, title)
        pd.DataFrame({"date": t.dates, "close": t.close, "actual_next_ret": t.y,
                      "pred_next_ret": y_pred}).to_csv(out / "test_predictions.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(), "x_scaler": data.x_scaler},
               out / "model.pt")

    print_report(results)
    print(f"\nArtifacts saved to {out}")
    return results


def print_report(r: dict) -> None:
    """Print test-set metrics, and the direction significance test to the console."""
    print("\n=== Test-set metrics ===")
    if "metrics" in r:  # volatility target
        print(pd.DataFrame(r["metrics"]).T.to_string(float_format=lambda v: f"{v:.5f}"))
        print("\n(log-volatility space; r2_vs_train_mean > 0 and a lower rmse than TRAILING_5D = real skill)")
        return
    rows = {"MODEL": r["model"], **r["baselines"]}
    table = pd.DataFrame(rows).T[["return_rmse", "return_mae", "price_rmse",
                                  "price_mape_%", "directional_accuracy_%"]]
    print(table.to_string(float_format=lambda v: f"{v:.5f}"))
    d = r["direction_test"]
    print(f"\nDirection: {d['hits']}/{d['n']} correct; majority-class rate "
          f"{d['majority_rate_%']:.1f}%; binomial p-value = {d['p_value']:.4f}")


def predict_next_day(cfg: Config) -> None:
    """Load the saved model and forecast the next trading day's return, close price and direction."""
    ckpt_path = run_dir(cfg) / "model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"No trained model at {ckpt_path}. Run `python main.py train` first.")
    ckpt = torch.load(ckpt_path, weights_only=False)
    saved = ckpt["config"]
    model = build_model(ckpt["x_scaler"].n_features_in_, saved["hidden_size"], saved["num_layers"], saved["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Always pull fresh data up to today for a live prediction.
    df = add_features(load_prices(cfg.ticker, saved["start"], None, cfg.data_dir, refresh=True), cfg.target)
    X = last_window(df, saved["seq_len"], ckpt["x_scaler"])
    model_result=model.predict(X)
    ret = float(model_result[0])
    if cfg.target == "volatility":
        print(f"Ticker: {cfg.ticker} | predicted daily volatility over the next 5 trading days: "
              f"{np.exp(ret) * 100:.2f}% (annualized ~{np.exp(ret) * np.sqrt(252) * 100:.1f}%)")
        print("Academic use only — not financial advice.")
        return
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
    p.add_argument("--target", default="all", choices=["all", "return", "volatility"],
                   help="next-day return, next-5-day volatility, or both (default: all)")
    p.add_argument("--seq-len", type=int, default=Config.seq_len, help="past trading days the LSTM sees")
    p.add_argument("--epochs", type=int, default=Config.epochs, help="maximum training epochs")
    p.add_argument("--refresh", action="store_true", help="re-download data instead of using cache")
    a = p.parse_args()

    targets = TARGETS if a.target == "all" else [a.target]
    for target in targets:
        print(f"\n########## target: {target} ##########")
        cfg = Config(ticker=a.ticker, start=a.start, end=a.end, target=target, seq_len=a.seq_len, epochs=a.epochs)
        if a.command == "train":
            train_and_evaluate(cfg, a.refresh)
        else:
            predict_next_day(cfg)


if __name__ == "__main__":
    main()
