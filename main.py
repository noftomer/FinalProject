"""Next-day stock market prediction — command-line entry point.

Examples
--------
    python main.py                      # train the LSTM with all defaults from src/config.py
    python main.py train --ticker ^GSPC
    python main.py predict --ticker ^GSPC
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data import load_prices
from src.dataset import last_window, loader, prepare
from src.evaluate import (backtest, baselines, direction_significance, plot_all,
                          regression_metrics)
from src.features import add_features
from src.models import build_model
from src.train import fit, predict, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_dir(cfg: Config) -> Path:
    """Output directory for this ticker/model combination (e.g. outputs/GSPC_lstm)."""
    return cfg.output_dir / f"{cfg.ticker.replace('^', '')}_{cfg.model}"


def train_and_evaluate(cfg: Config, refresh: bool = False) -> dict:
    """Train the model, evaluate it on the test split against baselines, and save plots, metrics and the checkpoint."""
    set_seed(cfg.seed)
    df = add_features(load_prices(cfg.ticker, cfg.start, cfg.end, cfg.data_dir, refresh))
    data = prepare(df, cfg.features, cfg.window, cfg.val_ratio, cfg.test_ratio)
    print(f"[{cfg.ticker} | {cfg.model}] samples: train={len(data.train.y)} "
          f"val={len(data.val.y)} test={len(data.test.y)} | "
          f"test period {data.test.dates[0].date()} -> {data.test.dates[-1].date()}")

    model = build_model(len(cfg.features), cfg.hidden_size, cfg.num_layers, cfg.dropout)
    loader_train=loader(data.train, data.y_scale, cfg.batch_size, shuffle=True)
    loader_val=loader(data.val, data.y_scale, cfg.batch_size, shuffle=False)
    history = fit(model,
                  loader_train,
                  loader_val,
                  cfg.epochs, cfg.lr, cfg.weight_decay, cfg.patience, DEVICE)

    t = data.test
    y_pred = predict(model, t.X, data.y_scale, DEVICE)
    prev_ret = df.loc[t.dates, "log_ret"].values
    bt = backtest(t.y, y_pred)
    equity = bt.pop("_equity")

    config = cfg.to_dict()
    model_metrics = regression_metrics(t.y, y_pred, t.close)
    direction_test = direction_significance(t.y, y_pred)
    baseline_metrics = baselines(t.y, prev_ret, float(data.train.y.mean()), t.close)
    epochs_trained = len(history["train_loss"])

    results = {
        "config": config,
        "model": model_metrics,
        "direction_test": direction_test,
        "baselines": baseline_metrics,
        "backtest": bt,
        "epochs_trained": epochs_trained,
    }

    out = run_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    plot_all(out, history, t.dates, t.y, y_pred, t.close, equity, f"{cfg.ticker} {cfg.model.upper()}")
    pd.DataFrame({"date": t.dates, "close": t.close, "actual_next_ret": t.y,
                  "pred_next_ret": y_pred}).to_csv(out / "test_predictions.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                "x_scaler": data.x_scaler, "y_scale": data.y_scale}, out / "model.pt")

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
    ckpt_path = run_dir(cfg) / "model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"No trained model at {ckpt_path}. Run `python main.py train` first.")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=DEVICE)
    saved = ckpt["config"]

    # Always pull fresh data up to today for a live prediction.
    df = add_features(load_prices(cfg.ticker, saved["start"], None, cfg.data_dir, refresh=True))
    model = build_model(len(saved["features"]), saved["hidden_size"],
                        saved["num_layers"], saved["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])

    X = last_window(df, saved["features"], saved["window"], ckpt["x_scaler"])
    ret = float(predict(model, X, ckpt["y_scale"], DEVICE)[0])
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
    p.add_argument("--window", type=int, default=Config.window)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--hidden", type=int, default=Config.hidden_size)
    p.add_argument("--layers", type=int, default=Config.num_layers)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--refresh", action="store_true", help="re-download data instead of using cache")
    a = p.parse_args()

    cfg = Config(ticker=a.ticker, start=a.start, end=a.end, window=a.window,
                 epochs=a.epochs, hidden_size=a.hidden, num_layers=a.layers, lr=a.lr, seed=a.seed)

    if a.command == "train":
        train_and_evaluate(cfg, a.refresh)
    else:
        predict_next_day(cfg)


if __name__ == "__main__":
    main()
