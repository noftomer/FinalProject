"""Error analysis of the stored test-set predictions (no retraining).

Usage:  python -m src.error_analysis --ticker AAPL

Reads ``outputs/<TICKER>_lstm*/test_predictions.csv`` and breaks the errors down by period and by market regime.
Writes ``outputs/<TICKER>_error_analysis/error_analysis.json`` and two plots, and prints the tables.
"""
import argparse
import json

import matplotlib
import numpy as np
import pandas as pd

from src.config import Config
from src.data import load_prices
from src.dataset import prepare
from src.features import add_features

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be selected first)

BLUE, GRAY = "#1f5fbf", "#9aa5b1"


def _rmse(err: pd.Series) -> float:
    return float(np.sqrt(np.mean(err ** 2)))


def volatility_analysis(pred: pd.DataFrame, train_mean: float) -> dict:
    """Errors in log-volatility space, by actual-volatility level and by year, versus the train-mean baseline."""
    pred = pred.assign(err=pred["pred_log_vol"] - pred["actual_log_vol"],
                       err_base=train_mean - pred["actual_log_vol"],
                       year=pd.to_datetime(pred["date"]).dt.year)
    pred["level"] = pd.qcut(pred["actual_log_vol"], 5, labels=["Q1 (calmest)", "Q2", "Q3", "Q4", "Q5 (most volatile)"])

    def summarize(g):
        return pd.Series({"n": len(g), "mean_actual": g["actual_log_vol"].mean(), "mean_pred": g["pred_log_vol"].mean(),
                          "bias": g["err"].mean(), "rmse_model": _rmse(g["err"]), "rmse_train_mean": _rmse(g["err_base"])})

    by_level = pred.groupby("level", observed=True).apply(summarize, include_groups=False)
    by_year = pred.groupby("year").apply(summarize, include_groups=False)
    worst = pred.reindex(pred["err"].abs().sort_values(ascending=False).index).head(10)
    return {
        "train_mean_log_vol": train_mean,
        "test_mean_log_vol": float(pred["actual_log_vol"].mean()),
        "overall_bias": float(pred["err"].mean()),
        "std_actual": float(pred["actual_log_vol"].std()),
        "std_pred": float(pred["pred_log_vol"].std()),
        "by_actual_level": by_level.round(4).reset_index().astype({"level": str}).to_dict("records"),
        "by_year": by_year.round(4).reset_index().to_dict("records"),
        "worst_days": worst[["date", "actual_log_vol", "pred_log_vol", "err"]].round(4).to_dict("records"),
    }


def return_analysis(pred: pd.DataFrame) -> dict:
    """Directional accuracy and error by year / volatility regime / move size for the next-day return."""
    p = pred.assign(year=pd.to_datetime(pred["date"]).dt.year,
                    err=pred["pred_next_ret"] - pred["actual_next_ret"],
                    hit=np.sign(pred["pred_next_ret"]) == np.sign(pred["actual_next_ret"]),
                    up=pred["actual_next_ret"] > 0)
    # regime = realised volatility of the 10 days up to the prediction date, rebuilt from the test closes
    p["recent_vol"] = np.log(p["close"] / p["close"].shift(1)).rolling(10).std()
    p["regime"] = pd.qcut(p["recent_vol"], 4, labels=["calm", "Q2", "Q3", "turbulent"])
    p["conf"] = pd.qcut(p["pred_next_ret"].abs(), 4, labels=["lowest", "Q2", "Q3", "highest"])

    def summarize(g):
        return pd.Series({"n": len(g), "directional_acc_%": 100 * g["hit"].mean(), "share_up_%": 100 * g["up"].mean(),
                          "rmse_model": _rmse(g["err"]), "rmse_zero": _rmse(g["actual_next_ret"])})

    big = p.reindex(p["actual_next_ret"].abs().sort_values(ascending=False).index).head(10)
    return {
        "std_actual": float(p["actual_next_ret"].std()),
        "std_pred": float(p["pred_next_ret"].std()),
        "corr": float(p["pred_next_ret"].corr(p["actual_next_ret"])),
        "share_pred_up_%": float(100 * (p["pred_next_ret"] > 0).mean()),
        "by_year": p.groupby("year").apply(summarize, include_groups=False).round(4).reset_index().to_dict("records"),
        "by_regime": p.dropna(subset=["regime"]).groupby("regime", observed=True).apply(summarize, include_groups=False)
                      .round(4).reset_index().astype({"regime": str}).to_dict("records"),
        "by_prediction_size": p.groupby("conf", observed=True).apply(summarize, include_groups=False)
                              .round(4).reset_index().astype({"conf": str}).to_dict("records"),
        "ten_largest_moves": big[["date", "actual_next_ret", "pred_next_ret"]].round(4).to_dict("records"),
    }


def _plots(out_dir, vol: pd.DataFrame | None, vol_res: dict | None, ret_res: dict | None) -> None:
    if vol is not None and vol_res is not None:
        lv = pd.DataFrame(vol_res["by_actual_level"])
        x = np.arange(len(lv))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].bar(x - 0.2, lv["rmse_model"], 0.4, color=BLUE, label="LSTM")
        ax[0].bar(x + 0.2, lv["rmse_train_mean"], 0.4, color=GRAY, label="train mean")
        ax[0].set_xticks(x, [s.split(" ")[0] for s in lv["level"]])
        ax[0].set(title="Volatility: RMSE by actual volatility level", ylabel="RMSE (log-vol)")
        ax[0].legend()
        ax[1].scatter(vol["actual_log_vol"], vol["pred_log_vol"], s=8, color=BLUE, alpha=0.5)
        lim = [vol["actual_log_vol"].min(), vol["actual_log_vol"].max()]
        ax[1].plot(lim, lim, color=GRAY, ls="--", label="perfect")
        ax[1].set(title="Predicted vs actual log-volatility", xlabel="actual", ylabel="predicted")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_dir / "volatility_errors.png", dpi=130)
        plt.close(fig)
    if ret_res is not None:
        yr = pd.DataFrame(ret_res["by_year"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(yr["year"].astype(str), yr["directional_acc_%"], color=BLUE, label="LSTM directional accuracy")
        ax.plot(yr["year"].astype(str), yr["share_up_%"], color=GRAY, marker="o", label="share of up days")
        ax.axhline(50, color="black", lw=0.6)
        ax.set(title="Return: directional accuracy by year", ylabel="%", ylim=(30, 70))
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "return_errors.png", dpi=130)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=Config.ticker)
    a = ap.parse_args()
    cfg = Config(ticker=a.ticker)
    t = a.ticker.replace("^", "")
    out_dir = cfg.output_dir / f"{t}_error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    result, vol, ret = {}, None, None

    vol_csv = cfg.output_dir / f"{t}_lstm_vol" / "test_predictions.csv"
    if vol_csv.exists():
        vol = pd.read_csv(vol_csv)
        prices = load_prices(cfg.ticker, cfg.start, cfg.end, cfg.data_dir)
        train_y = prepare(add_features(prices, "volatility"), cfg.seq_len, cfg.val_ratio, cfg.test_ratio).train.y
        result["volatility"] = volatility_analysis(vol, float(train_y.mean()))
    ret_csv = cfg.output_dir / f"{t}_lstm" / "test_predictions.csv"
    if ret_csv.exists():
        ret = pd.read_csv(ret_csv)
        result["return"] = return_analysis(ret)
    if not result:
        raise SystemExit(f"No test_predictions.csv under {cfg.output_dir}. Run `python main.py train --ticker {a.ticker}`.")

    (out_dir / "error_analysis.json").write_text(json.dumps(result, indent=2, default=str))
    _plots(out_dir, vol, result.get("volatility"), result.get("return"))
    for target, res in result.items():
        print(f"\n===== {target} =====")
        for k, v in res.items():
            print(f"\n-- {k}")
            print(pd.DataFrame(v).to_string(index=False) if isinstance(v, list) else v)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()
