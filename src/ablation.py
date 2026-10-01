"""Feature ablation: does adding technical indicators improve next-day forecasts?

Every variant is trained with identical data, splits, model and seeds; only the input
feature set changes. Results are averaged over several seeds because single runs differ
by more than the effect being measured.
"""
import contextlib
import io
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.config import Config
from src.data import load_prices
from src.dataset import loader, prepare
from src.evaluate import backtest, direction_significance, regression_metrics
from src.features import add_features
from src.models import build_model
from src.train import fit, predict, set_seed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BLUE, GRAY = "#1f5fbf", "#8a8a8a"

# Name -> feature subset. The last variant is the full set from Config.
VARIANTS = {
    "returns_only": ["log_ret"],
    "returns+volume+calendar": ["log_ret", "volume_change", "day_of_week"],
    "all_16_features": None,  # filled from cfg.features
}


def _run_once(df: pd.DataFrame, cfg: Config, features: list[str], seed: int) -> dict:
    """Train one model on ``features`` with ``seed`` and return its test-set metrics."""
    set_seed(seed)
    data = prepare(df, features, cfg.window, cfg.val_ratio, cfg.test_ratio)
    model = build_model(len(features), cfg.hidden_size, cfg.num_layers, cfg.dropout)
    with contextlib.redirect_stdout(io.StringIO()):  # silence per-epoch logging
        history = fit(model,
                      loader(data.train, data.y_scale, cfg.batch_size, shuffle=True),
                      loader(data.val, data.y_scale, cfg.batch_size, shuffle=False),
                      cfg.epochs, cfg.lr, cfg.weight_decay, cfg.patience, DEVICE)

    t = data.test
    y_pred = predict(model, t.X, data.y_scale, DEVICE)
    m = regression_metrics(t.y, y_pred, t.close)
    rw = regression_metrics(t.y, np.zeros_like(t.y), t.close)
    d = direction_significance(t.y, y_pred)
    bt = backtest(t.y, y_pred)
    return {
        "return_rmse": m["return_rmse"],
        "rmse_vs_random_walk_%": (m["return_rmse"] / rw["return_rmse"] - 1) * 100,
        "directional_accuracy_%": m["directional_accuracy_%"],
        "majority_rate_%": d["majority_rate_%"],
        "p_value": d["p_value"],
        "strategy_sharpe": bt["strategy"]["sharpe"],
        "buy_hold_sharpe": bt["buy_and_hold"]["sharpe"],
        "epochs_trained": len(history["train_loss"]),
    }


def run_ablation(cfg: Config, seeds: list[int], refresh: bool = False) -> pd.DataFrame:
    """Run every variant for every seed; save per-run and summary tables and a plot; return the per-run table."""
    df = add_features(load_prices(cfg.ticker, cfg.start, cfg.end, cfg.data_dir, refresh))
    # Drop rows missing ANY full-set feature so all variants share identical samples and test dates.
    df = df.dropna(subset=cfg.features)

    variants = {k: (v if v is not None else list(cfg.features)) for k, v in VARIANTS.items()}
    rows = []
    for name, feats in variants.items():
        for seed in seeds:
            res = _run_once(df, replace(cfg, seed=seed), feats, seed)
            rows.append({"variant": name, "n_features": len(feats), "seed": seed, **res})
            print(f"[{name} | seed {seed}] rmse={res['return_rmse']:.5f} "
                  f"dir={res['directional_accuracy_%']:.1f}% p={res['p_value']:.3f} "
                  f"sharpe={res['strategy_sharpe']:.2f}")

    runs = pd.DataFrame(rows)
    metrics = ["return_rmse", "rmse_vs_random_walk_%", "directional_accuracy_%",
               "majority_rate_%", "p_value", "strategy_sharpe", "buy_hold_sharpe"]
    summary = runs.groupby("variant", sort=False)[metrics].agg(["mean", "std"])

    out = cfg.output_dir / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "ablation_runs.csv", index=False)
    summary.to_csv(out / "ablation_summary.csv")
    _plot(runs, out / "ablation.png")

    print(f"\n=== Ablation summary (mean ± std over {len(seeds)} seeds) ===")
    for name in variants:
        r = runs[runs["variant"] == name]
        print(f"{name:26s} rmse vs RW {r['rmse_vs_random_walk_%'].mean():+.2f}% ± {r['rmse_vs_random_walk_%'].std():.2f} | "
              f"dir {r['directional_accuracy_%'].mean():.1f}% ± {r['directional_accuracy_%'].std():.1f} "
              f"(majority {r['majority_rate_%'].iloc[0]:.1f}%) | "
              f"Sharpe {r['strategy_sharpe'].mean():.2f} ± {r['strategy_sharpe'].std():.2f} "
              f"(buy&hold {r['buy_hold_sharpe'].iloc[0]:.2f})")
    print(f"\nArtifacts saved to {out}")
    return runs


def _plot(runs: pd.DataFrame, path) -> None:
    """Bar chart of mean directional accuracy and strategy Sharpe per variant, with std error bars."""
    g = runs.groupby("variant", sort=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title, ref_col in [
        (axes[0], "directional_accuracy_%", "Directional accuracy (%)", "majority_rate_%"),
        (axes[1], "strategy_sharpe", "Strategy Sharpe", "buy_hold_sharpe"),
    ]:
        mean, std = g[col].mean(), g[col].std().fillna(0)
        ax.bar(range(len(mean)), mean, yerr=std, color=BLUE, capsize=4)
        ax.axhline(g[ref_col].first().iloc[0], color=GRAY, ls="--",
                   label="always 'up'" if ref_col == "majority_rate_%" else "buy & hold")
        ax.set_xticks(range(len(mean)))
        ax.set_xticklabels(mean.index, rotation=15, ha="right")
        ax.set(title=title)
        ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
