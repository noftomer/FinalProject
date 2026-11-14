"""Feature ablation for the volatility target, scored on the VALIDATION split only, over several seeds.

The test split is never scored here, so feature choices cannot leak into the final test result.
Also scores simple baselines on the same validation rows: training mean, trailing 5-day vol,
GARCH(1,1) (fitted on the training period only) and a ridge regression on the last day's features.

Run:  python -m src.ablation --ticker AAPL --seeds 5
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.linear_model import Ridge

from src.config import Config
from src.data import load_market, load_prices
from src.dataset import prepare
from src.features import FEATURES, MARKET_FEATURES, VOL_HORIZON, add_features
from src.train import fit

# Feature groups removed one at a time ("drop-one-group" ablation).
GROUPS = {
    "market (SPY+VIX)": ["spy_ret", "vix_level", "vix_chg"],
    "VIX only": ["vix_level", "vix_chg"],
    "calendar (day of week)": ["dow_sin", "dow_cos"],
    "trend (RSI, MACD, SMA gaps)": ["rsi", "macd_hist", "sma_gap_10", "sma_gap_20", "sma_gap_50"],
    "volume change": ["vol_chg"],
    "range (HL, Parkinson)": ["hl_range", "log_parkinson"],
    "past vol (vol10, log_vol_22, |ret|)": ["volatility_10", "log_vol_22", "abs_ret"],
    "signed return/gap": ["log_ret", "open_gap"],
}
# A minimal set that only carries volatility information.
VOL_ONLY = ["log_vol_22", "log_parkinson", "volatility_10", "abs_ret", "vix_level"]


def val_metrics(y: np.ndarray, pred: np.ndarray, train_mean: float) -> dict:
    """RMSE, R^2 against the training mean, and correlation (log-volatility units)."""
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    base = float(np.sqrt(np.mean((y - train_mean) ** 2)))
    corr = float(np.corrcoef(y, pred)[0, 1]) if pred.std() > 0 else float("nan")
    return {"rmse": rmse, "r2": 1 - (rmse / base) ** 2, "corr": corr}


# ---------------------------------------------------------------- GARCH(1,1), no extra dependency
def _garch_var(params: np.ndarray, r: np.ndarray, v0: float) -> np.ndarray:
    """Conditional variance path; element t is the variance of r[t] given r[:t]."""
    omega, alpha, beta = params
    v = np.empty(len(r) + 1)
    v[0] = v0
    for t in range(len(r)):
        v[t + 1] = omega + alpha * r[t] ** 2 + beta * v[t]
    return v


def fit_garch(r_train: np.ndarray) -> np.ndarray:
    """Gaussian MLE for GARCH(1,1) with variance targeting start; returns (omega, alpha, beta)."""
    scale = r_train.std()
    r = r_train / scale   # work in standardized units for a stable optimizer

    def nll(p):
        if p[0] <= 0 or p[1] < 0 or p[2] < 0 or p[1] + p[2] >= 0.9999:
            return 1e10
        v = _garch_var(p, r, r.var())[:-1]
        return 0.5 * np.sum(np.log(v) + r ** 2 / v)

    best = None
    for a0, b0 in [(0.05, 0.90), (0.10, 0.85), (0.15, 0.70)]:
        res = minimize(nll, [r.var() * (1 - a0 - b0), a0, b0], method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, beta = best.x
    return np.array([omega * scale ** 2, alpha, beta])


def garch_forecast(params: np.ndarray, r_all: np.ndarray, h: int = VOL_HORIZON) -> np.ndarray:
    """log of the forecast RMS daily vol over days t+1..t+h, made at the close of each day t (params fixed)."""
    omega, alpha, beta = params
    persistence = alpha + beta
    long_run = omega / (1 - persistence)
    v_next = _garch_var(params, r_all, np.var(r_all[:250]))[1:]   # variance of day t+1 given data up to t
    k = np.arange(h)
    avg = long_run + (v_next[:, None] - long_run) * persistence ** k   # E[var of day t+1+k]
    return 0.5 * np.log(avg.mean(axis=1))


# ---------------------------------------------------------------- experiment
def _train_job(job: tuple) -> tuple:
    """Train one (variant, seed) model and score it on validation. Top-level so worker processes can run it."""
    name, cols, s, df, cfg, train_mean = job
    if mp.parent_process() is not None:   # in a worker: avoid CPU oversubscription between processes
        torch.set_num_threads(1)
    data = prepare(df, cfg.seq_len, cfg.val_ratio, cfg.test_ratio, cols)
    with contextlib.redirect_stdout(io.StringIO()):   # silence per-epoch logging
        model, hist = fit(data.train.X, data.train.y, data.val.X, data.val.y, replace(cfg, seed=s))
    m = val_metrics(data.val.y, model.predict(data.val.X), train_mean)
    return name, s, m, hist["best_epoch"], len(hist["val_loss"]), min(hist["val_loss"])


def run(ticker: str, seeds: int, epochs: int, workers: int) -> dict:
    cfg = Config(ticker=ticker, target="volatility", epochs=epochs)
    prices = load_prices(cfg.ticker, cfg.start, cfg.end, cfg.data_dir)
    df = add_features(prices, "volatility", load_market(cfg.start, cfg.end, cfg.data_dir))
    all_cols = FEATURES + MARKET_FEATURES

    variants = {"ALL (baseline model)": all_cols, "VOL-ONLY (5 features)": VOL_ONLY}
    for name, drop in GROUPS.items():
        variants[f"- {name}"] = [c for c in all_cols if c not in drop]

    rows, per_seed = {}, {}
    ref = prepare(df, cfg.seq_len, cfg.val_ratio, cfg.test_ratio, all_cols)
    y_val, y_train = ref.val.y, ref.train.y
    train_mean = float(y_train.mean())
    print(f"{ticker}: train {len(y_train)} | val {len(y_val)} "
          f"({ref.val.dates[0].date()} -> {ref.val.dates[-1].date()}) | test NOT used")

    # Every (variant, seed) pair is an independent, seeded training run, so they can run in parallel
    # without changing any result; only the order of the printed lines differs.
    jobs = [(name, cols, s, df, cfg, train_mean) for name, cols in variants.items() for s in range(seeds)]
    workers = max(1, min(workers, len(jobs)))
    per_seed = {name: [None] * seeds for name in variants}
    print(f"Running {len(jobs)} trainings on {workers} worker process(es)")

    def _record(res):
        name, s, m, best_epoch, n_epochs, val_loss = res
        per_seed[name][s] = m
        print(f"  {name:<38} seed {s + 1}/{seeds} | best epoch {best_epoch:>3}/{n_epochs} "
              f"| val loss {val_loss:.4f} | val RMSE {m['rmse']:.4f}", flush=True)

    if workers == 1:
        for job in jobs:
            _record(_train_job(job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in as_completed([ex.submit(_train_job, job) for job in jobs]):
                _record(fut.result())
    for name, lst in per_seed.items():
        rows[name] = {f"{m}_{agg}": float(getattr(np, agg)([x[m] for x in lst]))
                      for m in ("rmse", "r2", "corr") for agg in ("mean", "std")}

    # Baselines on the same validation rows.
    baselines = {"TRAIN_MEAN": val_metrics(y_val, np.full_like(y_val, train_mean), train_mean),
                 "TRAILING_5D": val_metrics(y_val, ref.val.trailing_vol, train_mean)}

    r_all = np.log(prices["Close"] / prices["Close"].shift(1)).dropna()
    train_end = ref.train.dates[-1]
    params = fit_garch(r_all[r_all.index <= train_end].values)
    g = pd.Series(garch_forecast(params, r_all.values), index=r_all.index)
    g_val, g_train = g.reindex(ref.val.dates).values, g.reindex(ref.train.dates).values
    bias = float(np.mean(ref.train.y - g_train))   # Jensen correction (log of mean variance), train only
    baselines["GARCH(1,1)"] = val_metrics(y_val, g_val, train_mean)
    baselines["GARCH(1,1) + train bias fix"] = val_metrics(y_val, g_val + bias, train_mean)

    # Ridge on last-day features: is a nonlinear sequence model needed at all?
    ridge = Ridge(alpha=10.0).fit(ref.train.X[:, -1], y_train)
    baselines["RIDGE (last day, all features)"] = val_metrics(y_val, ridge.predict(ref.val.X[:, -1]), train_mean)
    omega, alpha, beta = params
    print(f"GARCH params: omega={omega:.3e} alpha={alpha:.3f} beta={beta:.3f} persistence={alpha + beta:.3f}")
    return {"ticker": ticker, "seeds": seeds, "val_period": [str(ref.val.dates[0].date()), str(ref.val.dates[-1].date())],
            "n_val": int(len(y_val)), "garch_params": [float(x) for x in params],
            "variants": rows, "per_seed": per_seed, "baselines": baselines}


def print_tables(res: dict) -> None:
    v = res["variants"]
    full = v["ALL (baseline model)"]["rmse_mean"]
    print(f"\n=== Validation ablation, {res['seeds']} seeds, {res['val_period'][0]} -> {res['val_period'][1]} ===")
    print(f"{'variant':<40}{'RMSE mean+-std':>18}{'dRMSE vs ALL':>14}{'R2 mean':>10}{'corr mean':>11}")
    for name, m in v.items():
        print(f"{name:<40}{m['rmse_mean']:>11.4f}+-{m['rmse_std']:.4f}{m['rmse_mean'] - full:>+14.4f}"
              f"{m['r2_mean']:>10.3f}{m['corr_mean']:>11.3f}")
    print("\n=== Baselines on the same validation rows ===")
    print(f"{'baseline':<40}{'RMSE':>10}{'R2':>10}{'corr':>10}")
    for name, m in res["baselines"].items():
        print(f"{name:<40}{m['rmse']:>10.4f}{m['r2']:>10.3f}{m['corr']:>10.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                   help="parallel training processes (default: all CPU cores; 1 = sequential)")
    a = p.parse_args()
    out = run(a.ticker, a.seeds, a.epochs, a.workers)
    path = Config().output_dir / f"{a.ticker}_ablation" / "ablation_val.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))   # save first so a printing problem can't lose the results
    print_tables(out)
    print(f"\nSaved {path}")
