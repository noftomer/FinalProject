# Final Report: Predicting Stock Returns and Volatility with an LSTM

> For study purposes only. Not financial advice. See also [`PLAN.md`](PLAN.md) and [`../README.md`](../README.md).

## 1. Summary

An LSTM (PyTorch) reads the last 40 days of a stock's returns, volume and indicators and predicts the **next-day
return** and the **volatility over the next 5 days**. Main test: AAPL, 2010-01-04 to 2026-10-22 (4,227 days). The model
was tested once on the most recent, unseen data and compared with simple baselines.

| Target | Result on test data | Meaning |
|---|---|---|
| Volatility | RMSE 0.516 vs 0.537 (training average) and 0.672 (last 5 days). Correlation 0.34, R² 0.076 | Beats both baselines, but only slightly, mostly thanks to 2025 |
| Return | Same error as baselines. Direction right 55.7% vs 56.2% for always "up" (p = 0.61) | No skill. The model almost always says "up" |

Main finding: volatility is partly predictable, tomorrow's return looks like noise, as the efficient-market idea
suggests. We confirmed it with a statistical test.

## 2. What we did

- **Data:** daily prices and volume from Yahoo Finance (`yfinance`), cached as CSV in `data/`.
- **Features (18, all from past data):** 15 stock features (returns, volume change, RSI, MACD, several volatility
  measures, high-low range, distance from moving averages, overnight gap, day of week) and 3 market features (SPY
  return, VIX level and change).
- **Targets:** next-day log return, and log of 5-day volatility.
- **Split:** 40-day windows, split by time 70/15/15 (train/validation/test). Scaler fitted on train only.
- **Model:** LSTM, 1 layer, 32 hidden units, dropout 0.2. Small, because there are only about 3,000 noisy samples.
- **Training:** Huber loss, Adam (lr 3e-4), weight decay 1e-2, batch 64, early stopping (patience 10), seed 42.
- **Evaluation:** RMSE, MAE, MAPE, R² vs training average, correlation, directional accuracy with a binomial test.
- **System:** `main.py` (`train`, `predict`) and a `run.bat` launcher.

The test set is the last 15% of days (624 days from 2024-04-22; 623 used for direction). Not tried: GARCH/ARIMA, tree
models, Transformers.

## 3. Results

<!-- RESULTS:START -->
Source: `outputs/AAPL_lstm_vol/results.json` and `outputs/AAPL_lstm/results.json` (auto-generated).

### Volatility (log units)

| | RMSE | MAE | R² | Correlation |
|---|---|---|---|---|
| **MODEL** | **0.499** | **0.398** | **0.129** | **0.368** |
| TRAIN_MEAN | 0.535 | 0.416 | 0.000 | – |
| TRAILING_5D | 0.666 | 0.520 | -0.551 | 0.221 |

### Return

| | Return RMSE | Return MAE | Direction correct % |
|---|---|---|---|
| **MODEL** | 0.017555 | 0.011803 | 55.6 |
| ZERO (tomorrow = today) | 0.017594 | 0.011865 | – |
| TRAIN_MEAN | 0.017568 | 0.011790 | 55.2 |
| ALWAYS_UP | 0.017594 | 0.011865 | 55.2 |

Direction right on 345 of 621 days. Always "up" gives 55.2%. Binomial test p = 0.45.
<!-- RESULTS:END -->

The model beats both baselines, but the gain is small. Volatility level is the same in training and test (-4.34 vs
-4.36), so there is no level shift.

- Not better than the training average on RMSE or MAE. The small price error (MAPE about 1.18%) equals the "tomorrow =
  today" baseline, so it reflects small daily moves, not skill.
- Losses stay at the "predict the average" level and training stops early (15 epochs): there is nothing to learn.

Plots are in `outputs/<run>/` (`loss_curve.png`, `price_prediction.png`, `return_scatter.png`,
`volatility_prediction.png`).

## 4. Error analysis

Made with `python -m src.error_analysis --ticker AAPL`. Details in `outputs/AAPL_error_analysis/`.

**Volatility**
- Predictions stay near the average (spread 0.25 vs 0.54 for real values): too high on calm days (bias +0.74 in the
  calmest group), too low on wild days (-0.50 in the most volatile group).
- The gain comes from wild days: in the most volatile group RMSE is 0.637 vs 0.779 for the training average. In the
  middle groups the model is slightly worse than a constant.
- The gain comes from one year. The model wins only in 2025 (RMSE 0.582 vs 0.640) and is slightly worse in 2024 and
  2026. Seven of the ten biggest misses are in 2025. So the advantage depends on one stormy period and is not stable.

**Return**
- The model is basically a constant "up" predictor: it says "up" on 97.8% of days, and its yearly accuracy equals the
  share of up days (63.4%, 51.6%, 53.7% for 2024, 2025, 2026 vs 63.4%, 52.4%, 53.7%). Correlation with real returns
  is -0.02.
- It misses big moves: the ten largest (for example +14.3% on 2025-04-08) all got predictions between -0.06% and
  +0.14%.
- No market condition or prediction-size group shows hidden skill.

**Limitations:** one split and one seed (no confidence range); one stock; no macro or news inputs; no trading costs, so
nothing is said about profit; a test period of only 2.5 years; explanations are read from tables, not proven by an
experiment.

## 5. Speed and failure handling

Tests: `pytest tests` (21 tests, all pass, offline, about 3 seconds).

- **Speed** (CPU only, no network): one forward pass takes 0.34 ms. `predict` end to end takes 35 ms (return model)
  and 31 ms (volatility model). The Yahoo download is not included. Training time was not measured.
- **Failures handled** (tested in `tests/test_failures.py`): unknown ticker, download error, no network (falls back to
  saved data), missing or corrupted model file, wrong arguments, too little history, constant prices, missing values.
  Each gives a clear message instead of a crash.
- **Fixed while testing:** unclear errors on short history, unhandled `yfinance` errors, `predict` crashing offline
  even with saved data, and raw errors on a corrupted model. Fixes are in `src/data.py`, `src/dataset.py` and
  `main.py`.
- **Not covered:** a corrupted CSV cache, unexpected Yahoo columns, and a wrong but normal-looking output (for example
  -50% in one day).

## 6. Next steps

Feature ablation over several seeds, a GARCH(1,1) baseline, walk-forward validation, more inputs (interest rates, news),
more stocks, and a simple UI or API.

## 7. Reproducibility

```bash
pip install -r requirements.txt
python main.py train --ticker AAPL
python main.py predict --ticker AAPL
```

Seed 42 gives the same results on the same machine. No API keys are needed.

## 8. Conclusions

- The pipeline avoids future data, is repeatable and tested, and one prediction takes about 35 ms.
- For volatility, the LSTM beats both baselines, but the gain is small, comes mostly from 2025, and is not a stable
  skill.
- For returns, it does not beat trivial baselines. The test shows these inputs carry no useful directional
  information.
- The main contribution is reporting the negative result honestly, with baselines.
