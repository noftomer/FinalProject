# Final Report: Stock Return and Volatility Forecasting with an LSTM

> Academic use only. This is not financial advice.

Related documents: [`PLAN.md`](PLAN.md) (project plan, MVP and success criteria), [`../README.md`](../README.md)
(installation and usage).

## 1. Summary

A PyTorch LSTM is trained on the last 40 days of returns, volume and technical indicators of a stock (reference
experiment: AAPL, 2010-01-04 to 2026-10-22, 4,227 trading days) to predict two targets: the **next-day return** and the
**realized volatility over the next 5 trading days**. The model is evaluated once on a chronologically held-out test set
and compared with explicit baselines.

| Target | Result on the test set | Verdict |
|---|---|---|
| Volatility | RMSE 0.516 vs 0.537 (train mean) and 0.672 (trailing 5-day); correlation 0.34; R² 0.076 | beats both baselines overall, but the gain is modest and comes mainly from 2025 (section 6.1) |
| Return | RMSE equals the baselines; directional accuracy 55.7% vs 56.2% for always guessing "up" (p = 0.61) | no detectable signal; the model is effectively a constant "up" predictor (section 6.2) |

The main finding is the contrast: volatility is partly predictable from the history, while next-day returns are
indistinguishable from noise with these inputs. This is consistent with the efficient-market view, and the project
demonstrates it statistically rather than assuming it.

## 2. Problem and goals

See [`PLAN.md`](PLAN.md) for the full definition. In short: predict `ln(Close(t+1)/Close(t))` and the log realized
volatility of days t+1…t+5 from a 40-day window of features, and judge success against baselines on unseen data.
A negative result on the return target is an acceptable outcome as long as it is supported by a statistical test.

## 3. Work stages and technology choices

| Stage | What was done | Why |
|---|---|---|
| Data | Daily adjusted OHLCV from Yahoo Finance via `yfinance`, cached as CSV in `data/` | free, real data; the cache makes runs repeatable and works offline; no API key is needed |
| Features | 7 inputs per day: log return, change in log volume, RSI(14), price-normalized MACD histogram, 10-day volatility, high-low range, gap from the 20-day average | a compact set that covers price, volume, momentum, volatility and trend, built only from past values |
| Targets | Next-day log return; log of 5-day realized volatility | log returns are additive and roughly stationary; the log of volatility is closer to normal, which suits an MSE/Huber loss |
| Windows and split | Sliding windows of 40 days; 70/15/15 chronological train/validation/test; scaler fit on train only | shuffling would leak the future into training; time order is kept |
| Model | `nn.LSTM(7 → 32, 1 layer)`, dropout 0.2, linear head | a small model for roughly 3,000 noisy training samples; a larger one would mostly memorize noise |
| Training | Huber loss (delta 1.0), Adam (lr 3e-4), weight decay 1e-2, gradient clipping, batch 64, early stopping with patience 10 on validation loss, best-validation weights kept, seed 42 | robust to extreme days, regularized, reproducible |
| Evaluation | RMSE, MAE, MAPE, R² versus the training mean, correlation, directional accuracy with a one-sided binomial test | compares against what is easy to achieve, instead of reporting a bare number |
| System | `main.py` CLI (`train`, `predict`), `run.bat` launcher (builds a virtual environment and installs dependencies), ready-made `.bat` shortcuts | a runnable system that can be demonstrated |

Code layout: `src/data.py` (data), `src/features.py` and `src/dataset.py` (inputs), `src/models.py` and `src/train.py`
(model), `src/evaluate.py` (evaluation), `src/config.py` (all settings in one place).

### Options not implemented

The alternatives in the plan (GARCH/ARIMA, tree models, Transformers) were **not run**. The LSTM was chosen because it
models sequences directly and matches the course material; comparing it with a GARCH model on the volatility target is
the most valuable missing baseline (see section 8).

## 4. Experimental setup

- **Data:** AAPL, 2010-01-04 to 2026-10-22. The test set is the last 15% of days (624 days starting 2024-04-22; the
  directional test uses 623 of them).
- **Configuration:** `seq_len` 40, hidden size 32, 1 layer, dropout 0.2, lr 3e-4, weight decay 1e-2, batch 64,
  max 100 epochs, patience 10, seed 42.
- **Selection:** early stopping and model selection use the validation set only. The test set is used once at the end.
- **Training length:** the return model stopped after 15 epochs, the volatility model after 95.

## 5. Results

All numbers below come from the stored runs `outputs/AAPL_lstm_vol/results.json` and `outputs/AAPL_lstm/results.json`.

### 5.1 Volatility target (test set, log-volatility space)

| | RMSE | MAE | R² vs train mean | corr |
|---|---|---|---|---|
| **MODEL** | **0.516** | **0.405** | **0.076** | **0.337** |
| TRAIN_MEAN | 0.537 | 0.417 | 0.000 | – |
| TRAILING_5D | 0.672 | 0.526 | -0.569 | 0.209 |

- The model beats both baselines (criteria S1, S2 and S3 in the plan are met).
- The validation loss falls from about 1.0 to 0.48 while the training loss ends near 0.68, so the model learns a real
  signal in the validation period. On the test set the gain is smaller (R² 0.076). An earlier version of this report
  suggested that the volatility level differs between the training and test periods; the error analysis
  (section 6) **does not support this**: the mean log-volatility is -4.34 in training and -4.36 in the test set.
- `TRAILING_5D` is worse than the constant mean because the 5-day realized volatility is itself a noisy estimate of the
  future one; the LSTM acts partly as a smoother of it.

### 5.2 Return target (test set)

| | return RMSE | return MAE | price RMSE | price MAPE % | directional accuracy % |
|---|---|---|---|---|---|
| **MODEL** | 0.017588 | 0.011786 | 4.176 | 1.179 | 55.7 |
| ZERO (tomorrow = today) | 0.017615 | 0.011857 | 4.180 | 1.185 | – |
| TRAIN_MEAN | 0.017575 | 0.011762 | 4.172 | 1.176 | 56.2 |
| ALWAYS_UP | 0.017615 | 0.011857 | 4.180 | 1.185 | 56.2 |

- The model hits the correct direction on 347 of 623 days. Always guessing "up" would hit 56.2%. The one-sided binomial
  test against that rate gives **p = 0.61**, so there is no evidence of directional skill.
- The model is not better than the training-mean baseline on RMSE or MAE (the differences are in the fourth decimal).
- The training and validation losses stay at the "predict the mean" level (about 1.0 after standardization) and
  training stops early, which confirms the model finds nothing to learn.
- The price error (MAPE about 1.18%) looks small, but it is equal to the "tomorrow = today" baseline: it reflects how
  little the price moves in a day, not forecasting skill.
- Criterion S4 is met only in the weak sense of "not worse than the baseline"; S5 shows no significant skill.
- The error analysis (section 6) shows why the 55.7% is not skill: the model predicts "up" on 97.8% of days, so its
  accuracy is simply the share of up days in the test set.

### 5.3 Plots

`outputs/<run>/loss_curve.png` (train and validation loss), `price_prediction.png` and `return_scatter.png` (return run),
`volatility_prediction.png` (volatility run).

## 6. Error analysis and limitations

Produced by `python -m src.error_analysis --ticker AAPL` from the stored `test_predictions.csv` files (no retraining).
Full tables are in `outputs/AAPL_error_analysis/error_analysis.json`, with two plots (`volatility_errors.png`,
`return_errors.png`).

### 6.1 Volatility model

| Actual volatility level (test quintile) | n | mean actual | mean predicted | bias | RMSE model | RMSE train mean |
|---|---|---|---|---|---|---|
| Q1 (calmest) | 125 | -5.11 | -4.36 | +0.74 | 0.806 | 0.816 |
| Q2 | 125 | -4.63 | -4.33 | +0.30 | 0.379 | 0.308 |
| Q3 | 124 | -4.34 | -4.26 | +0.08 | 0.260 | 0.074 |
| Q4 | 125 | -4.09 | -4.28 | -0.19 | 0.256 | 0.258 |
| Q5 (most volatile) | 125 | -3.62 | -4.12 | -0.50 | 0.637 | 0.779 |

| Year | n | bias | RMSE model | RMSE train mean |
|---|---|---|---|---|
| 2024 (from April) | 175 | +0.02 | 0.454 | 0.440 |
| 2025 | 250 | +0.13 | 0.582 | 0.640 |
| 2026 (to October) | 199 | +0.09 | 0.479 | 0.468 |

Findings:
- **The predictions are compressed toward the mean.** Their standard deviation is 0.25 versus 0.54 for the actual
  values. The model overestimates volatility on calm days (bias +0.74 in the calmest quintile) and underestimates it on
  turbulent days (bias -0.50 in the most volatile quintile).
- **The gain over the baseline comes mainly from the turbulent days.** The model is better than the training mean in the
  most volatile quintile (0.637 vs 0.779), about equal in the calmest quintile and in Q4, and *worse* than a constant in
  Q2 and Q3 (the middle of the range), where the constant is already almost exact.
- **The gain is concentrated in one year.** The model beats the constant in 2025 only (0.582 vs 0.640), and is slightly
  worse in 2024 and 2026 (0.454 vs 0.440, 0.479 vs 0.468). Seven of the ten largest misses are in 2025: the sharp
  market move around 2025-04-01/02 was under-predicted by 1.4-1.8 log units, and several quiet days in 2025 were
  over-predicted by about 1.4. The overall advantage (RMSE 0.516 vs 0.537) therefore rests on how the model reacts to
  one volatile episode, and should not be read as a stable skill.
- There is no level shift between training and test (mean log-volatility -4.34 vs -4.36), and the overall bias is
  small (+0.09).

### 6.2 Return model

| Metric (test set) | Value |
|---|---|
| Standard deviation of predictions / of actual returns | 0.00057 / 0.0176 |
| Correlation between predicted and actual return | -0.02 |
| Share of days predicted "up" | 97.8% |

| Year | directional accuracy % | share of up days % |
|---|---|---|
| 2024 | 63.4 | 63.4 |
| 2025 | 51.6 | 52.4 |
| 2026 | 53.7 | 53.7 |

Findings:
- **The model is effectively a constant "up" predictor.** Its directional accuracy per year equals the share of up
  days in that year (to within one day), its correlation with the true return is zero, and its predictions vary by
  0.06% against real moves of 1.8%. The aggregate 55.7% is therefore the base rate of up days, not forecasting ability.
- **It does not anticipate large moves.** The ten largest absolute moves in the test set (for example +14.3% on
  2025-04-08 and -9.7% on 2025-04-02) all received predictions between -0.06% and +0.14%.
- The error equals that of the "zero" baseline in every market regime (calm to turbulent) and for every
  prediction-size quartile, so no subgroup shows a hidden skill. Directional accuracy by prediction size
  (51-60%) shows no monotone pattern.

### 6.3 Limitations

- A single chronological split and a single seed: the numbers have no confidence interval. The volatility gain is
  concentrated in one year, so it could change with another split. Walk-forward validation and multiple seeds are needed.
- One asset (AAPL) in the reference experiment.
- Only price and volume inputs. No market-wide, macro or news information.
- No transaction costs or trading rule: the results say nothing about profitability.
- The test period is about 2.5 years; extreme events are few.
- The error analysis is descriptive: the explanations above (for example the effect of one volatile episode) are read
  from the tables and were not isolated by a separate experiment.

## 7. Response time and failure behavior

Automated tests: `pytest tests` (21 tests, all passing; offline, about 3 seconds).

### 7.1 Response time

Measured on the development machine (Windows 11, CPU only, PyTorch 2.14), with the trained AAPL models and the cached
data. The Yahoo Finance download is mocked in the end-to-end test, so network time is not included.

| Measure | Return model | Volatility model |
|---|---|---|
| Model forward pass, one window (median of 200 calls; p95) | 0.34 ms (p95 0.43 ms) | 0.34 ms (p95 0.47 ms) |
| `predict` end to end without network (load checkpoint, features, scaling, forward pass, print) | 35 ms | 31 ms |

In real use `predict` also downloads fresh prices from Yahoo Finance first, so its latency is dominated by that
request. That time was **not measured**, since it depends on the network and on the external service. The training
time was also not measured.

### 7.2 Failure behavior

Each case below is covered by a test in `tests/test_failures.py`.

| Situation | Behavior |
|---|---|
| Unknown ticker (empty download) | `RuntimeError: No data returned for ticker ...`; no cache file is written |
| Network failure or download exception | `RuntimeError: Could not download data for ... : <reason>` |
| Network failure during `predict` | warning, and the forecast is made from the cached data. With no cache either, the error above |
| Cached data present, no network | `train` works offline |
| `predict` without a trained model | exits with "No trained model at ... Run `python main.py train` first." |
| Corrupted checkpoint file | exits with "Could not load the model ... Re-train it with `python main.py train`." |
| Invalid command-line arguments | rejected by the argument parser (exit code 2) |
| Too little history (10, 45 or 60 rows) | `ValueError: Not enough ...`, with the number of windows and a suggested fix |
| Constant prices (RSI undefined, every row dropped) | `ValueError: Not enough usable history ...` |
| Missing values in the raw data | rows are dropped; no NaN or infinity reaches the model |
| Untrained / garbage model | still prints a finite prediction; the CLI does not crash |

**Found and fixed while writing the tests.** Before this change, a short history failed with an internal message
("window shape cannot be larger than input array shape" or a scikit-learn error); 60 rows silently gave 1 training
sample and no validation or test samples; a download exception from `yfinance` was not wrapped; `predict` always
re-downloaded and crashed without a connection even though a cache existed; and a corrupted checkpoint raised a raw
`torch.load` exception. The fixes are in `src/data.py`, `src/dataset.py` and `main.py`.

**Not covered:** a corrupted CSV cache file; a Yahoo response with unexpected columns; and a model output that is wrong
but finite (for example a prediction of -50% in one day), which the system does not detect or flag.

## 8. Improvement directions

1. Feature ablation (cumulative groups and leave-one-out), repeated over several seeds and compared by mean ± standard
   deviation, as described in the README.
2. A GARCH(1,1) baseline for volatility, and a gradient-boosting model on lagged features.
3. Walk-forward validation instead of a single split.
4. Additional inputs: VIX, interest rates, sector indices, news sentiment.
5. More assets, to check whether the volatility result generalizes.
6. A simple UI or API on top of the CLI.

## 9. Reproducibility

```bash
pip install -r requirements.txt
python main.py train --ticker AAPL      # trains both targets and writes outputs/
python main.py predict --ticker AAPL
```

Training is seeded (42), so the same data and configuration give the same results on the same machine. Dependencies are
in `requirements.txt`; no API keys are used.

## 10. Conclusions

- The pipeline is leakage-safe, reproducible and runnable end to end, with automated tests of its failure behavior
  and a measured response time of about 35 ms per prediction (excluding the network download).
- The LSTM beats both baselines for volatility on the held-out test set, but the gain is small and concentrated in
  the turbulent 2025 period; its predictions are compressed toward the mean. It is a weak signal, not a stable skill.
- For next-day returns it does not outperform trivial baselines: it predicts "up" on 97.8% of days, and the statistical test supports the
  conclusion that the inputs carry no usable directional information.
- Honest reporting of the negative result, together with the baselines, is the main contribution; the next steps are to
  strengthen the evidence (seeds, walk-forward, stronger baselines) and to complete the remaining items (corrupted cache, output sanity check, a GARCH baseline).
