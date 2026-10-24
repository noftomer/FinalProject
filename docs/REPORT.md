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
| Volatility | RMSE 0.516 vs 0.537 (train mean) and 0.672 (trailing 5-day); correlation 0.34; R² 0.076 | beats both baselines, modest gain |
| Return | RMSE equals the baselines; directional accuracy 55.7% vs 56.2% for always guessing "up" (p = 0.61) | no detectable signal |

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
  signal in the validation period. On the test set the gain is smaller (R² 0.076), probably because the volatility level
  in 2024-2026 differs from the training period. This is a hypothesis; it was not tested separately.
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

### 5.3 Plots

`outputs/<run>/loss_curve.png` (train and validation loss), `price_prediction.png` and `return_scatter.png` (return run),
`volatility_prediction.png` (volatility run).

## 6. Error analysis and limitations

**Observed from the results**
- Return predictions have almost no spread: the model learns to stay close to the mean, so it cannot separate up days
  from down days (standard deviation of predicted next-day returns 0.0005, versus 0.0176 for the actual returns).
- The volatility model improves on the baseline only modestly, and its gain on the test period is much lower than on
  the validation period, which indicates sensitivity to a shift in the volatility regime.

**Limitations**
- A single chronological split and a single seed: the numbers have no confidence interval, and part of the small gain on
  volatility could be due to chance in the initialization. Walk-forward validation and multiple seeds are needed.
- One asset (AAPL) in the reference experiment.
- Only price and volume inputs. No market-wide, macro or news information.
- No transaction costs or trading rule: the results say nothing about profitability.
- The test period is one market regime; extreme events (crashes) are rare and weakly represented.

**To be added (planned, not yet done):** a quantitative error breakdown from `test_predictions.csv` by period and by
volatility level (for example, error on the highest-volatility decile versus the rest), and examples of the largest
misses.

## 7. Response time and failure behavior

**Code-level behavior (read from the source, not yet tested)**
- Unknown ticker or empty download: `RuntimeError("No data returned for ticker …")` in `src/data.py`.
- `predict` without a trained model: exits with "No trained model at … Run `python main.py train` first."
- Invalid `--target` value: rejected by the argument parser.
- Cached data lets `train` and `predict` run without network access, as long as the cache file exists.

**Not yet covered:** measured latency of `predict` and `train`; behavior when Yahoo Finance is unreachable on a first
run; a history too short for the 40-day window; corrupted cache or checkpoint files. These are planned as an
automated test suite (`tests/`) and a timing measurement, and the results will be added here.

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

- The pipeline is leakage-safe, reproducible and runnable end to end.
- The LSTM extracts a small but real signal for volatility (better than both baselines on the held-out test set).
- For next-day returns it does not outperform trivial baselines, and the statistical test supports the conclusion that
  the inputs carry no usable directional information.
- Honest reporting of the negative result, together with the baselines, is the main contribution; the next steps are to
  strengthen the evidence (seeds, walk-forward, stronger baselines) and to complete the failure and latency tests.
