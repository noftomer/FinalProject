# Project Plan: Stock Return and Volatility Forecasting with an LSTM

> Academic use only. This is not financial advice.

## 1. Problem definition

**Need.** Investors and risk managers want to anticipate the movement of a stock. Two questions are asked:

1. *What will the price do tomorrow?* (direction and size of the next-day return)
2. *How turbulent will the next week be?* (realized volatility, which drives risk, position sizing and option pricing)

**Problem statement.** Given the last `seq_len` = 40 trading days of daily market data for a ticker (default: AAPL),
predict:

| Target | Definition |
|---|---|
| `return` | next-day log return `r(t+1) = ln(Close(t+1) / Close(t))`; the price is rebuilt as `Close(t) · exp(r̂)` |
| `volatility` | log of realized volatility (RMS of daily log returns) over the next 5 trading days |

**Why two targets.** Financial theory (weak-form market efficiency) says next-day returns are close to unpredictable,
while volatility clusters in time and is known to be partly predictable. Training both lets the project test the model
on a hard target and on a feasible one, and makes the comparison itself a finding.

## 2. Technological options considered

| Option | Pros | Cons | Decision |
|---|---|---|---|
| Naive / statistical (zero change, historical mean, GARCH, ARIMA) | simple, strong on noisy data | no learned feature interaction | used as **baselines** (naive ones); GARCH/ARIMA left as future work |
| Tree models (XGBoost, Random Forest) on lag features | fast, robust on tabular data | the time structure has to be hand-encoded | future work |
| **LSTM (PyTorch)** | natively models sequences, course topic | easy to overfit on noisy data, needs a careful split | **chosen** |
| Transformer / large pre-trained time-series models | state of the art on some benchmarks | heavy, data-hungry, hard to explain | out of MVP scope |

**Rationale for the LSTM.** It fits the sequential nature of the data and the course material, and a small network
(32 hidden units, 1 layer) is appropriate for ~4,200 daily rows. The conclusions are drawn against explicit baselines,
so a negative result is still a valid finding.

## 3. Data

- **Source:** Yahoo Finance (`yfinance`), daily adjusted OHLCV, from 2010-01-01 until today. Real data, no synthetic data.
- **Management:** downloaded once and cached as CSV in `data/` (`--refresh` forces a re-download). No API key is needed.
- **Split:** chronological, 70% train / 15% validation / 15% test, with no shuffling across time.
- **Features (7 per day):** log return, change in log volume, RSI(14), MACD histogram (price-normalized), 10-day
  volatility, intraday high-low range, gap from the 20-day moving average. Standardized with training-set statistics only.
- **Leakage controls:** a window uses only data up to day *t*; only the label looks ahead; scaler and target statistics
  come from the training set; the test set is used once, at the end.

## 4. MVP scope

**In scope**
- Single-ticker pipeline (AAPL as the reference experiment, any Yahoo ticker runnable).
- Both targets (`return`, `volatility`), one LSTM architecture, one fixed configuration in `src/config.py`.
- CLI with `train` and `predict`, and Windows `.bat` launchers.
- Evaluation on a held-out test period against baselines, plus plots and saved metrics.

**Out of scope for the MVP** (listed as future work): multi-asset or portfolio models, news / macro inputs,
walk-forward validation, hyper-parameter search, a web UI, trading strategy and transaction costs.

## 5. Measurable success criteria

Evaluated on the held-out test set (the last 15% of days).

| # | Criterion | Target |
|---|---|---|
| S1 | Volatility model beats the **TRAIN_MEAN** baseline | RMSE lower, `R²_vs_train_mean` > 0 |
| S2 | Volatility model beats the **TRAILING_5D** baseline (volatility of the past 5 days) | RMSE lower |
| S3 | Volatility predictions are informative | correlation with actual > 0.25 |
| S4 | Return model price error is no worse than the **ZERO** (tomorrow = today) baseline | price RMSE ≤ baseline |
| S5 | Directional accuracy is judged **against always guessing the majority class**, with a binomial test | report hits, n, p-value; significant only if p < 0.05 |
| S6 | Reproducibility | fixed seed; a fresh clone follows the README and reproduces the results |
| S7 | Robustness | invalid input, missing model and a data-service failure give a clear message, not a stack trace (to be verified, see the report) |
| S8 | Response time | `predict` latency measured and reported (to be verified, see the report) |

An honest "no signal" outcome on the return target is an acceptable result, provided it is demonstrated with a
statistical test rather than only observed.

## 6. Evaluation plan

**Scenarios / data.** The held-out test period of AAPL (about 2.5 years, ~620 trading days), which includes regimes the
model did not see in training. The pipeline is also runnable on other tickers (e.g. MSFT) as a generalization check.

**Baselines.**

| Target | Baselines |
|---|---|
| `return` | `ZERO` (tomorrow = today), `TRAIN_MEAN`, `ALWAYS_UP` (majority class) |
| `volatility` | `TRAIN_MEAN`, `TRAILING_5D` |

**Metrics.** RMSE and MAE (in return, price and log-volatility space), MAPE for prices, `R²` versus the training mean,
correlation, directional accuracy with a binomial test.

**Additional checks.** Error analysis by period and volatility level; response-time measurement; failure-mode tests
(invalid ticker, no network, missing model, too-short history); feature ablation with several random seeds.

## 7. Work plan

| Stage | Output | Status |
|---|---|---|
| 1. Problem analysis and options | this document | done |
| 2. Data collection and caching | `src/data.py` | done |
| 3. Features and leakage-safe windows / split | `src/features.py`, `src/dataset.py` | done |
| 4. Model and training loop | `src/models.py`, `src/train.py` | done |
| 5. Evaluation against baselines | `src/evaluate.py`, `outputs/` | done |
| 6. Runnable system (CLI and launchers) | `main.py`, `*.bat` | done |
| 7. Error analysis, latency and failure tests | to be added to the report | planned |
| 8. Final report | `docs/REPORT.md` | in progress |
| 9. Ablation study | future work | optional |

## 8. Risks

| Risk | Mitigation |
|---|---|
| Returns are close to noise, so the model may learn nothing | baselines and a significance test, so a null result is still reported correctly |
| Overfitting on a small dataset | small network, dropout, weight decay, early stopping on validation loss |
| Look-ahead leakage | chronological split, training-only scaling, labels shifted separately from features |
| Yahoo Finance unavailable or changed | local CSV cache; clear error message |
| Run-to-run variance of LSTM training | fixed seed; multi-seed comparison planned for the ablation |
