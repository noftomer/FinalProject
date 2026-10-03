# Next-Day Stock Market Prediction (Ridge Regression)

An academic project that forecasts the **next trading day's return and closing price** of a stock or index
with a deliberately simple model: a Ridge regression on the last 5 daily returns, evaluated rigorously
against naive baselines.

> Academic use only. This is not financial advice.

## 1. Research question

*Can a simple model trained on recent daily returns predict the next-day movement of a financial market
better than naive forecasts?*

The Efficient Market Hypothesis (EMH, weak form) predicts it cannot. This project tests that claim
empirically with a leak-free methodology.

## 2. Project structure

```
FinalProject/
├── main.py              # CLI: train | predict
├── requirements.txt
├── src/
│   ├── config.py        # all settings (single source of truth)
│   ├── data.py          # Yahoo Finance download + CSV cache
│   ├── features.py      # lagged daily returns + next-day target
│   ├── dataset.py       # chronological split and scaling
│   ├── models.py        # Ridge regression
│   ├── train.py         # picks the regularization strength on validation data
│   └── evaluate.py      # metrics, baselines, significance test, backtest, plots
├── data/                # cached price CSVs (auto-created)
└── outputs/<TICKER>_ridge/
    ├── results.json          # all metrics and model coefficients
    ├── test_predictions.csv  # per-day predictions
    ├── model.joblib          # model + scaler + config
    └── *.png                 # price plot, scatter, backtest
```

## 3. Quick start

```bash
pip install -r requirements.txt
python main.py train --ticker ^GSPC
python main.py predict --ticker ^GSPC
```

Any Yahoo Finance ticker works (`AAPL`, `MSFT`, `^IXIC`, `TA35.TA`, `BTC-USD`, …).
Other options: `--start`, `--end`, `--lags` (number of past daily returns used, default 5), `--refresh`.

## 4. Methodology

### 4.1 Data
Daily adjusted OHLCV data from Yahoo Finance (2010 until today by default).

### 4.2 Target
The model predicts the **next-day log return** `r(t+1) = ln(Close(t+1) / Close(t))`, not the raw price.
Raw prices are non-stationary, while returns are roughly stationary. The predicted price is rebuilt as
`Close(t) · exp(r̂)`.

### 4.3 Features
The last 5 daily log returns (`ret_lag0` = today, `ret_lag1` = yesterday, …), standardized.

### 4.4 Leakage prevention
- **Chronological split** into 70% train, 15% validation, and 15% test, with no shuffling across time.
- The **scaler is fit on training data only**.
- Features at *t* use only data up to and including *t*. Only the label looks ahead.
- The regularization strength (alpha) is chosen on the **validation** set. The test set is used once, at the end.

### 4.5 Model
Ridge regression (linear regression with an L2 penalty). Alpha is picked from a small grid in
`src/config.py` by validation error. There is no random initialization, so results are exactly reproducible.

### 4.6 Evaluation
1. **Error metrics** on returns (RMSE, MAE) and on prices (RMSE, MAPE).
2. **Directional accuracy**, meaning the share of days where the sign of the predicted return was correct.
3. **Baselines**: random walk (predict a 0% return, so tomorrow's price equals today's), historical mean drift,
   and momentum (tomorrow's return equals today's).
4. A **binomial significance test** of whether directional accuracy beats always guessing the majority class ("up").
5. A **trading backtest**: go long when the predicted return is positive and stay flat otherwise, with 1 bp
   transaction cost. It reports total return, Sharpe ratio, and max drawdown compared with buy & hold.

## 5. Results (S&P 500, test period Mar 2024 – Sep 2026, 630 days according to sep 2026)

| Model | Return RMSE | Price MAPE % | Direction acc. % | p-value | Strategy Sharpe |
|---|---|---|---|---|---|
| Ridge | 0.00978 | 0.662 | 55.7 | 0.52 | 1.07 |
| Random walk | 0.00980 | 0.665 | n/a | – | – |
| Mean drift ("always up") | 0.00978 | 0.662 | 55.7 | – | – |
| Buy & hold | – | – | – | – | 1.07 |

### 5.1 Discussion
- **The price chart can mislead.** The predicted-price line tracks the actual price closely, but the random
  walk ("tomorrow equals today") gets almost the same MAPE. The fit comes from anchoring each forecast to
  today's price, not from learning.
- Validation picked the strongest regularization in the grid (alpha = 100,000). The coefficients are all near
  zero, so the model predicts roughly the training mean return every day. In other words, the past 5 daily
  returns carry no usable signal, and the model collapses to the "always up" baseline.
- Its direction accuracy equals the majority-class rate (p > 0.05), and the strategy is in the market nearly
  every day, so it matches buy & hold.
- **Conclusion:** the results are consistent with the weak-form EMH. This negative result is a valid
  scientific finding, and it shows why honest baselines matter.

## 6. Limitations and future work
- A single train/validation/test split was used. **Walk-forward (rolling) validation** would give more robust estimates.
- Try more features (technical indicators, VIX, interest rates, news sentiment) or a non-linear model.
- Reframe the task as **classification** (up or down) or as **probabilistic forecasting** to model uncertainty.

## 7. References
- Fama (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance.
