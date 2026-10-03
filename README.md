# Next-Day Stock Price Prediction (PyTorch LSTM)

Forecasts the **next trading day's return and closing price** of a stock or index with an LSTM
(PyTorch) trained on the last 20 days of returns, volume and technical indicators, and evaluates it on a held-out test period.

> Academic use only. This is not financial advice.

## 1. Project structure

```
FinalProject/
├── main.py              # CLI: train | predict
├── requirements.txt
├── src/
│   ├── config.py        # all settings (single source of truth)
│   ├── data.py          # Yahoo Finance download + CSV cache
│   ├── features.py      # returns, volume, RSI, MACD, ... + next-day target
│   ├── dataset.py       # sliding windows, chronological split and scaling
│   ├── models.py        # LSTM regressor (PyTorch)
│   ├── train.py         # Adam training loop with early stopping on validation loss
│   └── evaluate.py      # metrics, significance test, plots
├── data/                # cached price CSVs (auto-created)
└── outputs/<TICKER>_lstm/
    ├── results.json          # all metrics and train/val loss history
    ├── test_predictions.csv  # per-day predictions
    ├── model.pt              # weights + scaler + config
    └── *.png                 # train/val loss curve, price plot, scatter
```

## 2. Quick start

```bash
pip install -r requirements.txt
python main.py train --ticker ^GSPC
python main.py predict --ticker ^GSPC
```

Any Yahoo Finance ticker works (`AAPL`, `MSFT`, `^IXIC`, `TA35.TA`, `BTC-USD`, …).
Other options: `--start`, `--end`, `--seq-len` (input window length, default 20), `--epochs`, `--refresh`.
Model and training hyper-parameters (hidden size, layers, dropout, learning rate, …) are in `src/config.py`.

## 3. Methodology

### 3.1 Data
Daily adjusted OHLCV data from Yahoo Finance (2010 until today by default).

### 3.2 Target
The model predicts the **next-day log return** `r(t+1) = ln(Close(t+1) / Close(t))`. The predicted price is
rebuilt as `Close(t) · exp(r̂)`, so the next business day's price is the last close times the forecast change.

### 3.3 Input
A sliding window of the last `seq_len` (20) days, with 7 features per day, standardized with training-set statistics:
daily log return, change in log volume, RSI(14), MACD histogram (price-normalized), 10-day volatility,
intraday high-low range, and the gap from the 20-day moving average.

### 3.4 Leakage prevention
- **Chronological split** into 70% train, 15% validation, and 15% test, with no shuffling across time.
- The input scaler is fit on training returns only. The target's mean/std come from the training targets only.
- A window uses only data up to and including day *t*. Only the label looks ahead.
- Early stopping and model selection use the **validation** set. The test set is used once, at the end.

### 3.5 Model
`nn.LSTM(7 → 64, 2 layers)` → dropout → linear layer, trained with MSE loss, Adam, gradient clipping, and early
stopping (patience 10). The best-validation epoch's weights are kept. A fixed seed makes runs reproducible.

### 3.6 Evaluation
1. **Error metrics** on returns (RMSE, MAE) and on prices (RMSE, MAPE).
2. **Directional accuracy**: the share of days where the sign of the predicted return was correct.
3. A **binomial test** of whether directional accuracy beats always guessing the majority class.

## 4. Results (S&P 500, test period Apr 2024 – Oct 2026, 626 days)

| Model | Return RMSE | Price MAPE % | Direction acc. % |
|---|---|---|---|
| LSTM | 0.00976 | 0.665 | 51.0 |

Direction accuracy (51.0%, p = 0.99 on the binomial test) is not better than always guessing the majority class
("up"). Next-day prices are dominated by today's price, so the predicted-price line tracks the actual one
closely whatever the model learns.

## 5. Limitations and future work
- A single train/validation/test split was used. Walk-forward validation would be more robust.
- Volume and technical indicators with a larger LSTM did not clearly improve results. Other inputs (VIX, rates, news) or architectures could still be tried.
