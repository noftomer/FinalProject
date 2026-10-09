# Stock Volatility / Return Prediction (PyTorch LSTM)

Forecasts the **realized volatility over the next 5 trading days** (default) or the **next-day return** of a stock or
index with an LSTM (PyTorch) trained on the last 40 days of returns, volume and technical indicators, and evaluates it
on a held-out test period.

> Academic use only. This is not financial advice.

## 1. Project structure

```
FinalProject/
├── main.py              # CLI: train | predict
├── requirements.txt
├── run.bat              # Windows launcher: creates .venv, installs deps, runs main.py
├── train_aapl.bat, predict_aapl.bat, ...   # ready-made shortcuts that call run.bat
├── src/
│   ├── config.py        # all settings (single source of truth)
│   ├── data.py          # Yahoo Finance download + CSV cache
│   ├── features.py      # returns, volume, RSI, MACD, ... + target (volatility or return)
│   ├── dataset.py       # sliding windows, chronological split and scaling
│   ├── models.py        # LSTM regressor (PyTorch)
│   ├── train.py         # Adam training loop with early stopping on validation loss
│   └── evaluate.py      # metrics, significance test, plots
├── data/                # cached price CSVs (auto-created)
└── outputs/<TICKER>_lstm_vol/   (volatility target; <TICKER>_lstm/ for the return target)
    ├── results.json          # all metrics and train/val loss history
    ├── test_predictions.csv  # per-day predictions
    ├── model.pt              # weights + scaler + config
    └── *.png                 # train/val loss curve, price plot, scatter
```

## 2. Quick start

```bash
pip install -r requirements.txt
python main.py train --ticker AAPL                      # default: volatility target
python main.py predict --ticker AAPL
python main.py train --ticker AAPL --target return      # next-day return instead
```

### Windows (.bat launchers)

`run.bat` creates a `.venv` on first use, installs `requirements.txt` (again only when it changes) and forwards all
arguments to `main.py`:

```bat
run.bat                                       :: train AAPL (volatility target)
run.bat predict --ticker AAPL
run.bat train --ticker MSFT --target return --epochs 50
```

Double-clickable shortcuts (each calls `run.bat` and pauses at the end):

| File | Runs |
|---|---|
| `train_aapl.bat` | `run.bat` (default) |
| `predict_aapl.bat` | `run.bat predict --ticker AAPL` |
| `train_msft.bat` | `run.bat train --ticker MSFT` |
| `train_aapl_return.bat` | `run.bat train --ticker AAPL --target return` |
| `train_aapl_epochs50.bat` | `run.bat train --ticker AAPL --epochs 50 --seq-len 60` |
| `train_aapl_refresh.bat` | `run.bat train --ticker AAPL --refresh` |

`predict` needs a trained model, so run a train shortcut first.

Any Yahoo Finance ticker works (`AAPL`, `MSFT`, `^IXIC`, `TA35.TA`, `BTC-USD`, …).
Other options: `--start`, `--end`, `--target` (`volatility` default, or `return`), `--seq-len` (input window length, default 40), `--epochs`, `--refresh`.
Model and training hyper-parameters (hidden size, layers, dropout, learning rate, …) are in `src/config.py`.

## 3. Methodology

### 3.1 Data
Daily adjusted OHLCV data from Yahoo Finance (2010 until today by default).

### 3.2 Target
- **`volatility` (default):** the log of realized volatility (RMS of daily log returns) over the next 5 trading days.
  Volatility clusters in time, so unlike returns it is partly predictable.
- **`return`:** the next-day log return `r(t+1) = ln(Close(t+1) / Close(t))`; the predicted price is rebuilt as
  `Close(t) · exp(r̂)`. Daily returns are close to noise, so the model learns almost nothing here (see Results).

### 3.3 Input
A sliding window of the last `seq_len` (40) days, with 7 features per day, standardized with training-set statistics:
daily log return, change in log volume, RSI(14), MACD histogram (price-normalized), 10-day volatility,
intraday high-low range, and the gap from the 20-day moving average.

### 3.4 Leakage prevention
- **Chronological split** into 70% train, 15% validation, and 15% test, with no shuffling across time.
- The input scaler is fit on training returns only. The target's mean/std come from the training targets only.
- A window uses only data up to and including day *t*. Only the label looks ahead.
- Early stopping and model selection use the **validation** set. The test set is used once, at the end.

### 3.5 Model
`nn.LSTM(7 → 32, 1 layer)` → dropout → linear layer, trained with Huber loss (early stopping on validation MSE), Adam
with weight decay, gradient clipping, and early stopping (patience 10). The best-validation epoch's weights are kept. A fixed seed makes runs reproducible.

### 3.6 Evaluation
**Volatility target:** RMSE, MAE, R² versus the training-mean prediction, and correlation, compared against the
`TRAIN_MEAN` and `TRAILING_5D` (volatility of the past 5 days) baselines.

**Return target:** error metrics on returns and prices, directional accuracy, and a binomial test of whether it beats
always guessing the majority class.

## 4. Results (AAPL)

**Volatility target** (test set, log-volatility space):

| | RMSE | MAE | R² vs train mean | corr |
|---|---|---|---|---|
| MODEL | 0.516 | 0.403 | 0.064 | 0.32 |
| TRAIN_MEAN | 0.533 | 0.414 | 0.000 | – |
| TRAILING_5D | 0.681 | 0.534 | -0.633 | 0.19 |

Validation loss reaches 0.48 against a 0.82 baseline, so the model learns a real signal and beats both baselines on the
test set. The test R² is small (0.064), likely because the volatility level differs between the training and test
periods, so the gain is modest.

**Return target:** validation loss stays at the "predict the training mean" baseline (0.984 vs 0.985), test RMSE equals
the zero/mean baselines, and directional accuracy (55.3%) is below always guessing "up" (55.8%; binomial p = 0.61).
Next-day returns are essentially unpredictable with these inputs.

## 5. Limitations and future work
- A single train/validation/test split was used. Walk-forward validation would be more robust.
- Volume and technical indicators with a larger LSTM did not clearly improve results. Other inputs (VIX, rates, news) or architectures could still be tried.
