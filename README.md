# Next-Day Stock Market Prediction with Deep Learning (PyTorch)

An academic project that forecasts the **next trading day's return and closing price** of a stock or index
using an LSTM (Long Short-Term Memory) recurrent neural network, evaluated rigorously
against naive baselines.

> Academic use only. This is not financial advice.

---

## 1. Research question

*Can a deep sequence model trained on historical price and technical-indicator data predict the
next-day movement of a financial market better than naive forecasts?*

The Efficient Market Hypothesis (EMH, weak form) predicts they cannot. This project tests that claim
empirically with a leak-free methodology.

## 2. Project structure

```
FinalProject/
├── main.py              # CLI: train | predict | ablation
├── requirements.txt
├── src/
│   ├── config.py        # all hyper-parameters (single source of truth)
│   ├── data.py          # Yahoo Finance download + CSV cache
│   ├── features.py      # technical indicators + next-day target
│   ├── dataset.py       # chronological split, scaling, sliding windows
│   ├── models.py        # LSTM regressor
│   ├── train.py         # training loop, early stopping, seeding
│   ├── evaluate.py      # metrics, baselines, significance test, backtest, plots
│   └── ablation.py      # feature-set ablation across multiple seeds
├── data/                # cached price CSVs (auto-created)
└── outputs/
    ├── <TICKER>_lstm/
    │   ├── results.json          # all metrics
    │   ├── test_predictions.csv  # per-day predictions
    │   ├── model.pt              # weights + scaler + config
    │   └── *.png                 # loss curve, price plot, scatter, backtest
    └── ablation/
        ├── ablation_runs.csv     # one row per (variant, seed)
        ├── ablation_summary.csv  # mean and std per variant
        └── ablation.png          # directional accuracy and Sharpe per variant
```

## 3. Quick start

```bash
pip install -r requirements.txt
```

```bash
python main.py train --ticker ^GSPC
```

```bash
python main.py predict --ticker ^GSPC
```

```bash
python main.py ablation --ticker ^GSPC --seeds 5
```

Any Yahoo Finance ticker works (`AAPL`, `MSFT`, `^IXIC`, `TA35.TA`, `BTC-USD`, …).
Other options: `--start`, `--end`, `--window`, `--epochs`, `--hidden`, `--layers`, `--lr`, `--seed`, `--refresh`,
and `--seeds` (number of seeds per variant, used by `ablation`).

## 4. Methodology

### 4.1 Data
Daily adjusted OHLCV data from Yahoo Finance, 2010 until today by default (≈4,100 trading days for the S&P 500).

### 4.2 Target
The model predicts the **next-day log return** `r(t+1) = ln(Close(t+1) / Close(t))`, not the raw price.
Raw prices are non-stationary: a model trained on prices from 2010 to 2020 has never seen the price levels
of 2025. Returns are roughly stationary, so the model can generalize. The predicted price is rebuilt as
`Close(t) · exp(r̂)`.

### 4.3 Features (16, all computable at the close of day *t*)
| Group | Features |
|---|---|
| Momentum | 1-, 5-, 20-day log returns; RSI(14); MACD and signal line |
| Trend | price / SMA(10), SMA(50), EMA(12), EMA(26) ratios |
| Volatility | 20-day return std; Bollinger band position; high-low range |
| Intraday / volume | open-to-close change; log volume change |
| Calendar | day of week |

### 4.4 Leakage prevention
Leakage is the most common flaw in student stock-prediction projects. This project avoids it as follows:
- **Chronological split** into 70% train, 15% validation, and 15% test, with no shuffling across time.
- The **scaler is fit on training data only**.
- Features at *t* use only data up to and including *t*. Only the label looks ahead.
- Early stopping uses the **validation** set. The test set is used once, at the end.

### 4.5 Model
The model takes a 30-day window of features `(30 × 16)` and outputs a single scalar.
- **LSTM**: 2 layers, 64 hidden units, LayerNorm and a linear head.

Training setup: AdamW optimizer, **Huber loss** (robust to fat-tailed return distributions), gradient clipping,
ReduceLROnPlateau, and early stopping (patience 15). Seeds are fixed for reproducibility.

### 4.6 Evaluation
1. **Error metrics** on returns (RMSE, MAE) and on prices (RMSE, MAPE).
2. **Directional accuracy**, meaning the share of days where the sign of the predicted return was correct.
3. **Baselines**: random walk (predict a 0% return, so tomorrow's price equals today's), historical mean drift,
   and momentum (tomorrow's return equals today's).
4. A **binomial significance test** of whether directional accuracy beats always guessing the majority class ("up").
5. A **trading backtest**: go long when the predicted return is positive and stay flat otherwise, with 1 bp
   transaction cost. It reports total return, Sharpe ratio, and max drawdown compared with buy & hold.

## 5. Results (S&P 500, test period Apr 2024 – Sep 2026, 623 days)

| Model | Return RMSE | Price MAPE % | Direction acc. % | p-value | Strategy Sharpe |
|---|---|---|---|---|---|
| LSTM | 0.00980 | 0.669 | 50.1 | 0.999 | 0.91 |
| Random walk | 0.00984 | 0.668 | n/a | – | – |
| Mean drift ("always up") | 0.00982 | 0.664 | 56.0 | – | – |
| Buy & hold | – | – | – | – | **1.11** |


### 5.1 Discussion
- **The price chart can mislead.** The predicted-price line tracks the actual price very closely, with a MAPE
  under 0.7%. The random walk, which simply predicts "tomorrow equals today", gets the same MAPE. The good fit
  comes from anchoring each forecast to today's price, not from learning.
- The LSTM improves return RMSE over the random walk by only **about 0.4%**. It does not beat the naive
  "always up" baseline on direction, and its result is not statistically significant (p > 0.05).
- The long/flat strategy, which is out of the market on some days, earns a lower total return and a lower
  Sharpe ratio than buy & hold after costs. Lower drawdown does not mean skill here: it comes from being in the
  market less often.
- **Conclusion:** the results are consistent with the weak-form EMH. Public price history and technical
  indicators contain little exploitable information about the next day's return of a highly liquid index.
  This negative result is a valid scientific finding, and it shows why honest baselines matter.

### 5.2 Feature ablation

`python main.py ablation` tests whether the technical indicators add anything. Each variant is trained with
the same data, split, model, and seeds; only the input feature set changes. Rows with a missing value in any
of the 16 features are dropped for every variant, so all variants share identical samples and test dates.
Results are averaged over several seeds (`--seeds`, default 5, using `seed`, `seed+1`, …) because single runs
differ by more than the effect being measured.

| Variant | Features |
|---|---|
| `returns_only` | 1: daily log return |
| `returns+volume+calendar` | 3: log return, log volume change, day of week |
| `all_16_features` | 16: the full set from section 4.3 |

For each run it reports return RMSE relative to the random walk, directional accuracy against the
"always up" rate, p-value, and strategy Sharpe against buy & hold. Outputs are written to `outputs/ablation/`.

**Committed results** (S&P 500, a single seed, 42; the model stopped after 3 epochs in every variant):

| Variant | RMSE vs random walk | Direction acc. % | p-value | Strategy Sharpe |
|---|---|---|---|---|
| `returns_only` | -0.09% | 55.8 | 0.52 | 1.10 |
| `returns+volume+calendar` | -0.36% | 50.3 | 1.00 | 0.46 |
| `all_16_features` | -0.40% | 55.8 | 0.52 | 1.10 |
| *Reference: "always up" / buy & hold* | – | 55.8 | – | 1.10 |

- Adding features lowers return RMSE only marginally (at most 0.4% below the random walk).
- Where direction accuracy and Sharpe equal the "always up" and buy & hold references, the model is most likely
  predicting a positive return on nearly every day, so it is not showing skill. The 3-feature variant is the
  only one that deviates, and it does worse.
- With one seed the standard deviations are empty and these numbers cannot separate real effects from noise.
  Run with `--seeds 5` or more to get mean ± std before drawing conclusions.

## 6. Limitations and future work
- A single train/validation/test split was used. **Walk-forward (rolling) validation** would give more robust estimates.
- Add exogenous data: VIX, interest rates, sector indices, news or social-media sentiment (for example FinBERT).
- Reframe the task as **classification** (up or down) or as **probabilistic forecasting** (quantile loss) to model uncertainty.
- Run hyper-parameter search (Optuna) and ensembling across seeds.
- Test less efficient markets such as small caps and emerging markets, where predictability may be higher.

## 7. References
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.* Neural Computation.
- Fama (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance.
- Fischer & Krauss (2018). *Deep learning with LSTM networks for financial market predictions.* EJOR.
