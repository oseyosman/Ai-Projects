# Project 10: Stock Price Prediction (Time Series Forecasting)

## 🎯 Goal
Forecast stock price movements and daily forward returns using **Supervised Machine Learning** and **Financial Time Series Econometrics** powered by **Yahoo Finance** market data.

Predicting asset prices is one of the most challenging problems in machine learning due to non-stationarity, low signal-to-noise ratios, and regime shifts. This project demonstrates how to properly formulate financial time series forecasting with strict **lookahead bias prevention**, **stationary return transformations**, and **expanding window cross-validation (`TimeSeriesSplit`)**.

---

## 📈 Technical Indicators & Engineered Features

Raw market prices (Open, High, Low, Close, Volume) are transformed into 16 scale-invariant, stationary technical indicators:

### 1. Momentum & Oscillators
- **Return_1d & Lags**: Daily percentage return ($r_t$) and previous lags ($r_{t-1}, r_{t-2}, r_{t-3}, r_{t-5}$).
- **RSI-14 (Relative Strength Index)**: 14-period momentum oscillator measuring overbought ($>70$) and oversold ($<30$) levels:
  $$\text{RSI} = 100 - \frac{100}{1 + \text{RS}}$$
- **MACD (Moving Average Convergence Divergence)**: Difference between 12-day and 26-day EMAs, along with the 9-day signal line and normalized histogram.

### 2. Trend & Moving Averages
- **Ratio_SMA21 & Ratio_SMA50**: Current price divided by the 21-day and 50-day Simple Moving Averages ($\frac{Close_t}{\text{SMA}}$). Measures distance from key institutional trend supports.

### 3. Volatility & Bands
- **Vol_20**: 20-day rolling annualized realized volatility ($\sigma_{\text{daily}} \times \sqrt{252}$).
- **Bollinger Bands**: 20-day SMA $\pm$ 2 standard deviations. Includes **BB_Width** (volatility expansion/contraction) and **BB_Percent** (relative price position within the bands).
- **Spread_HL & Spread_CO**: Intraday range $\frac{High - Low}{Close}$ and candle body return $\frac{Close - Open}{Open}$.

### 4. Volume Activity
- **Volume_Ratio**: Current session volume divided by the 20-day average volume ($\frac{Volume_t}{\text{Volume}_{\text{SMA20}}}$).

---

## 📁 Project Structure

```
10_stock_price_prediction/
├── README.md               # Project documentation & time series benchmark guide
├── features.py             # Stationary technical indicator calculations & dataset preparation
├── eda.py                  # Exploratory data analysis, return histograms & ASCII trend charts
├── train.py                # 5-fold TimeSeriesSplit cross-validation & model benchmarking
├── predict.py              # Real-time Yahoo Finance CLI predictor, 5-day forecaster & backtester
├── data/
│   └── AAPL_stock_data.csv # 1,509 daily trading sessions (2018–2024 historical manifest)
└── models/
    └── stock_predictor.joblib  # Saved champion pipeline & time series metadata
```

---

## 🚀 Quick Start

### 1. Run Financial Exploratory Data Analysis
Visualizes the historical price trajectory, return distribution, excess kurtosis (fat tails), and technical regimes:
```bash
python3 10_stock_price_prediction/eda.py
```

### 2. Train and Benchmark Time Series Models
Evaluates 7 forecasting architectures across 5-fold expanding window cross-validation (`TimeSeriesSplit`):
```bash
python3 10_stock_price_prediction/train.py
```

### 3. Real-Time Live Stock Forecast (Any Ticker)
Fetches the latest real-time daily data directly from Yahoo Finance and outputs tomorrow's predicted close and quantitative signal:
```bash
# Apple Inc.
python3 10_stock_price_prediction/predict.py --ticker AAPL

# NVIDIA Corp.
python3 10_stock_price_prediction/predict.py --ticker NVDA

# Microsoft Corp.
python3 10_stock_price_prediction/predict.py --ticker MSFT
```

### 4. Multi-Day Forward Trajectory (1 to 10 Days)
Projects iterative forward prices over a multi-day horizon:
```bash
python3 10_stock_price_prediction/predict.py --ticker MSFT --days 5
```

### 5. Run Quantitative Strategy Backtest
Simulates an algorithmic trading strategy (Long when predicted return $> 0$, Cash otherwise) against the Buy & Hold benchmark:
```bash
python3 10_stock_price_prediction/predict.py --backtest
```

### 6. Interactive Trading Console Menu
Launch the interactive terminal wizard:
```bash
python3 10_stock_price_prediction/predict.py
```

---

## 🏆 Model Benchmark & Evaluation Leaderboard

Evaluated on 1,509 daily trading bars (Chronological split: 1,167 train / 292 held-out test sessions):

| Rank | Model Architecture | 5-Fold TS CV RMSE | Held-Out Test RMSE | Test MAE | MAPE (%) | Directional Acc (%) | Key Characteristics |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 🥇 | **Naive Persistence Baseline** | $2.23 | $2.38 | $1.79 | 1.10% | 46.6% | Classic random-walk zero-return benchmark |
| 🥈 | **Random Forest Regressor** | **$2.27** | **$2.43** | **$1.83** | **1.13%** | **52.1%** | 🏆 **Top ML Model**: Best directional classification |
| 🥉 | **Lasso Regression (L1)** | $2.25 | $2.43 | $1.85 | 1.14% | 48.3% | Sparse regularized indicator selection |
| 4 | **Ridge Regression (L2)** | $2.26 | $2.46 | $1.87 | 1.15% | 47.6% | L2 penalized shrinkage on collinear indicators |
| 5 | **Hist Gradient Boosting** | $2.30 | $2.47 | $1.88 | 1.15% | 48.6% | Histogram-binned gradient boosted trees |
| 6 | **Gradient Boosting Regressor** | $2.34 | $2.48 | $1.89 | 1.16% | 47.6% | Sequential gradient boosted decision trees |
| 7 | **Support Vector Regressor (SVR)** | $2.92 | $3.19 | $2.32 | 1.42% | 51.4% | Non-linear RBF kernel regression |

---

## 📊 Top Predictive Technical Indicators

Permutation and feature importance on the champion pipeline reveal the strongest drivers of forward returns:

1. **Return_1d (20.5%)**: Immediate short-term momentum and mean-reversion drift.
2. **Spread_HL (19.7%)**: Daily volatility spread indicating market uncertainty and liquidity.
3. **Ratio_SMA21 (11.5%)**: Relative position to the short-term 21-day institutional trend line.
4. **Return_Lag1 (7.1%)**: 2-day return autocorrelation signal.
5. **MACD_Normalized (6.5%)**: Trend acceleration normalized by asset price.
6. **Ratio_SMA50 (6.1%)**: Primary medium-term trend regime.
7. **Volume_Ratio (5.9%)**: Unusual institutional volume spikes preceding breakout continuation.

---

## 💡 Key Machine Learning & Financial Concepts Explored

- **Stationary vs Non-Stationary Formulations**: Stock prices drift and reach all-time highs that decision trees cannot predict if trained on raw prices. By modeling **stationary returns** ($r_{t+1}$), the model generalizes across any price level and any asset.
- **Lookahead Bias Prevention**: Features at time $t$ strictly use information available at or before session $t$. All validation uses expanding chronological windows (`TimeSeriesSplit`).
- **Directional Accuracy over Raw RMSE**: In financial markets, capturing direction (Up vs Down, $>50\%$) provides quantitative trading edge even when dollar RMSE is close to the random walk.
- **Backtesting & Drawdown Mitigation**: Algorithmic rules that step aside during anticipated downturns can dramatically curtail peak drawdown compared to passive buy-and-hold investing.
