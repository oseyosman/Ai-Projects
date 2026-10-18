"""
Technical Indicators & Feature Engineering Pipeline for Stock Price Forecasting.
Calculates stationary returns, momentum oscillators (RSI, MACD), volatility metrics,
Bollinger Bands, and moving average ratios with zero lookahead bias.
"""

import numpy as np
import pandas as pd


# Core stationary feature definitions
FEATURE_COLUMNS = [
    "Return_1d",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
    "Ratio_SMA21",
    "Ratio_SMA50",
    "MACD_Hist",
    "MACD_Normalized",
    "RSI_14",
    "BB_Width",
    "BB_Percent",
    "Vol_20",
    "Spread_HL",
    "Spread_CO",
    "Volume_Ratio",
]


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators and stationary time series features from OHLCV data.
    Ensures zero lookahead bias: all features at index t use only data up to and including t.
    """
    data = df.copy()

    # Normalize column names
    col_map = {c: c.capitalize() for c in data.columns}
    data.rename(columns=col_map, inplace=True)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Input DataFrame must contain '{col}' column.")
        data[col] = pd.to_numeric(data[col], errors="coerce")

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    open_p = data["Open"]
    volume = data["Volume"]

    # 1. Stationary Returns & Lags
    data["Return_1d"] = close.pct_change()
    for lag in [1, 2, 3, 5]:
        data[f"Return_Lag{lag}"] = data["Return_1d"].shift(lag)

    # 2. Moving Average Ratios (Stationary trend strength)
    data["SMA_7"] = close.rolling(7).mean()
    data["SMA_21"] = close.rolling(21).mean()
    data["SMA_50"] = close.rolling(50).mean()
    data["Ratio_SMA21"] = close / data["SMA_21"].replace(0, np.nan)
    data["Ratio_SMA50"] = close / data["SMA_50"].replace(0, np.nan)

    # 3. MACD (Moving Average Convergence Divergence)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    data["MACD"] = macd_line
    data["MACD_Signal"] = macd_signal
    data["MACD_Hist"] = macd_line - macd_signal
    data["MACD_Normalized"] = data["MACD_Hist"] / close.replace(0, np.nan)

    # 4. RSI (14-period Relative Strength Index)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    data["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 5. Bollinger Bands (20-period, 2 std dev)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    data["BB_Upper"] = sma20 + (std20 * 2.0)
    data["BB_Lower"] = sma20 - (std20 * 2.0)
    data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / sma20.replace(0, np.nan)
    data["BB_Percent"] = (close - data["BB_Lower"]) / (data["BB_Upper"] - data["BB_Lower"]).replace(0, 1e-9)

    # 6. Volatility & Spreads
    # 20-day annualized realized volatility
    data["Vol_20"] = data["Return_1d"].rolling(20).std() * np.sqrt(252)
    # Intraday High-Low relative spread
    data["Spread_HL"] = (high - low) / close.replace(0, np.nan)
    # Intraday Close-Open return
    data["Spread_CO"] = (close - open_p) / open_p.replace(0, np.nan)

    # 7. Volume Activity Ratio
    vol_sma20 = volume.rolling(20).mean()
    data["Volume_Ratio"] = volume / vol_sma20.replace(0, np.nan)

    # 8. Target Formulations
    # Next day close price: Close_{t+1}
    data["Target_Next_Close"] = close.shift(-1)
    # Next day return: (Close_{t+1} - Close_t) / Close_t
    data["Target_Next_Return"] = (data["Target_Next_Close"] - close) / close.replace(0, np.nan)
    # Directional Target (1 = Up, 0 = Down)
    data["Target_Next_Direction"] = (data["Target_Next_Return"] > 0).astype(int)

    return data


def prepare_dataset(df: pd.DataFrame, drop_target_na: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Cleans warm-up NaN rows and extracts (X, y_return, y_close, full_clean_df).
    If drop_target_na is False (e.g. inference on the most recent bar), retains the last row.
    """
    data = compute_technical_indicators(df)

    # Drop warm-up periods (first ~50 days for SMA_50)
    warmup_cleaned = data.dropna(subset=FEATURE_COLUMNS).copy()

    if drop_target_na:
        # Also drop the last row where Target_Next_Close is unknown
        clean_df = warmup_cleaned.dropna(subset=["Target_Next_Return", "Target_Next_Close"]).copy()
    else:
        clean_df = warmup_cleaned.copy()

    clean_df.reset_index(drop=True, inplace=True)

    X = clean_df[FEATURE_COLUMNS]
    y_return = clean_df.get("Target_Next_Return", pd.Series(dtype=float))
    y_close = clean_df.get("Target_Next_Close", pd.Series(dtype=float))

    return X, y_return, y_close, clean_df
