"""
Exploratory Data Analysis (EDA) & Financial Time Series Diagnostics.
Analyzes price trends, return distributions, volatility clustering,
leptokurtic fat-tails, momentum regimes, and stationarity properties.
"""

import os
import sys
import numpy as np
import pandas as pd

# Local feature imports
sys.path.insert(0, os.path.dirname(__file__))
from features import compute_technical_indicators, FEATURE_COLUMNS


def print_banner(text: str, char: str = "=", width: int = 74):
    print("\n" + char * width)
    print(f" {text}")
    print(char * width)


def render_ascii_histogram(data: np.ndarray, bins: int = 15, width: int = 40):
    counts, edges = np.histogram(data, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        cnt = counts[i]
        bar_len = int((cnt / max_count) * width)
        bar = "█" * bar_len
        print(f"  [{low:>6.2f}% to {high:>6.2f}%] | {cnt:>4} days | {bar}")


def render_ascii_trend_chart(prices: np.ndarray, dates: list[str], height: int = 12, width: int = 65):
    """Renders a scaled ASCII price progression chart across the historical period."""
    n = len(prices)
    step = n / width
    sampled = [prices[int(i * step)] for i in range(width)]

    min_p = min(sampled)
    max_p = max(sampled)
    p_range = max_p - min_p if max_p > min_p else 1.0

    grid = [[" " for _ in range(width)] for _ in range(height)]

    for col in range(width):
        norm = (sampled[col] - min_p) / p_range
        row = height - 1 - int(round(norm * (height - 1)))
        grid[row][col] = "■"

    print(f"  ${max_p:>6.2f} ┌" + "─" * width + "┐")
    for r in range(height):
        label = f"  ${(max_p - (r / (height - 1)) * p_range):>6.2f} │" if r % 3 == 0 else "          │"
        line = "".join(grid[r])
        print(f"{label}{line}│")
    print("          └" + "─" * width + "┘")
    start_d = str(dates[0])[:10]
    mid_d = str(dates[len(dates)//2])[:10]
    end_d = str(dates[-1])[:10]
    spacing = width - len(start_d) - len(end_d) - len(mid_d)
    sp1 = spacing // 2
    sp2 = spacing - sp1
    print(f"           {start_d}{' ' * sp1}{mid_d}{' ' * sp2}{end_d}")


def main():
    print_banner("📈 PROJECT 10: FINANCIAL TIME SERIES EXPLORATORY DATA ANALYSIS 📈")

    data_path = os.path.join(os.path.dirname(__file__), "data", "AAPL_stock_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}.")
        return

    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    n_bars = len(df)
    start_date = df["Date"].iloc[0].strftime("%Y-%m-%d")
    end_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    min_price = df["Close"].min()
    max_price = df["Close"].max()
    total_gain_pct = ((end_price - start_price) / start_price) * 100

    print(f"Asset Manifest: Apple Inc. (Ticker: AAPL) — NASDAQ Composite")
    print(f"Sample Period : {start_date} to {end_date} ({n_bars:,} daily trading sessions)")
    print(f"Price Spectrum: Min ${min_price:.2f} | Max ${max_price:.2f} | Start ${start_price:.2f} ➔ End ${end_price:.2f} (+{total_gain_pct:.1f}%)")

    # 1. Scaled Price Trajectory Chart
    print_banner("1. HISTORICAL PRICE TRAJECTORY (ASCII CHART)")
    render_ascii_trend_chart(df["Close"].values, df["Date"].dt.strftime("%Y-%m-%d").tolist(), height=10, width=58)

    # 2. Daily Returns & Statistical Moments
    print_banner("2. DAILY RETURNS DISTRIBUTION & STATISTICAL MOMENTS")
    df_calc = compute_technical_indicators(df)
    returns = df_calc["Return_1d"].dropna() * 100

    mean_ret = returns.mean()
    std_ret = returns.std()
    ann_vol = (std_ret * np.sqrt(252))
    skewness = returns.skew()
    kurtosis = returns.kurtosis()  # Excess kurtosis

    print(f"  • Mean Daily Return     : {mean_ret:+.3f}% (Expected positive equity drift)")
    print(f"  • Daily Volatility (1σ) : {std_ret:.2f}% (Annualized Realized Volatility: {ann_vol:.1f}%)")
    print(f"  • Return Skewness       : {skewness:+.2f} (Slight left skew, sudden downside drawdowns)")
    print(f"  • Excess Kurtosis       : {kurtosis:+.2f} (Leptokurtic: Heavy 'fat tails' vs Normal bell curve)")
    print(f"  • Best Trading Day      : +{returns.max():.2f}% (Post-earnings / stimulus rally)")
    print(f"  • Worst Trading Day     : {returns.min():.2f}% (Market shock / macro sell-off)\n")

    print("  Histogram of Daily Returns (%):")
    render_ascii_histogram(returns.values, bins=11, width=34)

    # 3. Technical Momentum Regimes
    print_banner("3. TECHNICAL OSCILLATOR REGIMES (RSI-14)")
    rsi = df_calc["RSI_14"].dropna()
    overbought = (rsi > 70).sum()
    neutral = ((rsi >= 30) & (rsi <= 70)).sum()
    oversold = (rsi < 30).sum()
    tot_rsi = len(rsi)

    print(f"  • Overbought (RSI > 70) : {overbought:>4} days ({(overbought/tot_rsi)*100:>5.1f}%) | Momentum climax, pullbacks likely")
    print(f"  • Neutral    (30 ≤ RSI ≤ 70): {neutral:>4} days ({(neutral/tot_rsi)*100:>5.1f}%) | Balanced trend consolidation")
    print(f"  • Oversold   (RSI < 30) : {oversold:>4} days ({(oversold/tot_rsi)*100:>5.1f}%) | Panic capitulation, mean-reversion bounces")

    # 4. Moving Average Trend State
    print_banner("4. MOVING AVERAGE REGIME ANALYSIS (SMA-50)")
    above_sma50 = (df_calc["Close"] > df_calc["SMA_50"]).sum()
    below_sma50 = (df_calc["Close"] <= df_calc["SMA_50"]).sum()
    valid_sma = above_sma50 + below_sma50

    print(f"  • Bullish Regime (Price > SMA-50): {above_sma50:>4} days ({(above_sma50/valid_sma)*100:>5.1f}%)")
    print(f"  • Bearish Regime (Price ≤ SMA-50): {below_sma50:>4} days ({(below_sma50/valid_sma)*100:>5.1f}%)")

    # 5. Stationarity & Econometric Takeaway
    print_banner("5. TIME SERIES STATIONARITY & MACHINE LEARNING IMPLICATIONS")
    print("  • Non-Stationarity of Raw Prices: Stock prices wander as non-stationary stochastic processes.")
    print("    A tree-based algorithm trained on $40–$130 prices CANNOT extrapolate when the price hits $190.")
    print("  • Stationarity of Returns: Percentage returns and ratio indicators (RSI, Bollinger %B, MACD")
    print("    normalized) are mean-reverting and stationary, enabling machine learning models to generalize")
    print("    consistently across multi-year market cycles.")
    print("  • The Lookahead Bias Trap: Standard K-Fold Cross-Validation leaks future information into past")
    print("    predictions. We must strictly employ TimeSeriesSplit (expanding chronological windows).")
    print("  • Directional vs Magnitude: While predicting exact dollar prices is bounded by the random walk,")
    print("    predicting the directional probability (Up vs Down) enables quantitative trading alpha.\n")


if __name__ == "__main__":
    main()
