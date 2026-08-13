import numpy as np
import pandas as pd


# =========================================================
# MOVING AVERAGES
# =========================================================

def sma(close, window):
    """
    Simple Moving Average.

    Calculates the average closing price over
    the specified number of trading days.
    """
    return close.rolling(window=window).mean()


# =========================================================
# RSI
# =========================================================

def rsi(close, window=14):
    """
    Calculate Relative Strength Index (RSI).

    RSI ranges from 0 to 100.
    """
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.rolling(window=window).mean()
    average_loss = losses.rolling(window=window).mean()

    relative_strength = average_gain / average_loss

    result = 100 - (
        100 / (1 + relative_strength)
    )

    return result


# =========================================================
# MACD
# =========================================================

def macd(
    close,
    fast_period=12,
    slow_period=26,
    signal_period=9
):
    """
    Calculate MACD, signal line and histogram.
    """

    fast_ema = close.ewm(
        span=fast_period,
        adjust=False
    ).mean()

    slow_ema = close.ewm(
        span=slow_period,
        adjust=False
    ).mean()

    macd_line = fast_ema - slow_ema

    signal_line = macd_line.ewm(
        span=signal_period,
        adjust=False
    ).mean()

    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# =========================================================
# BOLLINGER BANDS
# =========================================================

def bollinger_bands(
    close,
    window=20,
    num_std=2
):
    """
    Calculate Bollinger Bands.

    Returns:
        Middle Band
        Upper Band
        Lower Band
    """

    middle_band = close.rolling(
        window=window
    ).mean()

    rolling_std = close.rolling(
        window=window
    ).std()

    upper_band = middle_band + (
        rolling_std * num_std
    )

    lower_band = middle_band - (
        rolling_std * num_std
    )

    return middle_band, upper_band, lower_band


# =========================================================
# ADD ALL INDICATORS TO DATAFRAME
# =========================================================

def add_technical_indicators(df):
    """
    Add all technical indicators to a market-data DataFrame.
    """

    result = df.copy()

    close = result["Close"]

    # Moving averages
    result["SMA_50"] = sma(close, 50)
    result["SMA_200"] = sma(close, 200)

    # RSI
    result["RSI_14"] = rsi(close, 14)

    # MACD
    (
        result["MACD"],
        result["MACD_Signal"],
        result["MACD_Histogram"]
    ) = macd(close)

    # Bollinger Bands
    (
        result["BB_Middle"],
        result["BB_Upper"],
        result["BB_Lower"]
    ) = bollinger_bands(close)

    return result


# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":

    from data import get_price_data

    print("Loading SBI Life data...")

    df = get_price_data(
        "SBILIFE.NS",
        "2y"
    )

    df = add_technical_indicators(df)

    print("\nTechnical Indicators:")
    print("-" * 70)

    print(
        df[
            [
                "Date",
                "Close",
                "SMA_50",
                "SMA_200",
                "RSI_14",
                "MACD",
                "MACD_Signal",
                "BB_Upper",
                "BB_Lower"
            ]
        ].tail(5)
    )

    print("-" * 70)