import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(close):
    """
    Calculate daily percentage returns.
    """
    return close.pct_change().dropna()


def annual_return(close):
    """
    Calculate annualized historical return.
    """
    r = daily_returns(close)
    return r.mean() * TRADING_DAYS


def annual_volatility(close):
    """
    Calculate annualized volatility.
    """
    r = daily_returns(close)
    return r.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(close, risk_free_rate=0.06):
    """
    Calculate Sharpe Ratio using a 6% risk-free rate.
    """
    r = daily_returns(close)

    annualized_return = r.mean() * TRADING_DAYS
    annualized_vol = r.std() * np.sqrt(TRADING_DAYS)

    if annualized_vol == 0:
        return np.nan

    return (annualized_return - risk_free_rate) / annualized_vol


def maximum_drawdown(close):
    """
    Calculate the largest historical fall
    from a previous peak to a subsequent low.
    """
    running_max = close.cummax()
    drawdown = close / running_max - 1

    return drawdown.min()


def beta(stock_close, benchmark_close):
    """
    Calculate stock Beta relative to the benchmark.
    """
    stock_returns = daily_returns(stock_close)
    benchmark_returns = daily_returns(benchmark_close)

    aligned = pd.concat(
        [stock_returns, benchmark_returns],
        axis=1,
        join="inner"
    ).dropna()

    stock_returns = aligned.iloc[:, 0]
    benchmark_returns = aligned.iloc[:, 1]

    benchmark_variance = benchmark_returns.var()

    if benchmark_variance == 0:
        return np.nan

    return stock_returns.cov(benchmark_returns) / benchmark_variance


def performance_summary(stock_close, benchmark_close):
    """
    Calculate all five performance metrics.
    """
    return {
        "Annual Return": annual_return(stock_close),
        "Annual Volatility": annual_volatility(stock_close),
        "Sharpe Ratio": sharpe_ratio(stock_close),
        "Maximum Drawdown": maximum_drawdown(stock_close),
        "Beta": beta(stock_close, benchmark_close),
    }


if __name__ == "__main__":
    from data import get_price_data

    # Get SBI Life data
    df = get_price_data("SBILIFE.NS", "2y")

    # Get NIFTY 50 benchmark data
    benchmark_df = get_price_data("^NSEI", "2y")

    # Extract closing prices
    close = df.set_index("Date")["Close"]
    benchmark_close = benchmark_df.set_index("Date")["Close"]

    # Calculate all metrics
    summary = performance_summary(
        close,
        benchmark_close
    )

    # Display results
    print("\nPerformance Summary:")
    print("-" * 40)

    for metric, value in summary.items():

        if metric in ["Sharpe Ratio", "Beta"]:
            print(f"{metric}: {value:.2f}")

        else:
            print(f"{metric}: {value:.2%}")

    print("-" * 40)