import numpy as np
import pandas as pd

from analytics import performance_summary
from technicals import add_technical_indicators
from data import get_price_data


RISK_PROFILES = {
    "Conservative": {
        "return": 0.15,
        "sharpe": 0.30,
        "volatility": 0.20,
        "drawdown": 0.20,
        "momentum": 0.15,
    },
    "Moderate": {
        "return": 0.25,
        "sharpe": 0.25,
        "volatility": 0.15,
        "drawdown": 0.15,
        "momentum": 0.20,
    },
    "Aggressive": {
        "return": 0.35,
        "sharpe": 0.20,
        "volatility": 0.10,
        "drawdown": 0.10,
        "momentum": 0.25,
    },
}


def _minmax(series, higher_is_better=True):
    """Normalize a metric to 0-100 while handling flat/invalid data."""
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1 or values.max() == values.min():
        result = pd.Series(50.0, index=series.index)
    else:
        result = (values - values.min()) / (values.max() - values.min()) * 100
    if not higher_is_better:
        result = 100 - result
    return result.fillna(50.0)


def _momentum_score(df):
    """Score current technical momentum from SMA, RSI and MACD."""
    latest = df.iloc[-1]
    score = 50.0

    sma50 = latest.get("SMA_50", np.nan)
    sma200 = latest.get("SMA_200", np.nan)
    close = latest.get("Close", np.nan)
    rsi = latest.get("RSI_14", np.nan)
    macd = latest.get("MACD", np.nan)
    signal = latest.get("MACD_Signal", np.nan)

    if pd.notna(close) and pd.notna(sma50):
        score += 15 if close > sma50 else -15
    if pd.notna(sma50) and pd.notna(sma200):
        score += 15 if sma50 > sma200 else -15
    if pd.notna(rsi):
        if 50 <= rsi <= 70:
            score += 15
        elif rsi < 30:
            score += 5
        elif rsi > 70:
            score -= 10
    if pd.notna(macd) and pd.notna(signal):
        score += 10 if macd > signal else -10

    return float(np.clip(score, 0, 100))


def calculate_company_score(stock_close, benchmark_close, technical_df, risk_profile="Moderate"):
    """Calculate raw investment metrics for one company."""
    metrics = performance_summary(stock_close, benchmark_close)
    weights = RISK_PROFILES.get(risk_profile, RISK_PROFILES["Moderate"])

    return {
        **metrics,
        "Momentum": _momentum_score(technical_df),
        "return_weight": weights["return"],
        "sharpe_weight": weights["sharpe"],
        "volatility_weight": weights["volatility"],
        "drawdown_weight": weights["drawdown"],
        "momentum_weight": weights["momentum"],
    }


def rank_companies(company_metrics, risk_profile="Moderate"):
    """Rank companies using normalized performance, risk and momentum metrics."""
    df = pd.DataFrame(company_metrics).T
    weights = RISK_PROFILES.get(risk_profile, RISK_PROFILES["Moderate"])

    df["Return Score"] = _minmax(df["Annual Return"], True)
    df["Sharpe Score"] = _minmax(df["Sharpe Ratio"], True)
    df["Volatility Score"] = _minmax(df["Annual Volatility"], False)
    df["Drawdown Score"] = _minmax(df["Maximum Drawdown"], True)

    df["Investment Score"] = (
        df["Return Score"] * weights["return"]
        + df["Sharpe Score"] * weights["sharpe"]
        + df["Volatility Score"] * weights["volatility"]
        + df["Drawdown Score"] * weights["drawdown"]
        + df["Momentum"] * weights["momentum"]
    )

    return df.sort_values("Investment Score", ascending=False)


def build_portfolio(ranked_df, investment_amount, max_companies=5, max_allocation=0.35):
    """Allocate capital to top-ranked companies with a per-company diversification cap."""
    if investment_amount <= 0:
        raise ValueError("Investment amount must be greater than zero.")
    if ranked_df.empty:
        raise ValueError("No ranked companies are available.")

    selected = ranked_df.head(max_companies).copy()
    score_sum = selected["Investment Score"].sum()

    if score_sum <= 0:
        raw_weights = pd.Series(1 / len(selected), index=selected.index)
    else:
        raw_weights = selected["Investment Score"] / score_sum

    # Cap concentration, then redistribute excess proportionally among
    # companies that remain below the cap.
    weights = raw_weights.copy()
    for _ in range(20):
        excess = (weights - max_allocation).clip(lower=0).sum()
        if excess <= 1e-10:
            break
        weights = weights.clip(upper=max_allocation)
        eligible = weights < max_allocation - 1e-10
        if not eligible.any():
            break
        base = weights[eligible]
        base_sum = base.sum()
        if base_sum <= 0:
            weights.loc[eligible] += excess / eligible.sum()
        else:
            weights.loc[eligible] += excess * base / base_sum

    weights = weights / weights.sum()
    selected["Allocation %"] = weights * 100
    selected["Recommended Amount"] = investment_amount * weights
    return selected


def analyze_universe(insurance_stocks, benchmark_ticker, risk_profile="Moderate", period="2y"):
    """Download and score the complete investment universe."""
    benchmark_df = get_price_data(benchmark_ticker, period)
    benchmark_close = benchmark_df.set_index("Date")["Close"]

    results = {}
    failures = {}

    for company, ticker in insurance_stocks.items():
        try:
            stock_df = get_price_data(ticker, period)
            stock_df = add_technical_indicators(stock_df)
            close = stock_df.set_index("Date")["Close"]
            results[company] = calculate_company_score(
                close,
                benchmark_close,
                stock_df,
                risk_profile,
            )
            results[company]["Ticker"] = ticker
        except Exception as exc:
            failures[company] = str(exc)

    ranked = rank_companies(results, risk_profile) if results else pd.DataFrame()
    return ranked, failures


def recommendation_text(row):
    """Produce a simple explanation for a ranked company."""
    score = row["Investment Score"]
    sharpe = row["Sharpe Ratio"]
    volatility = row["Annual Volatility"]
    momentum = row["Momentum"]

    if score >= 75:
        rating = "Strong candidate"
    elif score >= 60:
        rating = "Good candidate"
    elif score >= 45:
        rating = "Watchlist"
    else:
        rating = "Lower priority"

    risk_text = "lower" if volatility < 0.20 else "moderate" if volatility < 0.35 else "higher"
    momentum_text = "positive" if momentum >= 60 else "mixed" if momentum >= 40 else "weak"

    return (
        f"{rating}. The model gives this company a score of {score:.0f}/100. "
        f"Its historical volatility is {risk_text}, risk-adjusted performance "
        f"(Sharpe ratio {sharpe:.2f}) is {'stronger' if sharpe >= 1 else 'moderate' if sharpe >= 0 else 'weaker'}, "
        f"and current technical momentum is {momentum_text}. "
        "The score reflects historical data and is not a guarantee of future returns."
    )
