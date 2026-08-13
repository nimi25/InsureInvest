import yfinance as yf
import pandas as pd

# Insurance companies we will analyse
INSURANCE_STOCKS = {
    "HDFC Life": "HDFCLIFE.NS",
    "SBI Life": "SBILIFE.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "LIC": "LICI.NS",
    "GIC Re": "GICRE.NS",
    "Star Health": "STARHEALTH.NS",
    "New India Assurance": "NIACL.NS",
}

# NIFTY 50 will be our benchmark
BENCHMARK = "^NSEI"


def get_price_data(ticker, period="5y"):
    data = yf.download(
        ticker,
        period=period,
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        raise ValueError(f"No data returned for {ticker}")

    # yfinance can sometimes return multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.dropna(subset=["Close"])

    return data


if __name__ == "__main__":
    df = get_price_data("SBILIFE.NS", "2y")
    print(df.tail())