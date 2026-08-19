import yfinance as yf
import pandas as pd

# Listed Indian insurance companies and insurance-linked businesses.
# The broader universe is useful for portfolio screening because India has
# relatively few separately listed pure-play insurers.
INSURANCE_STOCKS = {
    # Existing pure-play insurance companies
    "HDFC Life": "HDFCLIFE.NS",
    "SBI Life": "SBILIFE.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "LIC": "LICI.NS",
    "GIC Re": "GICRE.NS",
    "Star Health": "STARHEALTH.NS",
    "New India Assurance": "NIACL.NS",

    # Additional listed insurance / insurance-linked businesses
    "ICICI Lombard General Insurance": "ICICIGI.NS",
    "Go Digit General Insurance": "GODIGIT.NS",
    "Niva Bupa Health Insurance": "NIVABUPA.NS",
    "Max Financial Services": "MFSL.NS",
    "Aditya Birla Capital": "ABCAPITAL.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "PB Fintech": "POLICYBZR.NS",
    "Medi Assist Healthcare Services": "MEDIASSIST.NS",
    "Cholamandalam Investment & Finance": "CHOLAFIN.NS",
    "Shriram Finance": "SHRIRAMFIN.NS",
    "Muthoot Finance": "MUTHOOTFIN.NS",
    "Manappuram Finance": "MANAPPURAM.NS",
    "SBI Cards & Payment Services": "SBICARD.NS",
    "Bajaj Holdings & Investment": "BAJAJHLDNG.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "HDFC Asset Management": "HDFCAMC.NS",
    "SBI Funds Management / SBI-related": "SBIN.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "LIC Housing Finance": "LICHSGFIN.NS",
    "Can Fin Homes": "CANFINHOME.NS",
    "PNB Housing Finance": "PNBHOUSING.NS",
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
