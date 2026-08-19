import yfinance as yf
import pandas as pd

# Listed Indian insurance companies and insurance-linked businesses.
# The broader universe is useful for portfolio screening because India has
# relatively few separately listed pure-play insurers.
INSURANCE_STOCKS = {
    # Pure-play / directly listed insurance companies
    "HDFC Life": "HDFCLIFE.NS",
    "SBI Life": "SBILIFE.NS",
    "ICICI Prudential Life": "ICICIPRULI.NS",
    "LIC": "LICI.NS",
    "GIC Re": "GICRE.NS",
    "Star Health": "STARHEALTH.NS",
    "New India Assurance": "NIACL.NS",
    "ICICI Lombard General Insurance": "ICICIGI.NS",
    "Go Digit General Insurance": "GODIGIT.NS",
    "Niva Bupa Health Insurance": "NIVABUPA.NS",

    # Insurance / financial-services ecosystem companies
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
    "State Bank of India": "SBIN.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "LIC Housing Finance": "LICHSGFIN.NS",
    "Can Fin Homes": "CANFINHOME.NS",
    "PNB Housing Finance": "PNBHOUSING.NS",
}

COMPANY_CATEGORIES = {
    "HDFC Life": "Pure Insurance",
    "SBI Life": "Pure Insurance",
    "ICICI Prudential Life": "Pure Insurance",
    "LIC": "Pure Insurance",
    "GIC Re": "Pure Insurance",
    "Star Health": "Pure Insurance",
    "New India Assurance": "Pure Insurance",
    "ICICI Lombard General Insurance": "Pure Insurance",
    "Go Digit General Insurance": "Pure Insurance",
    "Niva Bupa Health Insurance": "Pure Insurance",
    "Max Financial Services": "Insurance-linked",
    "Aditya Birla Capital": "Insurance-linked",
    "Bajaj Finserv": "Insurance-linked",
    "PB Fintech": "Insurance-linked",
    "Medi Assist Healthcare Services": "Insurance-linked",
    "Cholamandalam Investment & Finance": "Financial Services",
    "Shriram Finance": "Financial Services",
    "Muthoot Finance": "Financial Services",
    "Manappuram Finance": "Financial Services",
    "SBI Cards & Payment Services": "Financial Services",
    "Bajaj Holdings & Investment": "Financial Services",
    "Kotak Mahindra Bank": "Financial Services",
    "HDFC Asset Management": "Financial Services",
    "State Bank of India": "Financial Services",
    "ICICI Bank": "Financial Services",
    "HDFC Bank": "Financial Services",
    "Axis Bank": "Financial Services",
    "Bajaj Finance": "Financial Services",
    "LIC Housing Finance": "Financial Services",
    "Can Fin Homes": "Financial Services",
    "PNB Housing Finance": "Financial Services",
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
