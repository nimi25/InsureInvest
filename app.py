import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import performance_summary
from data import BENCHMARK, INSURANCE_STOCKS, get_price_data
from technicals import add_technical_indicators

# ---------------------------------------------------------------------------
# HTML rendering helper
# ---------------------------------------------------------------------------
# st.markdown() still runs strings through a Markdown parser before treating
# them as HTML. Two Markdown rules were breaking our nested, indented HTML:
# (1) lines indented 4+ spaces become literal code blocks, and (2) a blank
# line starts a new block. Collapsing the markup to one contiguous line (no
# newlines/indentation) avoids both, so the HTML always renders as HTML.


def _collapse_html(markup):
    return re.sub(r"\n\s*", "", markup.strip())


def render_html(markup):
    st.markdown(_collapse_html(markup), unsafe_allow_html=True)


def render_sidebar_html(markup):
    st.sidebar.markdown(_collapse_html(markup), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chart theming
# ---------------------------------------------------------------------------
# Every chart in the app uses this single dark style - no light/dark
# switching. That switching was the actual bug: st.plotly_chart() applies
# Streamlit's own active theme on top of a figure's colors by default,
# which is what washed out the legend/axis text in some views. Passing
# theme=None on every st.plotly_chart() call (below) stops that override,
# so these colors always render exactly as set, everywhere.

DARK_CHART = dict(
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(family="Arial", color="#ffffff"),
)
DARK_GRID = "#374151"

LEGEND_TOP = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#ffffff"))
LEGEND_TOP_RIGHT = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#ffffff"))

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(page_title="InsureInvest", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

# =============================================================================
# CUSTOM CSS
# =============================================================================

render_html(
    """
    <style>
    .stApp { background-color: #f6f8fb; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

    section[data-testid="stSidebar"] { background-color: #111827; }
    section[data-testid="stSidebar"] * { color: #f9fafb !important; }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1e3a5f 100%);
        padding: 2rem 2.5rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.12);
    }
    .hero-title { color: #ffffff !important; font-size: 2.4rem; font-weight: 700; margin: 0; }
    .hero-subtitle { color: #cbd5e1 !important; font-size: 1rem; margin-top: 0.5rem; }

    .company-name { font-size: 2rem; font-weight: 700; color: #111827 !important; margin-bottom: 0.1rem; }
    .company-meta { color: #64748b !important; font-size: 0.9rem; margin-bottom: 1.5rem; }

    .section-title { color: #111827 !important; font-size: 1.35rem; font-weight: 700; margin-top: 1.8rem; margin-bottom: 0.7rem; }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        min-height: 125px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }
    .metric-label { color: #64748b !important; font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
    .metric-value { color: #111827 !important; font-size: 1.8rem; font-weight: 700; margin-top: 0.45rem; }
    .metric-description { color: #94a3b8 !important; font-size: 0.75rem; margin-top: 0.3rem; }

    .interpretation-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 14px;
        padding: 1.5rem 1.7rem;
        margin-top: 0.8rem;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.05);
    }
    .interpretation-card * { color: #334155 !important; }
    .interpretation-overall { color: #111827 !important; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.4rem; }
    .interpretation-overall-text { color: #475569 !important; font-size: 0.92rem; line-height: 1.6; margin-bottom: 1rem; }
    .interpretation-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem 2rem; margin-top: 1rem; }
    .interpretation-item { background: #f8fafc !important; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem; }
    .interpretation-label { color: #1e3a5f !important; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.35rem; }
    .interpretation-text { color: #475569 !important; font-size: 0.84rem; line-height: 1.55; }
    .interpretation-note { color: #64748b !important; font-size: 0.75rem; margin-top: 1.2rem; padding-top: 0.8rem; border-top: 1px solid #e2e8f0; }

    .footer { text-align: center; color: #94a3b8 !important; font-size: 0.75rem; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; }

    @media (max-width: 768px) {
        .interpretation-grid { grid-template-columns: 1fr; }
        .hero-title { font-size: 1.8rem; }
    }
    </style>
    """
)

# =============================================================================
# SIDEBAR
# =============================================================================

render_sidebar_html(
    """
    <div style="font-size: 1.7rem; font-weight: 700; color: #ffffff !important;">InsureInvest</div>
    <div style="color: #94a3b8 !important; font-size: 0.85rem; margin-top: 5px;">Insurance Investment Analytics</div>
    """
)
st.sidebar.markdown("---")
render_sidebar_html('<div style="font-weight: 600; font-size: 1rem; color: #ffffff !important;">Investment Selection</div>')

selected_company = st.sidebar.selectbox("Select an insurance company", list(INSURANCE_STOCKS.keys()))
selected_ticker = INSURANCE_STOCKS[selected_company]

render_sidebar_html(
    f"""
    <div style="background: #1f2937; padding: 10px; border-radius: 8px; margin-top: 10px; color: #cbd5e1 !important; font-size: 0.85rem;">
        NSE Ticker<br>
        <strong style="color: #ffffff !important;">{selected_ticker}</strong>
    </div>
    """
)
st.sidebar.markdown("---")
render_sidebar_html(
    """
    <div style="color: #94a3b8 !important; font-size: 0.78rem; line-height: 1.5;">
        InsureInvest evaluates listed insurance companies using historical market data and financial risk metrics.
    </div>
    """
)

# =============================================================================
# HERO HEADER
# =============================================================================

render_html(
    """
    <div class="hero">
        <div class="hero-title">InsureInvest</div>
        <div class="hero-subtitle">Insurance Stock Investment Decision Support System</div>
    </div>
    """
)

# =============================================================================
# LOAD DATA
# =============================================================================

with st.spinner("Loading market data..."):
    try:
        stock_df = get_price_data(selected_ticker, "2y")
        stock_df = add_technical_indicators(stock_df)
        benchmark_df = get_price_data(BENCHMARK, "2y")
    except Exception as exc:
        st.error(
            "Couldn't load market data from Yahoo Finance right now. "
            "This is usually temporary rate-limiting on their side — "
            "wait a minute and click below to retry."
        )
        st.caption(f"Details: {exc}")
        if st.button("Retry"):
            st.cache_data.clear()
            st.rerun()
        st.stop()

stock_close = stock_df.set_index("Date")["Close"]
benchmark_close = benchmark_df.set_index("Date")["Close"]
metrics = performance_summary(stock_close, benchmark_close)
latest_price = stock_close.iloc[-1]
latest_date = stock_df["Date"].max()

# =============================================================================
# COMPANY INFORMATION
# =============================================================================

render_html(
    f"""
    <div class="company-name">{selected_company}</div>
    <div class="company-meta">
        NSE: {selected_ticker} &nbsp; • &nbsp; Latest price: ₹{latest_price:,.2f}
        &nbsp; • &nbsp; Data through {latest_date.strftime('%d %B %Y')}
    </div>
    """
)

# =============================================================================
# KPI CARDS
# =============================================================================

render_html('<div class="section-title">Performance Snapshot</div>')

col1, col2, col3, col4, col5 = st.columns(5)
metric_cards = [
    (col1, "Annual Return", f"{metrics['Annual Return']:.2%}", "Historical annualized return"),
    (col2, "Annual Volatility", f"{metrics['Annual Volatility']:.2%}", "Historical price fluctuation"),
    (col3, "Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", "Risk-adjusted return"),
    (col4, "Maximum Drawdown", f"{metrics['Maximum Drawdown']:.2%}", "Worst peak-to-trough fall"),
    (col5, "Beta", f"{metrics['Beta']:.2f}", "Sensitivity to NIFTY 50"),
]
for column, label, value, description in metric_cards:
    with column:
        render_html(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-description">{description}</div>
            </div>
            """
        )

# =============================================================================
# PRICE CHART + TECHNICAL INDICATORS
# =============================================================================

st.markdown('<div class="section-title">Historical Price & Technical Indicators</div>', unsafe_allow_html=True)
st.caption("Daily price movement with 50-day and 200-day moving averages and Bollinger Bands.")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=stock_df["Date"], open=stock_df["Open"], high=stock_df["High"],
    low=stock_df["Low"], close=stock_df["Close"], name=selected_company,
))
fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["SMA_50"], mode="lines", name="50-Day SMA", line=dict(width=2)))
fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["SMA_200"], mode="lines", name="200-Day SMA", line=dict(width=2)))
fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["BB_Upper"], mode="lines", name="Bollinger Upper", line=dict(width=1, dash="dash")))
fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["BB_Lower"], mode="lines", name="Bollinger Lower", line=dict(width=1, dash="dash")))

fig.update_layout(
    **DARK_CHART,
    height=560,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(title="Date", rangeslider=dict(visible=False), showgrid=False),
    yaxis=dict(title="Price (₹)", showgrid=True, gridcolor=DARK_GRID),
    hovermode="x unified",
    legend=LEGEND_TOP,
)
st.plotly_chart(fig, use_container_width=True, theme=None)

# =============================================================================
# RSI CHART
# =============================================================================

st.markdown('<div class="section-title">Momentum Indicator — RSI</div>', unsafe_allow_html=True)
st.caption("14-day Relative Strength Index. Values above 70 may indicate overbought conditions, while values below 30 may indicate oversold conditions.")

rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["RSI_14"], mode="lines", name="RSI (14)", line=dict(width=2)))
rsi_fig.add_hline(y=70, line_dash="dash", annotation_text="Overbought (70)", annotation_position="top right")
rsi_fig.add_hline(y=30, line_dash="dash", annotation_text="Oversold (30)", annotation_position="bottom right")
rsi_fig.add_hline(y=50, line_dash="dot", annotation_text="50", annotation_position="top right")

rsi_fig.update_layout(
    **DARK_CHART,
    height=320,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(title="Date", showgrid=False),
    yaxis=dict(title="RSI", range=[0, 100], showgrid=True, gridcolor=DARK_GRID),
    hovermode="x unified",
    showlegend=True,
)
st.plotly_chart(rsi_fig, use_container_width=True, theme=None)

# =============================================================================
# MACD CHART
# =============================================================================

st.markdown('<div class="section-title">Trend & Momentum Indicator — MACD</div>', unsafe_allow_html=True)
st.caption("Moving Average Convergence Divergence using 12-day, 26-day and 9-day exponential moving averages.")

macd_fig = go.Figure()
macd_fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["MACD"], mode="lines", name="MACD", line=dict(width=2)))
macd_fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["MACD_Signal"], mode="lines", name="Signal Line", line=dict(width=2)))
macd_fig.add_trace(go.Bar(x=stock_df["Date"], y=stock_df["MACD_Histogram"], name="Histogram", opacity=0.6))
macd_fig.add_hline(y=0, line_dash="dash")

macd_fig.update_layout(
    **DARK_CHART,
    height=360,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(title="Date", showgrid=False),
    yaxis=dict(title="MACD", showgrid=True, gridcolor=DARK_GRID),
    hovermode="x unified",
    legend=LEGEND_TOP,
)
st.plotly_chart(macd_fig, use_container_width=True, theme=None)

# =============================================================================
# TECHNICAL SIGNAL SUMMARY
# =============================================================================

render_html('<div class="section-title">Technical Signal Summary</div>')
render_html(
    '<div style="color:#64748b; font-size:0.9rem; margin-bottom:1rem;">'
    'Latest technical signals based on the selected insurance company.</div>'
)

latest_close = stock_df["Close"].iloc[-1]
latest_sma_50 = stock_df["SMA_50"].iloc[-1]
latest_sma_200 = stock_df["SMA_200"].iloc[-1]
latest_rsi = stock_df["RSI_14"].iloc[-1]
latest_macd = stock_df["MACD"].iloc[-1]
latest_macd_signal = stock_df["MACD_Signal"].iloc[-1]
latest_bb_upper = stock_df["BB_Upper"].iloc[-1]
latest_bb_lower = stock_df["BB_Lower"].iloc[-1]

if latest_close > latest_sma_50 and latest_close > latest_sma_200:
    trend_signal = "Positive"
    trend_text = "The current price is above both the 50-day and 200-day moving averages, indicating a positive trend relative to these indicators."
elif latest_close < latest_sma_50 and latest_close < latest_sma_200:
    trend_signal = "Weak"
    trend_text = "The current price is below both the 50-day and 200-day moving averages, indicating a relatively weak price trend."
else:
    trend_signal = "Mixed"
    trend_text = "The current price is between the 50-day and 200-day moving averages, indicating mixed trend signals."

if latest_rsi >= 70:
    rsi_signal = "Overbought"
    rsi_text = f"RSI is {latest_rsi:.1f}, above the 70 threshold. This may indicate strong upward momentum or an overbought condition."
elif latest_rsi <= 30:
    rsi_signal = "Oversold"
    rsi_text = f"RSI is {latest_rsi:.1f}, below the 30 threshold. This may indicate strong downward momentum or an oversold condition."
else:
    rsi_signal = "Neutral"
    rsi_text = f"RSI is {latest_rsi:.1f}, within the 30–70 neutral range."

if latest_macd > latest_macd_signal:
    macd_signal = "Positive"
    macd_text = "The MACD line is above the signal line, indicating positive momentum under the MACD framework."
elif latest_macd < latest_macd_signal:
    macd_signal = "Negative"
    macd_text = "The MACD line is below the signal line, indicating negative momentum under the MACD framework."
else:
    macd_signal = "Neutral"
    macd_text = "The MACD line is approximately equal to the signal line."

if latest_close > latest_bb_upper:
    bollinger_signal = "Above Upper Band"
    bollinger_text = "The current price is above the upper Bollinger Band, indicating unusually strong recent price movement."
elif latest_close < latest_bb_lower:
    bollinger_signal = "Below Lower Band"
    bollinger_text = "The current price is below the lower Bollinger Band, indicating unusually weak recent price movement."
else:
    bollinger_signal = "Within Bands"
    bollinger_text = "The current price is currently within the Bollinger Bands."

technical_html = f"""
<div class="interpretation-card">
    <div class="interpretation-overall">Trend: {trend_signal}</div>
    <div class="interpretation-overall-text">{trend_text}</div>
    <div class="interpretation-grid">
        <div class="interpretation-item">
            <div class="interpretation-label">Current Price</div>
            <div class="interpretation-text">₹{latest_close:,.2f}</div>
        </div>
        <div class="interpretation-item">
            <div class="interpretation-label">Moving Averages</div>
            <div class="interpretation-text">50-Day SMA: ₹{latest_sma_50:,.2f}<br>200-Day SMA: ₹{latest_sma_200:,.2f}</div>
        </div>
        <div class="interpretation-item">
            <div class="interpretation-label">RSI — {rsi_signal}</div>
            <div class="interpretation-text">{rsi_text}</div>
        </div>
        <div class="interpretation-item">
            <div class="interpretation-label">MACD — {macd_signal}</div>
            <div class="interpretation-text">{macd_text}<br><br>MACD: {latest_macd:.2f}<br>Signal: {latest_macd_signal:.2f}</div>
        </div>
        <div class="interpretation-item">
            <div class="interpretation-label">Bollinger Bands — {bollinger_signal}</div>
            <div class="interpretation-text">{bollinger_text}<br><br>Upper Band: ₹{latest_bb_upper:,.2f}<br>Lower Band: ₹{latest_bb_lower:,.2f}</div>
        </div>
        <div class="interpretation-item">
            <div class="interpretation-label">Technical Overview</div>
            <div class="interpretation-text">Trend: {trend_signal}<br>Momentum: {rsi_signal}<br>MACD: {macd_signal}<br>Volatility Position: {bollinger_signal}</div>
        </div>
    </div>
    <div class="interpretation-note">
        These signals are generated automatically from the selected company's historical market data.
        They are technical indicators, not a buy/sell recommendation.
    </div>
</div>
"""
render_html(technical_html)

# =============================================================================
# BENCHMARK COMPARISON — STOCK VS NIFTY 50
# =============================================================================

render_html('<div class="section-title">Benchmark Comparison</div>')
render_html(
    '<div style="color:#64748b; font-size:0.9rem; margin-bottom:1rem;">'
    'Comparison of the selected insurance stock against the NIFTY 50 over the same historical period.</div>'
)

comparison_df = pd.concat(
    [stock_close.rename(selected_company), benchmark_close.rename("NIFTY 50")], axis=1
).dropna()
normalized_comparison = comparison_df / comparison_df.iloc[0] * 100

stock_total_return = (comparison_df[selected_company].iloc[-1] / comparison_df[selected_company].iloc[0]) - 1
benchmark_total_return = (comparison_df["NIFTY 50"].iloc[-1] / comparison_df["NIFTY 50"].iloc[0]) - 1

trading_days = len(comparison_df)
stock_annualized_return = (1 + stock_total_return) ** (252 / trading_days) - 1
benchmark_annualized_return = (1 + benchmark_total_return) ** (252 / trading_days) - 1

daily_returns_comparison = comparison_df.pct_change().dropna()
stock_volatility = daily_returns_comparison[selected_company].std() * np.sqrt(252)
benchmark_volatility = daily_returns_comparison["NIFTY 50"].std() * np.sqrt(252)

relative_total_return = stock_total_return - benchmark_total_return
relative_annualized_return = stock_annualized_return - benchmark_annualized_return

if relative_annualized_return > 0.05:
    benchmark_assessment = "Strong Outperformance"
    benchmark_text = f"{selected_company} generated an annualized return {relative_annualized_return:.2%} higher than the NIFTY 50 over the selected historical period."
elif relative_annualized_return > 0:
    benchmark_assessment = "Outperformed"
    benchmark_text = f"{selected_company} generated a higher annualized return than the NIFTY 50 by {relative_annualized_return:.2%}."
elif relative_annualized_return > -0.05:
    benchmark_assessment = "Slight Underperformance"
    benchmark_text = f"{selected_company} generated a lower annualized return than the NIFTY 50 by {abs(relative_annualized_return):.2%}."
else:
    benchmark_assessment = "Underperformed"
    benchmark_text = f"{selected_company} generated an annualized return {abs(relative_annualized_return):.2%} below the NIFTY 50."

benchmark_fig = go.Figure()
benchmark_fig.add_trace(go.Scatter(x=normalized_comparison.index, y=normalized_comparison[selected_company], mode="lines", name=selected_company, line=dict(width=3)))
benchmark_fig.add_trace(go.Scatter(x=normalized_comparison.index, y=normalized_comparison["NIFTY 50"], mode="lines", name="NIFTY 50", line=dict(width=3)))
benchmark_fig.add_hline(y=100, line_dash="dot", annotation_text="Starting value: ₹100", annotation_position="top left")

benchmark_fig.update_layout(
    **DARK_CHART,
    height=480,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(title="Date", showgrid=False),
    yaxis=dict(title="Growth of ₹100", showgrid=True, gridcolor=DARK_GRID),
    hovermode="x unified",
    legend=LEGEND_TOP,
)
st.plotly_chart(benchmark_fig, use_container_width=True, theme=None)

benchmark_col1, benchmark_col2, benchmark_col3, benchmark_col4 = st.columns(4)

with benchmark_col1:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">{selected_company} Return</div>
            <div class="metric-value">{stock_total_return:.2%}</div>
            <div class="metric-description">Total historical return</div>
        </div>
        """
    )
with benchmark_col2:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">NIFTY 50 Return</div>
            <div class="metric-value">{benchmark_total_return:.2%}</div>
            <div class="metric-description">Total historical return</div>
        </div>
        """
    )
with benchmark_col3:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">Relative Performance</div>
            <div class="metric-value">{relative_total_return:.2%}</div>
            <div class="metric-description">Stock return minus NIFTY 50</div>
        </div>
        """
    )
with benchmark_col4:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">Benchmark Assessment</div>
            <div class="metric-value" style="font-size:1.15rem;">{benchmark_assessment}</div>
            <div class="metric-description">Based on annualized performance</div>
        </div>
        """
    )

comparison_table = pd.DataFrame({
    "Metric": ["Annualized Return", "Annual Volatility"],
    selected_company: [stock_annualized_return, stock_volatility],
    "NIFTY 50": [benchmark_annualized_return, benchmark_volatility],
})
comparison_table["Metric"] = comparison_table["Metric"].astype(str)

render_html('<div style="margin-top:1.5rem; font-size:1rem; font-weight:700; color:#0f172a;">Return & Risk Comparison</div>')
st.dataframe(
    comparison_table.style.format({selected_company: "{:.2%}", "NIFTY 50": "{:.2%}"}),
    use_container_width=True,
    hide_index=True,
)

# =============================================================================
# PEER COMPARISON — INSURANCE COMPANY SCORECARD
# =============================================================================

render_html('<div class="section-title">Insurance Peer Comparison</div>')
render_html(
    '<div style="color:#64748b; font-size:0.9rem; margin-bottom:1rem;">'
    'Comparison of listed insurance companies using historical risk and return metrics over the same two-year period.</div>'
)


@st.cache_data(ttl=3600)
def load_peer_data():
    peer_results = []
    for company_name, ticker in INSURANCE_STOCKS.items():
        try:
            peer_df = get_price_data(ticker, "2y")
            peer_close = peer_df.set_index("Date")["Close"].dropna()
            peer_metrics = performance_summary(peer_close, benchmark_close)
            peer_results.append({
                "Company": company_name,
                "Ticker": ticker,
                "Annual Return": peer_metrics["Annual Return"],
                "Annual Volatility": peer_metrics["Annual Volatility"],
                "Sharpe Ratio": peer_metrics["Sharpe Ratio"],
                "Maximum Drawdown": peer_metrics["Maximum Drawdown"],
                "Beta": peer_metrics["Beta"],
            })
        except Exception:
            continue
    return pd.DataFrame(peer_results)


with st.spinner("Comparing insurance companies..."):
    peer_df = load_peer_data()

if peer_df.empty:
    st.warning("Peer comparison data could not be loaded right now. Please refresh the dashboard and try again.")
else:
    numeric_columns = ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown", "Beta"]
    for column in numeric_columns:
        peer_df[column] = pd.to_numeric(peer_df[column], errors="coerce")
    peer_df = peer_df.dropna(subset=["Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown"])

    return_score = peer_df["Annual Return"].rank(pct=True) * 100
    sharpe_score = peer_df["Sharpe Ratio"].rank(pct=True) * 100
    volatility_score = peer_df["Annual Volatility"].rank(pct=True, ascending=False) * 100
    drawdown_score = peer_df["Maximum Drawdown"].rank(pct=True) * 100

    peer_df["Peer Score"] = return_score * 0.30 + sharpe_score * 0.30 + volatility_score * 0.20 + drawdown_score * 0.20
    peer_df["Rank"] = peer_df["Peer Score"].rank(ascending=False, method="min").astype(int)
    peer_df = peer_df.sort_values("Peer Score", ascending=False).reset_index(drop=True)

    peer_chart_df = peer_df.sort_values("Peer Score", ascending=True).copy()
    peer_chart_df["Selected"] = peer_chart_df["Company"] == selected_company
    other_peers = peer_chart_df[~peer_chart_df["Selected"]]
    selected_peer = peer_chart_df[peer_chart_df["Selected"]]

    peer_fig = go.Figure()
    peer_fig.add_trace(go.Bar(
        x=other_peers["Peer Score"], y=other_peers["Company"], orientation="h",
        name="Other Insurance Companies", marker=dict(color="#CBD5E1"),
        text=[f"{value:.1f}" for value in other_peers["Peer Score"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Peer Score: %{x:.1f}/100<extra></extra>",
    ))
    if not selected_peer.empty:
        peer_fig.add_trace(go.Bar(
            x=selected_peer["Peer Score"], y=selected_peer["Company"], orientation="h",
            name=f"Selected: {selected_company}", marker=dict(color="#3B82F6"),
            text=[f"{value:.1f}" for value in selected_peer["Peer Score"]], textposition="outside",
            hovertemplate="<b>%{y}</b><br>SELECTED COMPANY<br>Peer Score: %{x:.1f}/100<extra></extra>",
        ))

    peer_fig.update_layout(
        **DARK_CHART,
        height=420,
        margin=dict(l=20, r=80, t=40, b=20),
        title=dict(text=f"Peer Score Ranking — {selected_company} Highlighted", font=dict(size=16, color="#ffffff")),
        xaxis=dict(title="Peer Score", range=[0, max(100, peer_df["Peer Score"].max() + 10)], showgrid=True, gridcolor=DARK_GRID),
        yaxis=dict(title="", showgrid=False, categoryorder="array", categoryarray=peer_chart_df["Company"].tolist()),
        legend=LEGEND_TOP_RIGHT,
        bargap=0.25,
    )
    st.plotly_chart(peer_fig, use_container_width=True, theme=None)

    selected_peer_row = peer_df[peer_df["Company"] == selected_company]

    if not selected_peer_row.empty:
        selected_rank = int(selected_peer_row["Rank"].iloc[0])
        selected_score = float(selected_peer_row["Peer Score"].iloc[0])
        total_companies = len(peer_df)

        if selected_rank == 1:
            peer_assessment = "The selected company ranks first among the insurance peers in this scorecard."
        elif selected_rank <= 3:
            peer_assessment = "The selected company is among the top three insurance peers in this scorecard."
        elif selected_rank <= max(3, total_companies // 2):
            peer_assessment = "The selected company has a middle-to-upper peer position based on the historical metrics."
        else:
            peer_assessment = "The selected company ranks below the midpoint of the insurance peer group based on the historical metrics."

        render_html(
            f"""
            <div class="interpretation-card">
                <div class="interpretation-overall">{selected_company}: Rank {selected_rank} of {total_companies}</div>
                <div class="interpretation-overall-text">{peer_assessment} Its peer score is <strong>{selected_score:.1f}/100</strong>.</div>
                <div class="interpretation-grid">
                    <div class="interpretation-item">
                        <div class="interpretation-label">Annual Return</div>
                        <div class="interpretation-text">{selected_peer_row["Annual Return"].iloc[0]:.2%}</div>
                    </div>
                    <div class="interpretation-item">
                        <div class="interpretation-label">Annual Volatility</div>
                        <div class="interpretation-text">{selected_peer_row["Annual Volatility"].iloc[0]:.2%}</div>
                    </div>
                    <div class="interpretation-item">
                        <div class="interpretation-label">Sharpe Ratio</div>
                        <div class="interpretation-text">{selected_peer_row["Sharpe Ratio"].iloc[0]:.2f}</div>
                    </div>
                    <div class="interpretation-item">
                        <div class="interpretation-label">Maximum Drawdown</div>
                        <div class="interpretation-text">{selected_peer_row["Maximum Drawdown"].iloc[0]:.2%}</div>
                    </div>
                    <div class="interpretation-item">
                        <div class="interpretation-label">Beta</div>
                        <div class="interpretation-text">{selected_peer_row["Beta"].iloc[0]:.2f}</div>
                    </div>
                    <div class="interpretation-item">
                        <div class="interpretation-label">Peer Score</div>
                        <div class="interpretation-text">{selected_score:.1f}/100</div>
                    </div>
                </div>
                <div class="interpretation-note">
                    The peer score is a relative historical ranking, not an investment recommendation. It combines
                    return, risk-adjusted performance, volatility and maximum drawdown.
                </div>
            </div>
            """
        )

        display_peer_df = peer_df[
            ["Rank", "Company", "Annual Return", "Annual Volatility", "Sharpe Ratio", "Maximum Drawdown", "Beta", "Peer Score"]
        ].copy()
        display_peer_df.columns = ["Rank", "Company", "Annual Return", "Volatility", "Sharpe Ratio", "Maximum Drawdown", "Beta", "Peer Score"]

        render_html('<div style="margin-top:1.5rem; font-size:1rem; font-weight:700; color:#111827;">Insurance Peer Scorecard</div>')
        st.dataframe(
            display_peer_df.style.format({
                "Annual Return": "{:.2%}", "Volatility": "{:.2%}", "Sharpe Ratio": "{:.2f}",
                "Maximum Drawdown": "{:.2%}", "Beta": "{:.2f}", "Peer Score": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("How is the Peer Score calculated?"):
            st.markdown(
                """
                The Peer Score is a relative score out of 100 based on the historical performance of the
                insurance companies included in the prototype.

                **Scoring weights**
                - Annual Return: 30%
                - Sharpe Ratio: 30%
                - Annual Volatility: 20%
                - Maximum Drawdown: 20%

                Higher return and Sharpe Ratio receive higher scores. Lower volatility receives a higher score.
                A less severe maximum drawdown receives a higher score. Beta is displayed separately because
                higher or lower beta is not inherently better; it represents sensitivity to movements in the NIFTY 50.
                """
            )

# =============================================================================
# EFFICIENT FRONTIER — PORTFOLIO OPTIMIZATION
# =============================================================================

render_html('<div class="section-title">Portfolio Efficient Frontier</div>')
render_html(
    '<div style="color:#64748b; font-size:0.9rem; margin-bottom:1rem;">'
    'Portfolio optimization showing how diversification across insurance companies changes the risk-return trade-off.</div>'
)


@st.cache_data(ttl=3600)
def load_portfolio_data():
    price_series = {}
    for company_name, ticker in INSURANCE_STOCKS.items():
        try:
            company_df = get_price_data(ticker, "2y")
            company_close = company_df.set_index("Date")["Close"].dropna()
            if len(company_close) > 50:
                price_series[company_name] = company_close
        except Exception:
            continue
    if not price_series:
        return pd.DataFrame()
    prices = pd.concat(price_series, axis=1).dropna()
    return prices.pct_change().dropna()


with st.spinner("Preparing portfolio data..."):
    portfolio_returns = load_portfolio_data()

if portfolio_returns.empty:
    st.warning("Portfolio data could not be loaded. Please refresh the dashboard.")
else:
    assets = list(portfolio_returns.columns)
    n_assets = len(assets)

    if n_assets < 2:
        st.warning("At least two insurance companies are required to construct a portfolio.")
    else:
        TRADING_DAYS = 252
        annual_returns = portfolio_returns.mean() * TRADING_DAYS
        covariance_matrix = portfolio_returns.cov() * TRADING_DAYS

        def calculate_portfolio(weights):
            portfolio_return = np.dot(weights, annual_returns.values)
            portfolio_variance = np.dot(weights, np.dot(covariance_matrix.values, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            portfolio_sharpe = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
            return portfolio_return, portfolio_volatility, portfolio_sharpe

        np.random.seed(42)
        NUM_PORTFOLIOS = 5000
        portfolio_results = []
        portfolio_weights = []

        for _ in range(NUM_PORTFOLIOS):
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()
            portfolio_results.append(calculate_portfolio(weights))
            portfolio_weights.append(weights)

        portfolio_results = np.array(portfolio_results)
        portfolio_weights = np.array(portfolio_weights)

        returns_array = portfolio_results[:, 0]
        volatility_array = portfolio_results[:, 1]
        sharpe_array = portfolio_results[:, 2]

        max_sharpe_index = np.argmax(sharpe_array)
        min_volatility_index = np.argmin(volatility_array)

        max_sharpe_return = returns_array[max_sharpe_index]
        max_sharpe_volatility = volatility_array[max_sharpe_index]
        max_sharpe_value = sharpe_array[max_sharpe_index]

        min_volatility_return = returns_array[min_volatility_index]
        min_volatility_value = volatility_array[min_volatility_index]

        frontier_fig = go.Figure()
        frontier_fig.add_trace(go.Scatter(
            x=volatility_array * 100, y=returns_array * 100, mode="markers",
            marker=dict(size=4, color=sharpe_array, colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe"), opacity=0.45),
            name="Simulated Portfolios",
            hovertemplate="Risk: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
        ))
        frontier_fig.add_trace(go.Scatter(
            x=[max_sharpe_volatility * 100], y=[max_sharpe_return * 100], mode="markers",
            marker=dict(size=16, color="#16A34A", symbol="star"), name="Maximum Sharpe Portfolio",
            hovertemplate="<b>Maximum Sharpe Portfolio</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))
        frontier_fig.add_trace(go.Scatter(
            x=[min_volatility_value * 100], y=[min_volatility_return * 100], mode="markers",
            marker=dict(size=15, color="#2563EB", symbol="diamond"), name="Minimum Volatility Portfolio",
            hovertemplate="<b>Minimum Volatility Portfolio</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>",
        ))

        frontier_fig.update_layout(
            **DARK_CHART,
            height=520,
            margin=dict(l=20, r=40, t=30, b=20),
            xaxis=dict(title="Annualized Risk / Volatility (%)", showgrid=True, gridcolor=DARK_GRID),
            yaxis=dict(title="Expected Annual Return (%)", showgrid=True, gridcolor=DARK_GRID),
            legend=LEGEND_TOP_RIGHT,
        )
        st.plotly_chart(frontier_fig, use_container_width=True, theme=None)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Maximum Sharpe Return", f"{max_sharpe_return:.2%}")
            st.caption(f"Portfolio risk: {max_sharpe_volatility:.2%}")
        with col2:
            st.metric("Maximum Sharpe Ratio", f"{max_sharpe_value:.2f}")
            st.caption("Best simulated risk-adjusted portfolio")
        with col3:
            st.metric("Minimum Risk", f"{min_volatility_value:.2%}")
            st.caption(f"Expected return: {min_volatility_return:.2%}")

        max_sharpe_weights = portfolio_weights[max_sharpe_index]
        weights_df = pd.DataFrame({"Company": assets, "Weight": max_sharpe_weights})
        weights_df = weights_df[weights_df["Weight"] >= 0.01].sort_values("Weight", ascending=False)

        render_html('<div style="margin-top:1.5rem; font-size:1rem; font-weight:700; color:#111827;">Maximum Sharpe Portfolio Allocation</div>')
        weights_display = weights_df.copy()
        weights_display["Weight"] = weights_display["Weight"].map(lambda x: f"{x:.1%}")
        st.dataframe(weights_display, use_container_width=True, hide_index=True)

        if max_sharpe_value >= 1:
            portfolio_interpretation = "The simulated maximum-Sharpe portfolio provides relatively strong risk-adjusted performance under the historical assumptions."
        elif max_sharpe_value >= 0.5:
            portfolio_interpretation = "The simulated maximum-Sharpe portfolio provides a moderate risk-adjusted trade-off under the historical assumptions."
        else:
            portfolio_interpretation = "The simulated portfolio opportunities show limited risk-adjusted performance under the historical assumptions."

        render_html(
            f"""
            <div class="interpretation-card">
                <div class="interpretation-overall">Portfolio Optimization Interpretation</div>
                <div class="interpretation-overall-text">{portfolio_interpretation}</div>
                <div class="interpretation-note">
                    The efficient frontier is based on historical returns and covariance between the selected
                    insurance companies. It does not guarantee future portfolio performance.
                </div>
            </div>
            """
        )

        with st.expander("How to read the Efficient Frontier"):
            st.markdown(
                """
                Each point represents a simulated portfolio containing different combinations of the insurance companies.

                **Risk** is represented by annualized portfolio volatility.

                **Return** represents the historical annualized return implied by the portfolio weights.

                The **Maximum Sharpe Portfolio** is the simulated portfolio with the highest return relative to risk.

                The **Minimum Volatility Portfolio** is the simulated portfolio with the lowest overall volatility.

                The key idea is diversification: combining different insurance stocks can produce a different
                risk-return profile from holding individual companies alone.
                """
            )

# =============================================================================
# MONTE CARLO SIMULATION
# =============================================================================

render_html('<div class="section-title">Monte Carlo Price Simulation</div>')
render_html(
    '<div style="color:#64748b; font-size:0.9rem; margin-bottom:1rem;">'
    'A probabilistic simulation of potential one-year price paths based on the selected company’s historical return and volatility.</div>'
)

NUM_SIMULATIONS = 10000
TRADING_DAYS = 252
DISPLAY_PATHS = 100

historical_returns = stock_df["Close"].pct_change().dropna()
daily_mean_return = historical_returns.mean()
daily_volatility = historical_returns.std()
current_price = stock_df["Close"].iloc[-1]

np.random.seed(42)
random_returns = np.random.normal(loc=daily_mean_return, scale=daily_volatility, size=(TRADING_DAYS, NUM_SIMULATIONS))

price_paths = np.zeros((TRADING_DAYS + 1, NUM_SIMULATIONS))
price_paths[0] = current_price
for day in range(1, TRADING_DAYS + 1):
    price_paths[day] = price_paths[day - 1] * (1 + random_returns[day - 1])

final_prices = price_paths[-1]
median_price = np.percentile(final_prices, 50)
lower_price = np.percentile(final_prices, 5)
upper_price = np.percentile(final_prices, 95)
probability_above_current = np.mean(final_prices > current_price)
simulated_median_return = (median_price / current_price) - 1

simulation_fig = go.Figure()
days = np.arange(TRADING_DAYS + 1)

for i in range(DISPLAY_PATHS):
    simulation_fig.add_trace(go.Scatter(
        x=days, y=price_paths[:, i], mode="lines", line=dict(width=1),
        opacity=0.12, showlegend=False, hoverinfo="skip",
    ))

median_path = np.percentile(price_paths, 50, axis=1)
simulation_fig.add_trace(go.Scatter(x=days, y=median_path, mode="lines", name="Median Path", line=dict(width=3)))

lower_path = np.percentile(price_paths, 5, axis=1)
simulation_fig.add_trace(go.Scatter(x=days, y=lower_path, mode="lines", name="5th Percentile", line=dict(width=2, dash="dash")))

upper_path = np.percentile(price_paths, 95, axis=1)
simulation_fig.add_trace(go.Scatter(x=days, y=upper_path, mode="lines", name="95th Percentile", line=dict(width=2, dash="dash")))

simulation_fig.add_hline(y=current_price, line_dash="dot", annotation_text=f"Current Price: ₹{current_price:,.2f}", annotation_position="top left")

simulation_fig.update_layout(
    **DARK_CHART,
    height=520,
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(title="Trading Days Ahead", showgrid=False),
    yaxis=dict(title="Simulated Price (₹)", showgrid=True, gridcolor=DARK_GRID),
    hovermode="x unified",
    legend=LEGEND_TOP_RIGHT,
)
st.plotly_chart(simulation_fig, use_container_width=True, theme=None)

mc_col1, mc_col2, mc_col3, mc_col4 = st.columns(4)

with mc_col1:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">Current Price</div>
            <div class="metric-value">₹{current_price:,.2f}</div>
            <div class="metric-description">Latest market price</div>
        </div>
        """
    )
with mc_col2:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">Median Price</div>
            <div class="metric-value">₹{median_price:,.2f}</div>
            <div class="metric-description">50th percentile after 1 year</div>
        </div>
        """
    )
with mc_col3:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">5th–95th Range</div>
            <div class="metric-value" style="font-size:1.1rem;">₹{lower_price:,.0f} – ₹{upper_price:,.0f}</div>
            <div class="metric-description">Simulated price range</div>
        </div>
        """
    )
with mc_col4:
    render_html(
        f"""
        <div class="metric-card">
            <div class="metric-label">Probability Above Current</div>
            <div class="metric-value">{probability_above_current:.1%}</div>
            <div class="metric-description">Simulated paths ending higher</div>
        </div>
        """
    )

if simulated_median_return > 0.10:
    simulation_assessment = "Positive Simulated Outlook"
    simulation_text = f"The median simulated price after approximately one year is ₹{median_price:,.2f}, representing a simulated return of {simulated_median_return:.2%} from the current price."
elif simulated_median_return > 0:
    simulation_assessment = "Moderately Positive Simulated Outlook"
    simulation_text = f"The median simulated price is ₹{median_price:,.2f}, slightly above the current price. The simulation indicates a median return of {simulated_median_return:.2%}."
elif simulated_median_return > -0.10:
    simulation_assessment = "Moderately Negative Simulated Outlook"
    simulation_text = f"The median simulated price is ₹{median_price:,.2f}, slightly below the current price. The simulation indicates a median return of {simulated_median_return:.2%}."
else:
    simulation_assessment = "Negative Simulated Outlook"
    simulation_text = f"The median simulated price is ₹{median_price:,.2f}, representing a simulated return of {simulated_median_return:.2%}."

render_html(
    f"""
    <div class="interpretation-card">
        <div class="interpretation-overall">{simulation_assessment}</div>
        <div class="interpretation-overall-text">{simulation_text}</div>
        <div class="interpretation-grid">
            <div class="interpretation-item">
                <div class="interpretation-label">Current Price</div>
                <div class="interpretation-text">₹{current_price:,.2f}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">Median Simulated Price</div>
                <div class="interpretation-text">₹{median_price:,.2f}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">5th Percentile</div>
                <div class="interpretation-text">₹{lower_price:,.2f}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">95th Percentile</div>
                <div class="interpretation-text">₹{upper_price:,.2f}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">Probability Above Current Price</div>
                <div class="interpretation-text">{probability_above_current:.1%}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">Simulated Median Return</div>
                <div class="interpretation-text">{simulated_median_return:.2%}</div>
            </div>
        </div>
        <div class="interpretation-note">
            Monte Carlo simulations are probabilistic scenarios based on historical return and volatility.
            They are not forecasts or guarantees of future market performance.
        </div>
    </div>
    """
)

# =============================================================================
# DYNAMIC BENCHMARK INTERPRETATION
# =============================================================================

render_html(
    f"""
    <div class="interpretation-card">
        <div class="interpretation-overall">Benchmark Assessment: {benchmark_assessment}</div>
        <div class="interpretation-overall-text">{benchmark_text}</div>
        <div class="interpretation-grid">
            <div class="interpretation-item">
                <div class="interpretation-label">{selected_company} Annualized Return</div>
                <div class="interpretation-text">{stock_annualized_return:.2%}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">NIFTY 50 Annualized Return</div>
                <div class="interpretation-text">{benchmark_annualized_return:.2%}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">{selected_company} Volatility</div>
                <div class="interpretation-text">{stock_volatility:.2%}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">NIFTY 50 Volatility</div>
                <div class="interpretation-text">{benchmark_volatility:.2%}</div>
            </div>
        </div>
        <div class="interpretation-note">
            The benchmark comparison shows relative historical performance and volatility. It does not predict
            future returns or guarantee investment outcomes.
        </div>
    </div>
    """
)

# =============================================================================
# DYNAMIC RISK & RETURN INTERPRETATION
# =============================================================================

render_html('<div class="section-title">Dynamic Risk & Return Interpretation</div>')

annual_return_value = metrics["Annual Return"]
volatility_value = metrics["Annual Volatility"]
sharpe_value = metrics["Sharpe Ratio"]
drawdown_value = metrics["Maximum Drawdown"]
beta_value = metrics["Beta"]

if annual_return_value >= 0.15:
    return_text = f"The stock has generated a strong historical annualized return of {annual_return_value:.2%}."
elif annual_return_value >= 0.06:
    return_text = f"The stock has generated a positive historical annualized return of {annual_return_value:.2%}, indicating moderate historical growth."
elif annual_return_value >= 0:
    return_text = f"The stock has generated a positive but relatively modest historical annualized return of {annual_return_value:.2%}."
else:
    return_text = f"The stock has produced a negative historical annualized return of {annual_return_value:.2%} over the analysis period."

if volatility_value < 0.15:
    volatility_text = f"Annual volatility of {volatility_value:.2%} indicates relatively low historical price fluctuation."
elif volatility_value <= 0.25:
    volatility_text = f"Annual volatility of {volatility_value:.2%} indicates a moderate level of historical price fluctuation."
else:
    volatility_text = f"Annual volatility of {volatility_value:.2%} indicates relatively high historical price fluctuation."

if sharpe_value >= 1:
    sharpe_text = f"The Sharpe Ratio of {sharpe_value:.2f} indicates strong historical risk-adjusted performance."
elif sharpe_value >= 0.5:
    sharpe_text = f"The Sharpe Ratio of {sharpe_value:.2f} indicates reasonably good historical risk-adjusted performance."
elif sharpe_value >= 0:
    sharpe_text = f"The Sharpe Ratio of {sharpe_value:.2f} indicates weak historical risk-adjusted performance."
else:
    sharpe_text = f"The Sharpe Ratio of {sharpe_value:.2f} indicates that the historical return did not compensate sufficiently for the risk taken under the model assumptions."

if drawdown_value >= -0.15:
    drawdown_text = f"The maximum drawdown of {drawdown_value:.2%} indicates a relatively limited historical peak-to-trough decline."
elif drawdown_value >= -0.25:
    drawdown_text = f"The maximum drawdown of {drawdown_value:.2%} indicates a moderate level of historical downside exposure."
else:
    drawdown_text = f"The maximum drawdown of {drawdown_value:.2%} indicates a substantial historical peak-to-trough decline."

if beta_value < 0.8:
    beta_text = f"Beta of {beta_value:.2f} indicates relatively low sensitivity to movements in the NIFTY 50."
elif beta_value <= 1.2:
    beta_text = f"Beta of {beta_value:.2f} indicates sensitivity broadly in line with the NIFTY 50."
else:
    beta_text = f"Beta of {beta_value:.2f} indicates higher sensitivity to movements in the NIFTY 50."

positive_signals = 0
negative_signals = 0
positive_signals += 1 if annual_return_value > 0 else 0
negative_signals += 1 if annual_return_value <= 0 else 0
positive_signals += 1 if volatility_value <= 0.25 else 0
negative_signals += 1 if volatility_value > 0.25 else 0
positive_signals += 1 if sharpe_value > 0.5 else 0
negative_signals += 1 if sharpe_value <= 0.5 else 0
positive_signals += 1 if drawdown_value >= -0.25 else 0
negative_signals += 1 if drawdown_value < -0.25 else 0

if positive_signals >= 3:
    overall_text = "Overall, the historical profile shows a relatively favourable balance between return and risk."
elif negative_signals >= 3:
    overall_text = "Overall, the historical profile shows notable risk or weaker risk-adjusted performance that warrants caution."
else:
    overall_text = "Overall, the historical profile is mixed, with both positive and negative risk-return characteristics."

render_html(
    f"""
    <div class="interpretation-card">
        <div class="interpretation-overall">Overall Assessment</div>
        <div class="interpretation-overall-text">{overall_text}</div>
        <div class="interpretation-grid">
            <div class="interpretation-item">
                <div class="interpretation-label">Return</div>
                <div class="interpretation-text">{return_text}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">Risk</div>
                <div class="interpretation-text">{volatility_text}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">Risk-adjusted Performance</div>
                <div class="interpretation-text">{sharpe_text}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">Downside Risk</div>
                <div class="interpretation-text">{drawdown_text}</div>
            </div>
            <div class="interpretation-item">
                <div class="interpretation-label">Market Sensitivity</div>
                <div class="interpretation-text">{beta_text}</div>
            </div>
        </div>
        <div class="interpretation-note">
            Interpretation is generated automatically from the historical metrics displayed above.
            It is not a buy/sell recommendation.
        </div>
    </div>
    """
)

# =============================================================================
# HISTORICAL DATA
# =============================================================================

render_html('<div class="section-title">Historical Market Data</div>')
st.caption("Most recent 20 trading days for the selected insurance company.")
st.dataframe(stock_df.tail(20), use_container_width=True, hide_index=True)

# =============================================================================
# FOOTER
# =============================================================================

render_html(
    """
    <div class="footer">
        InsureInvest • Historical market analytics prototype
        <br>
        Data sourced through Yahoo Finance via yfinance.
    </div>
    """
)
