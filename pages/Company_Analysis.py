import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from analytics import performance_summary, daily_returns
from data import BENCHMARK, INSURANCE_STOCKS, COMPANY_CATEGORIES, get_price_data
from technicals import add_technical_indicators

st.set_page_config(page_title="InsureInvest — Company Analysis", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp{background:#f6f8fb}.block-container{max-width:1450px;padding-top:1.5rem;padding-bottom:3rem}
.hero{background:linear-gradient(135deg,#111827,#1e3a5f);padding:2rem 2.4rem;border-radius:18px;margin-bottom:1.3rem}.hero h1{color:white;margin:0;font-size:2.35rem}.hero p{color:#cbd5e1;margin:.4rem 0 0}
.card{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:1rem 1.2rem}.small{color:#64748b;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;font-weight:650}.big{color:#111827;font-size:1.7rem;font-weight:750;margin-top:.3rem}
.explain{background:white;border:1px solid #e2e8f0;border-left:5px solid #2563eb;border-radius:14px;padding:1.15rem 1.4rem;margin:.7rem 0 1.4rem;color:#334155;line-height:1.6}.note{color:#64748b;margin-top:-.45rem;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_company(ticker, period="2y"):
    return add_technical_indicators(get_price_data(ticker, period))

@st.cache_data(ttl=900, show_spinner=False)
def load_benchmark(period="2y"):
    return get_price_data(BENCHMARK, period)

@st.cache_data(ttl=900, show_spinner=False)
def load_peer_returns(items, period="2y"):
    series = {}
    for company, ticker in items:
        try:
            df = get_price_data(ticker, period)
            series[company] = daily_returns(df.set_index("Date")["Close"])
        except Exception:
            pass
    return pd.concat(series, axis=1).dropna() if len(series) >= 2 else pd.DataFrame()

def explanation(text):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)

companies = list(INSURANCE_STOCKS.keys())
default = st.session_state.get("selected_company", companies[0])
if default not in companies:
    default = companies[0]

st.sidebar.markdown("## InsureInvest")
st.sidebar.caption("Company Analysis")
st.sidebar.markdown("---")
selected = st.sidebar.selectbox("Select a company", companies, index=companies.index(default))
st.session_state["selected_company"] = selected
ticker = INSURANCE_STOCKS[selected]
category = COMPANY_CATEGORIES.get(selected, "Investment")
st.sidebar.markdown(f"**NSE:** `{ticker}`")
st.sidebar.caption(category)

try:
    with st.spinner(f"Loading {selected} market data..."):
        stock = load_company(ticker)
        bench = load_benchmark()
except Exception as exc:
    st.error("Market data could not be loaded right now.")
    st.info("Yahoo Finance may temporarily rate-limit requests. Refresh and retry in a few seconds.")
    st.caption(f"Technical details: {exc}")
    st.stop()

close = stock.set_index("Date")["Close"]
bench_close = bench.set_index("Date")["Close"]
metrics = performance_summary(close, bench_close)
latest = stock.iloc[-1]
price = float(latest["Close"])

st.markdown(f'<div class="hero"><h1>{selected}</h1><p>{category} • NSE: {ticker} • Full historical company analytics</p></div>', unsafe_allow_html=True)

st.subheader("Performance Snapshot")
cols = st.columns(5)
items = [
    ("Annual Return", f"{metrics['Annual Return']:.2%}", "Historical annualized return"),
    ("Annual Volatility", f"{metrics['Annual Volatility']:.2%}", "Historical price fluctuation"),
    ("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", "Return per unit of risk"),
    ("Maximum Drawdown", f"{metrics['Maximum Drawdown']:.2%}", "Largest peak-to-trough fall"),
    ("Beta", f"{metrics['Beta']:.2f}", "Sensitivity to NIFTY 50"),
]
for col, (label, value, desc) in zip(cols, items):
    with col:
        st.markdown(f'<div class="card"><div class="small">{label}</div><div class="big">{value}</div><div style="color:#94a3b8;font-size:.75rem">{desc}</div></div>', unsafe_allow_html=True)
explanation(f"<b>Quick read:</b> {selected} is at <b>₹{price:,.2f}</b>. Historical annual return is <b>{metrics['Annual Return']:.2%}</b>, volatility is <b>{metrics['Annual Volatility']:.2%}</b>, and Sharpe is <b>{metrics['Sharpe Ratio']:.2f}</b>. These are historical measures, not guarantees of future performance.")

st.header("1. Price & Technical Analysis")
st.markdown('<div class="note">Trend, moving averages, volatility bands and momentum indicators.</div>', unsafe_allow_html=True)
fig = go.Figure(go.Candlestick(x=stock.Date, open=stock.Open, high=stock.High, low=stock.Low, close=stock.Close, name=selected))
for col, name, dash in [("SMA_50","50-Day SMA","solid"),("SMA_200","200-Day SMA","solid"),("BB_Upper","Bollinger Upper","dash"),("BB_Lower","Bollinger Lower","dash")]:
    fig.add_trace(go.Scatter(x=stock.Date, y=stock[col], mode="lines", name=name, line=dict(width=1.6, dash=dash)))
fig.update_layout(template="plotly_dark", height=540, title="Historical Price with Moving Averages and Bollinger Bands", xaxis_rangeslider_visible=False, hovermode="x unified")
st.plotly_chart(fig, use_container_width=True, theme=None)
above50 = price > float(latest["SMA_50"])
above200 = price > float(latest["SMA_200"])
explanation(f"<b>What this shows:</b> Candlesticks show daily price movement. The 50-day and 200-day SMAs describe short- and long-term trend, while Bollinger Bands show a volatility envelope. <b>{selected}</b> is currently {'above' if above50 else 'below'} its 50-day average and {'above' if above200 else 'below'} its 200-day average.")

rsi = float(latest["RSI_14"])
rsi_state = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
rfig = go.Figure(go.Scatter(x=stock.Date, y=stock.RSI_14, mode="lines", name="RSI-14"))
rfig.add_hline(y=70, line_dash="dash", annotation_text="70 — overbought")
rfig.add_hline(y=30, line_dash="dash", annotation_text="30 — oversold")
rfig.update_layout(template="plotly_dark", height=320, title="RSI-14 Momentum Indicator", yaxis=dict(range=[0,100]), hovermode="x unified")
st.plotly_chart(rfig, use_container_width=True, theme=None)
explanation(f"<b>RSI:</b> The latest RSI is <b>{rsi:.1f}</b>, indicating <b>{rsi_state} momentum</b>. RSI is a momentum indicator and should not be treated as a standalone buy/sell signal.")

macd = float(latest["MACD"])
signal = float(latest["MACD_Signal"])
macd_state = "positive" if macd > signal else "negative" if macd < signal else "neutral"
mfig = go.Figure()
mfig.add_trace(go.Scatter(x=stock.Date, y=stock.MACD, mode="lines", name="MACD"))
mfig.add_trace(go.Scatter(x=stock.Date, y=stock.MACD_Signal, mode="lines", name="Signal"))
mfig.add_trace(go.Bar(x=stock.Date, y=stock.MACD_Histogram, name="Histogram", opacity=.5))
mfig.update_layout(template="plotly_dark", height=350, title="MACD Momentum and Signal Line", hovermode="x unified")
st.plotly_chart(mfig, use_container_width=True, theme=None)
explanation(f"<b>MACD:</b> MACD is <b>{macd:.2f}</b> versus a signal value of <b>{signal:.2f}</b>, giving a <b>{macd_state} momentum reading</b>.")

trend = "positive" if above50 and above200 else "weak" if not above50 and not above200 else "mixed"
explanation(f"<b>Technical summary:</b> Trend is <b>{trend}</b>, RSI is <b>{rsi_state}</b>, and MACD momentum is <b>{macd_state}</b>. This is a historical technical snapshot, not a forecast.")

st.header("2. Risk, Return & Benchmark")
st.markdown('<div class="note">Historical performance and risk relative to the NIFTY 50.</div>', unsafe_allow_html=True)
bench_metrics = performance_summary(bench_close, bench_close)
risk = pd.DataFrame({"Metric":["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown","Beta"], selected:[metrics["Annual Return"],metrics["Annual Volatility"],metrics["Sharpe Ratio"],metrics["Maximum Drawdown"],metrics["Beta"]], "NIFTY 50":[bench_metrics["Annual Return"],bench_metrics["Annual Volatility"],bench_metrics["Sharpe Ratio"],bench_metrics["Maximum Drawdown"],1.0]})
st.dataframe(risk, use_container_width=True, hide_index=True)
explanation(f"<b>Risk interpretation:</b> Sharpe measures historical return relative to volatility, Beta measures sensitivity to NIFTY 50 movements, and maximum drawdown captures the largest historical fall from a previous peak. {selected} has Sharpe <b>{metrics['Sharpe Ratio']:.2f}</b> and Beta <b>{metrics['Beta']:.2f}</b>.")

sn = close / close.iloc[0] * 100
bn = bench_close / bench_close.iloc[0] * 100
gfig = go.Figure()
gfig.add_trace(go.Scatter(x=sn.index, y=sn, mode="lines", name=selected))
gfig.add_trace(go.Scatter(x=bn.index, y=bn, mode="lines", name="NIFTY 50"))
gfig.update_layout(template="plotly_dark", height=420, title="Growth of ₹100: Company vs NIFTY 50", yaxis_title="Indexed Value", hovermode="x unified")
st.plotly_chart(gfig, use_container_width=True, theme=None)
relative = "outperformed" if sn.iloc[-1] > bn.iloc[-1] else "underperformed"
explanation(f"Both series start at 100 for comparability. Over the displayed historical period, <b>{selected}</b> has <b>{relative}</b> the NIFTY 50 based on the indexed endpoints.")

st.header("3. Insurance Peer Comparison")
pure = [(k,v) for k,v in INSURANCE_STOCKS.items() if COMPANY_CATEGORIES.get(k) == "Pure Insurance"]
peer_rows = []
with st.spinner("Loading insurance peers..."):
    for name, tk in pure:
        try:
            pdf = load_company(tk)
            pc = pdf.set_index("Date")["Close"]
            pm = performance_summary(pc, bench_close)
            peer_rows.append({"Company":name,"Annual Return":pm["Annual Return"],"Volatility":pm["Annual Volatility"],"Sharpe":pm["Sharpe Ratio"],"Max Drawdown":pm["Maximum Drawdown"],"Beta":pm["Beta"]})
        except Exception:
            pass
if peer_rows:
    peers = pd.DataFrame(peer_rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
    st.dataframe(peers, use_container_width=True, hide_index=True)
    pfig = go.Figure(go.Bar(x=peers["Company"], y=peers["Sharpe"], name="Sharpe Ratio"))
    pfig.update_layout(template="plotly_dark", height=390, title="Sharpe Ratio Ranking Across Pure-Insurance Peers", xaxis_title="Company", yaxis_title="Sharpe Ratio")
    st.plotly_chart(pfig, use_container_width=True, theme=None)
    if selected in peers["Company"].values:
        rank = int(peers.index[peers["Company"] == selected][0] + 1)
        explanation(f"<b>Peer ranking:</b> {selected} ranks <b>#{rank} of {len(peers)}</b> by historical Sharpe ratio in the available pure-insurance peer set.")
    else:
        explanation(f"{selected} is not in the pure-insurance peer subset, so no peer rank is assigned.")
else:
    st.warning("No peer data is currently available.")

st.header("4. Efficient Frontier")
st.markdown('<div class="note">Simulated historical portfolio combinations showing the risk-return trade-off.</div>', unsafe_allow_html=True)
returns = load_peer_returns(tuple(pure))
if returns.shape[1] >= 3:
    mu = returns.mean() * 252
    cov = returns.cov() * 252
    n = len(mu)
    rng = np.random.default_rng(42)
    weights = rng.dirichlet(np.ones(n), size=5000)
    port_ret = weights @ mu.values
    port_vol = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov.values, weights))
    rf = 0.06
    sharpe = np.divide(port_ret-rf, port_vol, out=np.full_like(port_ret,np.nan), where=port_vol>0)
    best = int(np.nanargmax(sharpe))
    efig = go.Figure(go.Scatter(x=port_vol, y=port_ret, mode="markers", marker=dict(size=4,color=sharpe,colorscale="Viridis",showscale=True,colorbar=dict(title="Sharpe")), name="Simulated portfolios"))
    efig.add_trace(go.Scatter(x=[port_vol[best]], y=[port_ret[best]], mode="markers+text", text=["Max Sharpe"], textposition="top center", marker=dict(size=13,symbol="star"), name="Maximum Sharpe"))
    if selected in mu.index:
        efig.add_trace(go.Scatter(x=[math.sqrt(cov.loc[selected,selected])], y=[mu[selected]], mode="markers+text", text=[selected], textposition="bottom center", marker=dict(size=12,symbol="diamond"), name=selected))
    efig.update_layout(template="plotly_dark", height=520, title="Efficient Frontier: Historical Risk vs Return", xaxis_title="Annualized Volatility", yaxis_title="Annualized Return", hovermode="closest")
    st.plotly_chart(efig, use_container_width=True, theme=None)
    explanation("Each dot represents a simulated portfolio of the pure-insurance companies. The star is the simulated portfolio with the highest Sharpe ratio. The diamond, when present, shows the selected company's standalone historical risk-return position. This is a historical portfolio-efficiency exercise, not a forecast.")
else:
    st.warning("At least three peer return series are required to build the Efficient Frontier.")

st.header("5. Monte Carlo Scenario Simulation")
st.markdown('<div class="note">A probabilistic 1-year price-path simulation based on historical daily return and volatility.</div>', unsafe_allow_html=True)
sims = st.slider("Number of simulations",1000,10000,3000,500,key="mc_sims")
days = 252
hist_ret = daily_returns(close).dropna()
mu_daily = float(hist_ret.mean())
sigma_daily = float(hist_ret.std())
rng = np.random.default_rng(42)
shocks = rng.normal(mu_daily,sigma_daily,size=(days,sims))
paths = price * np.exp(np.cumsum(shocks,axis=0))
finals = paths[-1]
percentiles = np.percentile(finals,[5,25,50,75,95])
mcfig = go.Figure()
step = max(1,sims//80)
mcfig.add_trace(go.Scatter(x=np.arange(days), y=paths[:,::step], mode="lines", line=dict(width=1), opacity=.18, showlegend=False))
mcfig.add_trace(go.Scatter(x=np.arange(days), y=np.median(paths,axis=1), mode="lines", line=dict(width=3), name="Median path"))
mcfig.add_hline(y=price,line_dash="dash",annotation_text="Current price")
mcfig.update_layout(template="plotly_dark",height=520,title=f"{selected}: 1-Year Monte Carlo Price Paths",xaxis_title="Trading Days",yaxis_title="Simulated Price",hovermode="x unified")
st.plotly_chart(mcfig,use_container_width=True,theme=None)
mc_table = pd.DataFrame({"Scenario":["5th percentile","25th percentile","Median","75th percentile","95th percentile"],"Simulated price":percentiles})
st.dataframe(mc_table,use_container_width=True,hide_index=True)
explanation(f"The simulation uses <b>{sims:,} trials</b> and the selected company's historical daily return distribution. The median simulated ending price is <b>₹{percentiles[2]:,.2f}</b>; the 5th–95th percentile range is <b>₹{percentiles[0]:,.2f}–₹{percentiles[4]:,.2f}</b>. These are scenarios, not predicted prices.")

st.header("6. Black-Scholes Option Valuation")
st.markdown('<div class="note">Theoretical European call and put values using current price and historical volatility.</div>', unsafe_allow_html=True)
b1,b2,b3,b4 = st.columns(4)
with b1: strike = st.number_input("Strike price (₹)",min_value=0.01,value=round(price,2),step=1.0)
with b2: expiry = st.number_input("Time to expiry (years)",min_value=0.01,value=1.0,step=0.25)
with b3: rate = st.number_input("Risk-free rate",min_value=0.0,max_value=1.0,value=0.06,step=0.01,format="%.2f")
with b4: dividend = st.number_input("Dividend yield",min_value=0.0,max_value=1.0,value=0.0,step=0.01,format="%.2f")
vol = float(metrics["Annual Volatility"])
S,K,T,r,q = price,float(strike),float(expiry),float(rate),float(dividend)
if vol > 0 and T > 0:
    d1 = (math.log(S/K)+(r-q+0.5*vol**2)*T)/(vol*math.sqrt(T))
    d2 = d1-vol*math.sqrt(T)
    call = S*math.exp(-q*T)*norm.cdf(d1)-K*math.exp(-r*T)*norm.cdf(d2)
    put = K*math.exp(-r*T)*norm.cdf(-d2)-S*math.exp(-q*T)*norm.cdf(-d1)
    oc = st.columns(4)
    vals=[("Current Price",f"₹{S:,.2f}"),("Historical Volatility",f"{vol:.2%}"),("Theoretical Call",f"₹{call:,.2f}"),("Theoretical Put",f"₹{put:,.2f}")]
    for col,(lab,val) in zip(oc,vals):
        with col: st.metric(lab,val)
    opt=pd.DataFrame({"Parameter":["S — Current price","K — Strike","T — Expiry","r — Risk-free rate","q — Dividend yield","σ — Historical volatility","d1","d2"],"Value":[S,K,T,r,q,vol,d1,d2]})
    st.dataframe(opt,use_container_width=True,hide_index=True)
    explanation(f"Black-Scholes gives a theoretical European <b>call of ₹{call:,.2f}</b> and <b>put of ₹{put:,.2f}</b> for the supplied assumptions. This is a model valuation, not the observed market option price.")
else:
    st.warning("Black-Scholes requires positive historical volatility and positive expiry.")

st.caption("InsureInvest • Historical market analytics prototype • Data via Yahoo Finance / yfinance")
