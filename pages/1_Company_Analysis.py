import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from scipy.stats import norm

from analytics import performance_summary, daily_returns
from data import BENCHMARK, INSURANCE_STOCKS, COMPANY_CATEGORIES, get_price_data
from technicals import add_technical_indicators

st.set_page_config(page_title="InsureInvest — Company Analysis", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp { background:#f6f8fb; }
.block-container { max-width:1450px; padding-top:1.5rem; padding-bottom:3rem; }
.hero { background:linear-gradient(135deg,#111827,#1e3a5f); padding:2.2rem 2.5rem; border-radius:18px; margin-bottom:1.4rem; }
.hero h1 { color:white; font-size:2.4rem; margin:0; }
.hero p { color:#cbd5e1; margin:.4rem 0 0; }
.card { background:white; border:1px solid #e5e7eb; border-radius:14px; padding:1.1rem 1.25rem; box-shadow:0 4px 15px rgba(15,23,42,.05); }
.small { color:#64748b; font-size:.82rem; text-transform:uppercase; letter-spacing:.04em; font-weight:650; }
.big { color:#111827; font-size:1.75rem; font-weight:750; margin-top:.35rem; }
.explain { background:white; border:1px solid #e2e8f0; border-left:5px solid #2563eb; border-radius:14px; padding:1.15rem 1.4rem; margin:.75rem 0 1.35rem; color:#334155; line-height:1.6; }
.graph-title { font-size:1.2rem; font-weight:750; color:#111827; margin:1.1rem 0 .2rem; }
.graph-subtitle { color:#64748b; font-size:.9rem; margin-bottom:.4rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def load_company(ticker, period="2y"):
    return add_technical_indicators(get_price_data(ticker, period))

@st.cache_data(ttl=900, show_spinner=False)
def load_benchmark(period="2y"):
    return get_price_data(BENCHMARK, period)

@st.cache_data(ttl=900, show_spinner=False)
def load_peer_returns(companies, period="2y"):
    output = {}
    for company, ticker in companies:
        try:
            df = get_price_data(ticker, period)
            output[company] = daily_returns(df.set_index("Date")["Close"])
        except Exception:
            continue
    if not output:
        return pd.DataFrame()
    return pd.concat(output, axis=1).dropna(how="all").ffill().dropna()

def graph_explanation(text):
    st.markdown(f'<div class="explain"><b>What this shows:</b> {text}</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## InsureInvest")
st.sidebar.caption("Company Analysis")
st.sidebar.markdown("---")

company_options = list(INSURANCE_STOCKS.keys())
default_company = st.session_state.get("selected_company", company_options[0])
if default_company not in company_options:
    default_company = company_options[0]
selected_company = st.sidebar.selectbox("Select a company", company_options, index=company_options.index(default_company))
st.session_state["selected_company"] = selected_company
selected_ticker = INSURANCE_STOCKS[selected_company]
category = COMPANY_CATEGORIES.get(selected_company, "Investment")

st.sidebar.markdown(f"**NSE:** `{selected_ticker}`")
st.sidebar.caption(category)
st.sidebar.markdown("---")
if st.sidebar.button("← Back to Investment Planner", use_container_width=True):
    st.switch_page("app.py")

st.markdown(f"""
<div class="hero">
<h1>{selected_company}</h1>
<p>{category} &nbsp;•&nbsp; NSE: {selected_ticker} &nbsp;•&nbsp; Full company analytics</p>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner(f"Loading {selected_company} market data..."):
        stock_df = load_company(selected_ticker, "2y")
        benchmark_df = load_benchmark("2y")
except Exception as exc:
    st.error("Market data could not be loaded right now.")
    st.info("Yahoo Finance can temporarily rate-limit requests. Wait a few seconds and retry.")
    st.caption(f"Technical details: {exc}")
    if st.button("Retry"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

stock_close = stock_df.set_index("Date")["Close"]
benchmark_close = benchmark_df.set_index("Date")["Close"]
metrics = performance_summary(stock_close, benchmark_close)
latest = stock_df.iloc[-1]
latest_price = float(latest["Close"])
latest_date = pd.to_datetime(latest["Date"])

st.subheader("Performance Snapshot")
cols = st.columns(5)
metric_data = [
    ("Annual Return", f"{metrics['Annual Return']:.2%}", "Historical annualized return"),
    ("Annual Volatility", f"{metrics['Annual Volatility']:.2%}", "Historical price fluctuation"),
    ("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}", "Return relative to risk"),
    ("Maximum Drawdown", f"{metrics['Maximum Drawdown']:.2%}", "Largest peak-to-trough fall"),
    ("Beta", f"{metrics['Beta']:.2f}", "Sensitivity to NIFTY 50"),
]
for c, (label, value, desc) in zip(cols, metric_data):
    with c:
        st.markdown(f'<div class="card"><div class="small">{label}</div><div class="big">{value}</div><div style="color:#94a3b8;font-size:.75rem">{desc}</div></div>', unsafe_allow_html=True)

st.markdown(f'<div class="explain"><b>Quick read:</b> {selected_company} is trading at <b>₹{latest_price:,.2f}</b>. Its historical annual return is <b>{metrics["Annual Return"]:.2%}</b>, volatility is <b>{metrics["Annual Volatility"]:.2%}</b>, and its Sharpe ratio is <b>{metrics["Sharpe Ratio"]:.2f}</b>. These describe historical behaviour, not guaranteed future performance.</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Price & Technicals", "Risk & Return", "Peer Comparison", "Efficient Frontier", "Monte Carlo", "Black-Scholes"
])

with tab1:
    st.subheader("Price & Technical Indicators")
    st.markdown('<div class="graph-title">Historical Price with Moving Averages & Bollinger Bands</div><div class="graph-subtitle">Candlestick price history with trend and volatility indicators.</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_df["Date"], open=stock_df["Open"], high=stock_df["High"], low=stock_df["Low"], close=stock_df["Close"], name=selected_company))
    for col, name, dash in [("SMA_50", "50-Day SMA", "solid"), ("SMA_200", "200-Day SMA", "solid"), ("BB_Upper", "Bollinger Upper", "dash"), ("BB_Lower", "Bollinger Lower", "dash")]:
        fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df[col], mode="lines", name=name, line=dict(width=1.7, dash=dash)))
    fig.update_layout(template="plotly_dark", height=560, margin=dict(l=15,r=15,t=25,b=15), title=f"{selected_company}: Price Trend & Volatility", xaxis_rangeslider_visible=False, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, theme=None)
    close = float(latest["Close"]); sma50 = float(latest["SMA_50"]); sma200 = float(latest["SMA_200"]); upper = float(latest["BB_Upper"]); lower = float(latest["BB_Lower"])
    trend = "positive" if close > sma50 and close > sma200 else "weak" if close < sma50 and close < sma200 else "mixed"
    band = "above the upper Bollinger Band" if close > upper else "below the lower Bollinger Band" if close < lower else "inside the Bollinger Bands"
    graph_explanation(f"The current price is <b>₹{close:,.2f}</b>, which gives a <b>{trend}</b> longer-term trend relative to the 50-day and 200-day averages. The price is currently <b>{band}</b>, which indicates its present position within the recent volatility range.")

    st.markdown('<div class="graph-title">RSI-14 Momentum</div><div class="graph-subtitle">Relative Strength Index on a 0–100 scale.</div>', unsafe_allow_html=True)
    rsi_fig = go.Figure(go.Scatter(x=stock_df["Date"], y=stock_df["RSI_14"], mode="lines", name="RSI-14"))
    rsi_fig.add_hline(y=70, line_dash="dash", annotation_text="70 — overbought")
    rsi_fig.add_hline(y=30, line_dash="dash", annotation_text="30 — oversold")
    rsi_fig.update_layout(template="plotly_dark", height=300, yaxis=dict(range=[0,100]), title=f"{selected_company}: RSI-14", hovermode="x unified")
    st.plotly_chart(rsi_fig, use_container_width=True, theme=None)
    rsi = float(latest["RSI_14"])
    rsi_state = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
    graph_explanation(f"The latest RSI is <b>{rsi:.1f}</b>, indicating <b>{rsi_state}</b> momentum. Values above 70 can indicate strong upward momentum that is becoming overextended, while values below 30 can indicate strong downward momentum.")

    st.markdown('<div class="graph-title">MACD Momentum & Signal</div><div class="graph-subtitle">MACD, signal line and histogram show changes in momentum.</div>', unsafe_allow_html=True)
    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["MACD"], mode="lines", name="MACD"))
    macd_fig.add_trace(go.Scatter(x=stock_df["Date"], y=stock_df["MACD_Signal"], mode="lines", name="Signal"))
    macd_fig.add_trace(go.Bar(x=stock_df["Date"], y=stock_df["MACD_Histogram"], name="Histogram", opacity=.5))
    macd_fig.update_layout(template="plotly_dark", height=330, title=f"{selected_company}: MACD", hovermode="x unified")
    st.plotly_chart(macd_fig, use_container_width=True, theme=None)
    macd = float(latest["MACD"]); signal = float(latest["MACD_Signal"])
    macd_state = "positive" if macd > signal else "negative" if macd < signal else "neutral"
    graph_explanation(f"The latest MACD is <b>{macd:.2f}</b> versus a signal value of <b>{signal:.2f}</b>. This gives a <b>{macd_state}</b> momentum signal because the MACD is {'above' if macd > signal else 'below' if macd < signal else 'at'} its signal line.")

    st.subheader("Technical Signal Summary")
    st.dataframe(pd.DataFrame({"Indicator":["Trend","RSI","MACD","Bollinger Bands"],"Latest value":[trend.title(),f"{rsi:.1f}",f"{macd:.2f}",band.title()],"Simple interpretation":["Price vs 50/200-day averages",f"Momentum condition: {rsi_state}",f"MACD vs signal: {macd_state}","Current price vs volatility bands"]}), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Risk, Return & Benchmark")
    benchmark_metrics = performance_summary(benchmark_close, benchmark_close)
    risk_table = pd.DataFrame({
        "Metric":["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown","Beta"],
        selected_company:[metrics["Annual Return"],metrics["Annual Volatility"],metrics["Sharpe Ratio"],metrics["Maximum Drawdown"],metrics["Beta"]],
        "NIFTY 50 Benchmark":[benchmark_metrics["Annual Return"],benchmark_metrics["Annual Volatility"],benchmark_metrics["Sharpe Ratio"],benchmark_metrics["Maximum Drawdown"],1.0],
    })
    formatted = risk_table.copy()
    for col in [selected_company, "NIFTY 50 Benchmark"]:
        formatted[col] = [f"{x:.2%}" if i in [0,1,3] else f"{x:.2f}" for i,x in enumerate(formatted[col])]
    st.dataframe(formatted, use_container_width=True, hide_index=True)

    st.markdown('<div class="graph-title">Growth of ₹100: Company vs NIFTY 50</div><div class="graph-subtitle">Both series are indexed to 100 at the start of the analysis period.</div>', unsafe_allow_html=True)
    fig = go.Figure()
    stock_norm = stock_close / stock_close.iloc[0] * 100
    bench_norm = benchmark_close / benchmark_close.iloc[0] * 100
    fig.add_trace(go.Scatter(x=stock_norm.index, y=stock_norm, mode="lines", name=selected_company))
    fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, mode="lines", name="NIFTY 50"))
    fig.update_layout(template="plotly_dark", height=420, title=f"{selected_company} vs NIFTY 50", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, theme=None)
    company_end = float(stock_norm.iloc[-1]); benchmark_end = float(bench_norm.iloc[-1])
    graph_explanation(f"If ₹100 had followed each historical price series over the same period, it would now be indexed at <b>{company_end:.1f}</b> for {selected_company} versus <b>{benchmark_end:.1f}</b> for the NIFTY 50. A higher indexed value means stronger historical price growth over this period.")

    st.markdown(f'<div class="explain"><b>How to read the metrics:</b> Sharpe measures historical return relative to volatility; higher is generally better. Beta above 1 means greater sensitivity to NIFTY 50 movements. Maximum drawdown is the largest historical peak-to-trough decline.</div>', unsafe_allow_html=True)

with tab3:
    st.subheader("Listed Insurance Peer Comparison")
    pure_insurance = {k:v for k,v in INSURANCE_STOCKS.items() if COMPANY_CATEGORIES.get(k) == "Pure Insurance"}
    peer_rows=[]
    with st.spinner("Loading listed insurance peers..."):
        for company,ticker in pure_insurance.items():
            try:
                df=load_company(ticker,"2y")
                m=performance_summary(df.set_index("Date")["Close"],benchmark_close)
                peer_rows.append({"Company":company,"Annual Return":m["Annual Return"],"Volatility":m["Annual Volatility"],"Sharpe":m["Sharpe Ratio"],"Max Drawdown":m["Maximum Drawdown"],"Beta":m["Beta"]})
            except Exception:
                pass
    if peer_rows:
        peers=pd.DataFrame(peer_rows).sort_values("Sharpe",ascending=False)
        peers_display=peers.copy()
        peers_display["Annual Return"]=peers_display["Annual Return"].map(lambda x:f"{x:.2%}")
        peers_display["Volatility"]=peers_display["Volatility"].map(lambda x:f"{x:.2%}")
        peers_display["Sharpe"]=peers_display["Sharpe"].map(lambda x:f"{x:.2f}")
        peers_display["Max Drawdown"]=peers_display["Max Drawdown"].map(lambda x:f"{x:.2%}")
        peers_display["Beta"]=peers_display["Beta"].map(lambda x:f"{x:.2f}")
        st.dataframe(peers_display,use_container_width=True,hide_index=True)
        st.markdown('<div class="graph-title">Peer Sharpe Ratio Ranking</div><div class="graph-subtitle">Companies are ranked by historical risk-adjusted performance.</div>', unsafe_allow_html=True)
        fig=go.Figure(go.Bar(x=peers["Company"],y=peers["Sharpe"],name="Sharpe Ratio"))
        fig.update_layout(template="plotly_dark",height=420,title="Sharpe Ratio by Listed Pure-Insurance Company",xaxis_tickangle=-35)
        st.plotly_chart(fig,use_container_width=True,theme=None)
        rank = int(peers.reset_index(drop=True).index[peers["Company"].eq(selected_company)][0] + 1) if selected_company in peers["Company"].values else None
        best = peers.iloc[0]
        graph_explanation(f"{selected_company} has a historical Sharpe ratio of <b>{metrics['Sharpe Ratio']:.2f}</b>. The highest-ranked peer in this comparison is <b>{best['Company']}</b> with a Sharpe ratio of <b>{best['Sharpe']:.2f}</b>. {'The selected company ranks ' + str(rank) + ' of ' + str(len(peers)) + ' on Sharpe.' if rank else 'The selected company is not currently available in the pure-insurance peer dataset.'}")
    else:
        st.warning("No peer data is currently available.")

with tab4:
    st.subheader("Efficient Frontier")
    st.caption("The frontier uses historical returns and covariance across listed pure-insurance peers. It illustrates the historical risk-return trade-off, not a guaranteed optimal future portfolio.")
    pure_insurance = {k:v for k,v in INSURANCE_STOCKS.items() if COMPANY_CATEGORIES.get(k) == "Pure Insurance"}
    returns_df = load_peer_returns(tuple(pure_insurance.items()), "2y")
    if returns_df.shape[1] >= 3:
        ann_mu=returns_df.mean()*252; cov=returns_df.cov()*252; names=list(returns_df.columns); n=len(names)
        def port_stats(w):
            return float(np.dot(w,ann_mu.values)), float(np.sqrt(np.dot(w,np.dot(cov.values,w))))
        def objective(w):
            ret,vol=port_stats(w); return -((ret-0.06)/vol) if vol>0 else 999
        opt=minimize(objective,np.ones(n)/n,bounds=tuple((0,0.5) for _ in range(n)),constraints=({"type":"eq","fun":lambda w:np.sum(w)-1},),method="SLSQP")
        rng=np.random.default_rng(42); W=rng.dirichlet(np.ones(n),size=5000); rets=W@ann_mu.values; vols=np.sqrt(np.einsum("ij,jk,ik->i",W,cov.values,W))
        st.markdown('<div class="graph-title">Historical Risk–Return Opportunity Set</div><div class="graph-subtitle">Each dot is a simulated portfolio of pure-insurance companies.</div>', unsafe_allow_html=True)
        fig=go.Figure(go.Scatter(x=vols,y=rets,mode="markers",marker=dict(size=4,opacity=.35),name="Simulated portfolios"))
        if opt.success:
            oret,ovol=port_stats(opt.x); fig.add_trace(go.Scatter(x=[ovol],y=[oret],mode="markers",marker=dict(size=14,symbol="star"),name="Maximum Sharpe portfolio"))
        if selected_company in names:
            idx=names.index(selected_company); fig.add_trace(go.Scatter(x=[math.sqrt(cov.iloc[idx,idx])],y=[ann_mu.iloc[idx]],mode="markers+text",text=[selected_company],textposition="top center",marker=dict(size=14),name="Selected company"))
        fig.update_layout(template="plotly_dark",height=500,title="Efficient Frontier: Historical Return vs Volatility",xaxis_title="Annualized Volatility",yaxis_title="Annualized Return",hovermode="closest")
        st.plotly_chart(fig,use_container_width=True,theme=None)
        if opt.success:
            graph_explanation(f"The simulated opportunity set shows the historical relationship between risk and return. The maximum-Sharpe portfolio estimate is <b>{oret:.2%}</b> annualized return at <b>{ovol:.2%}</b> volatility. {selected_company} is plotted separately so you can compare its standalone historical position with diversified peer portfolios.")
    else:
        st.warning("Not enough peer price histories are available to construct the frontier.")

with tab5:
    st.subheader("1-Year Monte Carlo Scenario Simulation")
    st.caption("This is a stochastic simulation using historical mean return and volatility. It is not a forecast or guaranteed return.")
    simulations=st.slider("Number of simulations",1000,10000,5000,1000); horizon=252
    hist_returns=daily_returns(stock_close).dropna(); mu=float(hist_returns.mean()); sigma=float(hist_returns.std()); rng=np.random.default_rng(42)
    shocks=rng.normal(mu,sigma,size=(horizon,simulations)); paths=latest_price*np.exp(np.cumsum(shocks,axis=0)); final_values=paths[-1]; q=np.percentile(final_values,[5,25,50,75,95])
    st.markdown('<div class="graph-title">Simulated 1-Year Price Paths</div><div class="graph-subtitle">A sample of simulated paths illustrates the range of possible historical-parameter outcomes.</div>', unsafe_allow_html=True)
    fig=go.Figure()
    for idx in rng.choice(simulations,size=min(60,simulations),replace=False):
        fig.add_trace(go.Scatter(y=paths[:,idx],mode="lines",line=dict(width=1),showlegend=False,opacity=.15))
    fig.add_hline(y=latest_price,line_dash="dash",annotation_text="Current price")
    fig.update_layout(template="plotly_dark",height=500,title=f"{selected_company}: Monte Carlo Simulation",xaxis_title="Trading Days",yaxis_title="Simulated Price (₹)")
    st.plotly_chart(fig,use_container_width=True,theme=None)
    median=q[2]; probability_above=float(np.mean(final_values>latest_price));
    graph_explanation(f"The simulation produces a median 1-year endpoint of <b>₹{median:,.2f}</b>. About <b>{probability_above:.1%}</b> of simulated endpoints finish above the current price. The spread is intentionally shown to communicate uncertainty rather than predict one exact price.")
    st.dataframe(pd.DataFrame({"Scenario":["5th percentile","25th percentile","Median","75th percentile","95th percentile"],"Simulated price (₹)":q}),use_container_width=True,hide_index=True)

with tab6:
    st.subheader("Black-Scholes Option Valuation")
    st.caption("The model estimates theoretical European option values from price, volatility, time to expiry, risk-free rate and dividend yield. It is a pricing model, not an investment recommendation.")
    c1,c2,c3,c4=st.columns(4)
    with c1: strike=st.number_input("Strike price (₹)",min_value=1.0,value=float(round(latest_price,0)),step=1.0)
    with c2: expiry=st.number_input("Time to expiry (years)",min_value=.01,value=1.0,step=.25)
    with c3: rf=st.number_input("Risk-free rate",min_value=0.0,max_value=1.0,value=.06,step=.01,format="%.2f")
    with c4: dividend=st.number_input("Dividend yield",min_value=0.0,max_value=1.0,value=0.0,step=.01,format="%.2f")
    vol=float(metrics["Annual Volatility"]); S=latest_price; K=float(strike); T=float(expiry)
    if vol>0 and S>0 and K>0:
        d1=(math.log(S/K)+(rf-dividend+0.5*vol**2)*T)/(vol*math.sqrt(T)); d2=d1-vol*math.sqrt(T)
        call=S*math.exp(-dividend*T)*norm.cdf(d1)-K*math.exp(-rf*T)*norm.cdf(d2)
        put=K*math.exp(-rf*T)*norm.cdf(-d2)-S*math.exp(-dividend*T)*norm.cdf(-d1)
        o1,o2,o3=st.columns(3); o1.metric("Call value",f"₹{call:,.2f}"); o2.metric("Put value",f"₹{put:,.2f}"); o3.metric("Volatility used",f"{vol:.2%}")
        st.dataframe(pd.DataFrame({"Input":["Current price","Strike","Volatility","Risk-free rate","Dividend yield","Time to expiry","d1","d2"],"Value":[f"₹{S:,.2f}",f"₹{K:,.2f}",f"{vol:.2%}",f"{rf:.2%}",f"{dividend:.2%}",f"{T:.2f} years",f"{d1:.4f}",f"{d2:.4f}"]}),use_container_width=True,hide_index=True)
        st.markdown(f'<div class="explain"><b>Interpretation:</b> using {selected_company}’s historical annual volatility of <b>{vol:.2%}</b>, the model gives a theoretical call value of <b>₹{call:,.2f}</b> and put value of <b>₹{put:,.2f}</b> for the inputs above. These are model outputs and depend heavily on the assumptions entered.</div>',unsafe_allow_html=True)
    else:
        st.warning("Black-Scholes requires positive price, strike, time and volatility.")

st.markdown("---")
st.caption(f"InsureInvest • {selected_company} • Data through {latest_date.strftime('%d %B %Y')} • Historical analytics prototype; not financial advice.")
