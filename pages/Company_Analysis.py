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

@st.cache_data(ttl=900, show_spinner=False)
def load_company(ticker, period="2y"):
    return add_technical_indicators(get_price_data(ticker, period))

@st.cache_data(ttl=900, show_spinner=False)
def load_benchmark(period="2y"):
    return get_price_data(BENCHMARK, period)

@st.cache_data(ttl=900, show_spinner=False)
def load_peer_returns(items, period="2y"):
    data = {}
    for name, ticker in items:
        try:
            df = get_price_data(ticker, period)
            data[name] = daily_returns(df.set_index("Date")["Close"])
        except Exception:
            continue
    return pd.concat(data, axis=1).dropna().ffill() if data else pd.DataFrame()

st.markdown("""
<style>
.stApp{background:#f6f8fb}.block-container{max-width:1450px;padding-top:1.5rem;padding-bottom:3rem}
.hero{background:linear-gradient(135deg,#111827,#1e3a5f);padding:2rem 2.4rem;border-radius:18px;margin-bottom:1.3rem}.hero h1{color:white;margin:0;font-size:2.35rem}.hero p{color:#cbd5e1;margin:.4rem 0 0}
.explain{background:white;border:1px solid #e2e8f0;border-left:5px solid #2563eb;border-radius:14px;padding:1.1rem 1.4rem;margin:.7rem 0 1.5rem;color:#334155;line-height:1.6}
.note{color:#64748b;margin-top:-.5rem;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

companies = list(INSURANCE_STOCKS.keys())
default = st.session_state.get("selected_company", companies[0])
if default not in companies: default = companies[0]

st.sidebar.markdown("## InsureInvest")
st.sidebar.caption("Company Analysis")
st.sidebar.markdown("---")
selected = st.sidebar.selectbox("Select a company", companies, index=companies.index(default))
st.session_state["selected_company"] = selected
ticker = INSURANCE_STOCKS[selected]
category = COMPANY_CATEGORIES.get(selected, "Investment")
st.sidebar.markdown(f"**NSE:** `{ticker}`")
st.sidebar.caption(category)
st.sidebar.markdown("---")
if st.sidebar.button("← Back to Investment Planner", use_container_width=True):
    st.switch_page("home.py")

try:
    with st.spinner(f"Loading {selected} market data..."):
        stock = load_company(ticker, "2y")
        bench = load_benchmark("2y")
except Exception as exc:
    st.error("Market data could not be loaded right now.")
    st.info("Yahoo Finance may be temporarily rate-limiting requests. Retry in a few seconds.")
    st.caption(str(exc))
    if st.button("Retry"):
        st.cache_data.clear(); st.rerun()
    st.stop()

sc = stock.set_index("Date")["Close"]
bc = bench.set_index("Date")["Close"]
m = performance_summary(sc, bc)
latest = stock.iloc[-1]
price = float(latest["Close"])

st.markdown(f'<div class="hero"><h1>{selected}</h1><p>{category} • NSE: {ticker} • Full historical analytics</p></div>', unsafe_allow_html=True)

st.subheader("Performance Snapshot")
metric_cols = st.columns(5)
metrics = [("Annual Return",m["Annual Return"],"Historical annualized return",".2%"),("Annual Volatility",m["Annual Volatility"],"Historical price fluctuation",".2%"),("Sharpe Ratio",m["Sharpe Ratio"],"Return relative to risk",".2f"),("Maximum Drawdown",m["Maximum Drawdown"],"Largest peak-to-trough fall",".2%"),("Beta",m["Beta"],"Sensitivity to NIFTY 50",".2f")]
for col,(name,value,desc,fmt) in zip(metric_cols,metrics):
    with col:
        st.metric(name,format(value,fmt),help=desc)
st.markdown(f'<div class="explain"><b>Quick read:</b> {selected} is trading at <b>₹{price:,.2f}</b>. Historical annual return is <b>{m["Annual Return"]:.2%}</b>, volatility is <b>{m["Annual Volatility"]:.2%}</b>, and Sharpe is <b>{m["Sharpe Ratio"]:.2f}</b>. These describe historical behaviour, not guaranteed future results.</div>',unsafe_allow_html=True)

# ---------------- TECHNICALS ----------------
st.header("1. Price & Technical Analysis")
st.markdown('<div class="note">Trend, volatility bands and momentum indicators.</div>',unsafe_allow_html=True)
fig=go.Figure(go.Candlestick(x=stock.Date,open=stock.Open,high=stock.High,low=stock.Low,close=stock.Close,name=selected))
for c,n,d in [("SMA_50","50-Day SMA","solid"),("SMA_200","200-Day SMA","solid"),("BB_Upper","Bollinger Upper","dash"),("BB_Lower","Bollinger Lower","dash")]:
    fig.add_trace(go.Scatter(x=stock.Date,y=stock[c],mode="lines",name=n,line=dict(dash=d,width=1.6)))
fig.update_layout(template="plotly_dark",height=540,title="Historical Price, Moving Averages & Bollinger Bands",xaxis_rangeslider_visible=False,hovermode="x unified")
st.plotly_chart(fig,use_container_width=True,theme=None)
st.markdown(f'<div class="explain"><b>What this shows:</b> Candlesticks show daily price movement; the 50/200-day SMAs show short- and long-term trend; Bollinger Bands show a volatility range. <b>{selected}</b> is currently {"above" if price>float(latest["SMA_50"]) else "below"} its 50-day average and {"above" if price>float(latest["SMA_200"]) else "below"} its 200-day average.</div>',unsafe_allow_html=True)

rsi=float(latest["RSI_14"]); rsi_state="overbought" if rsi>=70 else "oversold" if rsi<=30 else "neutral"
fig=go.Figure(go.Scatter(x=stock.Date,y=stock.RSI_14,mode="lines",name="RSI-14")); fig.add_hline(y=70,line_dash="dash",annotation_text="70 — overbought"); fig.add_hline(y=30,line_dash="dash",annotation_text="30 — oversold"); fig.update_layout(template="plotly_dark",height=320,title="RSI-14 Momentum Indicator",yaxis=dict(range=[0,100]),hovermode="x unified")
st.plotly_chart(fig,use_container_width=True,theme=None)
st.markdown(f'<div class="explain"><b>What this shows:</b> RSI ranges from 0–100 and is commonly used to identify momentum extremes. <b>{selected}</b> has an RSI of <b>{rsi:.1f}</b>, which indicates <b>{rsi_state} momentum</b>.</div>',unsafe_allow_html=True)

macd=float(latest["MACD"]); sig=float(latest["MACD_Signal"]); macd_state="positive" if macd>sig else "negative" if macd<sig else "neutral"
fig=go.Figure(); fig.add_trace(go.Scatter(x=stock.Date,y=stock.MACD,mode="lines",name="MACD")); fig.add_trace(go.Scatter(x=stock.Date,y=stock.MACD_Signal,mode="lines",name="Signal")); fig.add_trace(go.Bar(x=stock.Date,y=stock.MACD_Histogram,name="Histogram",opacity=.5)); fig.update_layout(template="plotly_dark",height=350,title="MACD Momentum & Signal Line",hovermode="x unified")
st.plotly_chart(fig,use_container_width=True,theme=None)
st.markdown(f'<div class="explain"><b>What this shows:</b> MACD compares short- and long-term momentum. <b>{selected}</b> has MACD <b>{macd:.2f}</b> versus signal <b>{sig:.2f}</b>, giving a <b>{macd_state}</b> momentum reading.</div>',unsafe_allow_html=True)

# ---------------- RISK / RETURN ----------------
st.header("2. Risk, Return & Benchmark")
st.markdown('<div class="note">Historical performance and risk relative to the NIFTY 50.</div>',unsafe_allow_html=True)
bm=performance_summary(bc,bc)
risk=pd.DataFrame({"Metric":["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown","Beta"],selected:[m["Annual Return"],m["Annual Volatility"],m["Sharpe Ratio"],m["Maximum Drawdown"],m["Beta"]],"NIFTY 50 Benchmark":[bm["Annual Return"],bm["Annual Volatility"],bm["Sharpe Ratio"],bm["Maximum Drawdown"],1.0]})
st.dataframe(risk.style.format({selected:"{:.2%}","NIFTY 50 Benchmark":"{:.2%}"},subset=["Annual Return","Annual Volatility","Maximum Drawdown"]),use_container_width=True,hide_index=True)
st.markdown(f'<div class="explain"><b>What this shows:</b> Sharpe measures historical return per unit of volatility; Beta measures sensitivity to the NIFTY 50; maximum drawdown measures the worst historical decline. <b>{selected}</b> has Sharpe <b>{m["Sharpe Ratio"]:.2f}</b> and Beta <b>{m["Beta"]:.2f}</b>.</div>',unsafe_allow_html=True)

sn=sc/sc.iloc[0]*100; bn=bc/bc.iloc[0]*100
fig=go.Figure(); fig.add_trace(go.Scatter(x=sn.index,y=sn,mode="lines",name=selected)); fig.add_trace(go.Scatter(x=bn.index,y=bn,mode="lines",name="NIFTY 50")); fig.update_layout(template="plotly_dark",height=420,title="Growth of ₹100: Company vs NIFTY 50",xaxis_title="Date",yaxis_title="Indexed Value",hovermode="x unified")
st.plotly_chart(fig,use_container_width=True,theme=None)
st.markdown(f'<div class="explain"><b>What this shows:</b> Both series start at 100, making relative performance easy to compare. Over the displayed period, <b>{selected}</b> has <b>{"outperformed" if sn.iloc[-1]>bn.iloc[-1] else "underperformed"}</b> the NIFTY 50 based on the indexed endpoints.</div>',unsafe_allow_html=True)

# ---------------- PEERS ----------------
st.header("3. Insurance Peer Comparison")
st.markdown('<div class="note">Comparison with separately listed pure-insurance peers.</div>',unsafe_allow_html=True)
pure={k:v for k,v in INSURANCE_STOCKS.items() if COMPANY_CATEGORIES.get(k)=="Pure Insurance"}
rows=[]
with st.spinner("Loading pure-insurance peers..."):
    for name,t in pure.items():
        try:
            d=load_company(t,"2y"); c=d.set_index("Date")["Close"]; x=performance_summary(c,bc); rows.append({"Company":name,"Annual Return":x["Annual Return"],"Volatility":x["Annual Volatility"],"Sharpe":x["Sharpe Ratio"],"Max Drawdown":x["Maximum Drawdown"],"Beta":x["Beta"]})
        except Exception: pass
if rows:
    peers=pd.DataFrame(rows).sort_values("Sharpe",ascending=False)
    st.dataframe(peers.style.format({"Annual Return":"{:.2%}","Volatility":"{:.2%}","Sharpe":"{:.2f}","Max Drawdown":"{:.2%}","Beta":"{:.2f}"}),use_container_width=True,hide_index=True)
    fig=go.Figure(go.Bar(x=peers.Company,y=peers.Sharpe,name="Sharpe")); fig.update_layout(template="plotly_dark",height=380,title="Sharpe Ratio Ranking Across Pure-Insurance Peers",xaxis_title="Company",yaxis_title="Sharpe Ratio")
    st.plotly_chart(fig,use_container_width=True,theme=None)
    st.markdown(f'<div class="explain"><b>What this shows:</b> This ranks the available pure-insurance companies by historical Sharpe ratio. {selected} is {"included in" if selected in peers.Company.values else "not included in"} this pure-insurance peer set.</div>',unsafe_allow_html=True)
else: st.warning("No peer data is currently available.")

# ---------------- EFFICIENT FRONTIER ----------------
st.header("4. Efficient Frontier")
st.markdown('<div class="note">Simulated portfolios show the historical trade-off between expected annual return and annualized volatility.</div>',unsafe_allow_html=True)
ret_df=load_peer_returns(tuple(pure.items()),"2y")
if ret_df.shape[1]>=3:
    mu=ret_df.mean()*252; cov=ret_df.cov()*252; names=list(ret_df.columns); n=len(names)
    def stats(w):
        return float(w@mu.values),float(np.sqrt(w@cov.values@w))
    def objective(w):
        r,v=stats(w); return -(r-.06)/v if v>0 else 999
    opt=minimize(objective,np.ones(n)/n,bounds=tuple((0,.5) for _ in range(n)),constraints={"type":"eq","fun":lambda w:w.sum()-1},method="SLSQP")
    rng=np.random.default_rng(42); W=rng.dirichlet(np.ones(n),5000); rs=W@mu.values; vs=np.sqrt(np.einsum("ij,jk,ik->i",W,cov.values,W))
    fig=go.Figure(go.Scatter(x=vs,y=rs,mode="markers",marker=dict(size=4,opacity=.3),name="Simulated portfolios"))
    if opt.success:
        or_,ov=stats(opt.x); fig.add_trace(go.Scatter(x=[ov],y=[or_],mode="markers",marker=dict(size=15,symbol="star"),name="Maximum-Sharpe portfolio"))
    if selected in names:
        i=names.index(selected); fig.add_trace(go.Scatter(x=[math.sqrt(cov.iloc[i,i])],y=[mu.iloc[i]],mode="markers+text",text=[selected],textposition="top center",marker=dict(size=14),name="Selected company"))
    fig.update_layout(template="plotly_dark",height=520,title="Efficient Frontier: Historical Risk vs Return",xaxis_title="Annualized Volatility",yaxis_title="Annualized Return",hovermode="closest")
    st.plotly_chart(fig,use_container_width=True,theme=None)
    if opt.success: st.markdown(f'<div class="explain"><b>What this shows:</b> Each dot is a simulated portfolio. Higher means greater historical return; further right means greater historical volatility. The star is the maximum-Sharpe portfolio under the model constraints. <b>Model result:</b> estimated return <b>{or_:.2%}</b> and volatility <b>{ov:.2%}</b>.</div>',unsafe_allow_html=True)
else: st.warning("Not enough pure-insurance histories are available to build the frontier.")

# ---------------- MONTE CARLO ----------------
st.header("5. Monte Carlo Scenario Simulation")
st.markdown('<div class="note">A 1-year stochastic simulation based on the selected company's historical daily return and volatility.</div>',unsafe_allow_html=True)
sims=st.slider("Number of simulations",1000,10000,5000,1000,key="mc_sims"); hist=daily_returns(sc).dropna(); mu_d=float(hist.mean()); sig_d=float(hist.std()); rng=np.random.default_rng(42)
paths=price*np.exp(np.cumsum(rng.normal(mu_d,sig_d,size=(252,sims)),axis=0)); finals=paths[-1]; qs=np.percentile(finals,[5,25,50,75,95])
fig=go.Figure(); chosen_paths=rng.choice(sims,size=min(60,sims),replace=False)
for i in chosen_paths: fig.add_trace(go.Scatter(y=paths[:,i],mode="lines",showlegend=False,opacity=.15))
fig.add_hline(y=price,line_dash="dash",annotation_text="Current price"); fig.update_layout(template="plotly_dark",height=520,title=f"Monte Carlo: {sims:,} Simulated 1-Year Price Paths",xaxis_title="Trading Days",yaxis_title="Simulated Price (₹)")
st.plotly_chart(fig,use_container_width=True,theme=None)
st.dataframe(pd.DataFrame({"Scenario":["5th percentile","25th percentile","Median","75th percentile","95th percentile"],"Simulated price (₹)":qs}),use_container_width=True,hide_index=True)
st.markdown(f'<div class="explain"><b>What this shows:</b> Thousands of possible price paths are generated from historical mean return and volatility. <b>{selected}</b> has a median simulated endpoint of <b>₹{qs[2]:,.2f}</b>. The range shows uncertainty; it is not a forecast or guarantee.</div>',unsafe_allow_html=True)

# ---------------- BLACK-SCHOLES ----------------
st.header("6. Black-Scholes Option Valuation")
st.markdown('<div class="note">Theoretical European call and put valuation using the selected company's current price and historical volatility.</div>',unsafe_allow_html=True)
a,b,c,d=st.columns(4)
with a: K=st.number_input("Strike price (₹)",1.0,float(round(price,0)),1.0,key="bs_k")
with b: T=st.number_input("Time to expiry (years)",.01,20.0,1.0,.25,key="bs_t")
with c: rf=st.number_input("Risk-free rate",0.0,1.0,.06,.01,format="%.2f",key="bs_rf")
with d: div=st.number_input("Dividend yield",0.0,1.0,0.0,.01,format="%.2f",key="bs_div")
vol=float(m["Annual Volatility"]); S=price
if vol>0:
    d1=(math.log(S/K)+(rf-div+.5*vol**2)*T)/(vol*math.sqrt(T)); d2=d1-vol*math.sqrt(T)
    call=S*math.exp(-div*T)*norm.cdf(d1)-K*math.exp(-rf*T)*norm.cdf(d2); put=K*math.exp(-rf*T)*norm.cdf(-d2)-S*math.exp(-div*T)*norm.cdf(-d1)
    x,y,z=st.columns(3); x.metric("Theoretical Call Value",f"₹{call:,.2f}"); y.metric("Theoretical Put Value",f"₹{put:,.2f}"); z.metric("Historical Volatility Used",f"{vol:.2%}")
    st.dataframe(pd.DataFrame({"Input":["Current price","Strike","Volatility","Risk-free rate","Dividend yield","Time to expiry","d1","d2"],"Value":[f"₹{S:,.2f}",f"₹{K:,.2f}",f"{vol:.2%}",f"{rf:.2%}",f"{div:.2%}",f"{T:.2f} years",f"{d1:.4f}",f"{d2:.4f}"]}),use_container_width=True,hide_index=True)
    st.markdown(f'<div class="explain"><b>What this shows:</b> Black-Scholes estimates theoretical option values from price, strike, expiry, interest rate, dividends and volatility. For <b>{selected}</b>, the model gives a theoretical call of <b>₹{call:,.2f}</b> and put of <b>₹{put:,.2f}</b> using the inputs above. This is a pricing model, not an investment recommendation.</div>',unsafe_allow_html=True)
else: st.warning("Black-Scholes requires positive historical volatility.")

st.markdown("---")
st.caption(f"InsureInvest • {selected} • Data through {pd.to_datetime(latest['Date']).strftime('%d %B %Y')} • Historical analytics prototype; not financial advice.")
