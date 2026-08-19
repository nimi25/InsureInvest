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
.stApp{background:#f6f8fb}.block-container{max-width:1450px;padding-top:1.5rem;padding-bottom:3rem}
.hero{background:linear-gradient(135deg,#111827,#1e3a5f);padding:2rem 2.4rem;border-radius:18px;margin-bottom:1.3rem}.hero h1{color:white;font-size:2.35rem;margin:0}.hero p{color:#cbd5e1;margin:.4rem 0 0}
.card{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:1rem 1.2rem;box-shadow:0 4px 15px rgba(15,23,42,.05)}
.small{color:#64748b;font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;font-weight:650}.big{color:#111827;font-size:1.7rem;font-weight:750;margin-top:.3rem}
.explain{background:white;border:1px solid #e2e8f0;border-left:5px solid #2563eb;border-radius:14px;padding:1.15rem 1.4rem;margin:.7rem 0 1.4rem;color:#334155;line-height:1.6}
.section-note{color:#64748b;margin-top:-.45rem;margin-bottom:1rem}
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
    data = {}
    for company, ticker in companies:
        try:
            df = get_price_data(ticker, period)
            data[company] = daily_returns(df.set_index("Date")["Close"])
        except Exception:
            pass
    return pd.concat(data, axis=1).dropna(how="all").ffill().dropna() if data else pd.DataFrame()

company_options = list(INSURANCE_STOCKS.keys())
default_company = st.session_state.get("selected_company", company_options[0])
if default_company not in company_options:
    default_company = company_options[0]

st.sidebar.markdown("## InsureInvest")
st.sidebar.caption("Company Analysis")
st.sidebar.markdown("---")
selected_company = st.sidebar.selectbox("Select a company", company_options, index=company_options.index(default_company))
st.session_state["selected_company"] = selected_company
selected_ticker = INSURANCE_STOCKS[selected_company]
category = COMPANY_CATEGORIES.get(selected_company, "Investment")
st.sidebar.markdown(f"**NSE:** `{selected_ticker}`")
st.sidebar.caption(category)
st.sidebar.markdown("---")
if st.sidebar.button("← Back to Investment Planner", use_container_width=True):
    st.switch_page("app.py")

try:
    with st.spinner(f"Loading {selected_company} market data..."):
        stock_df = load_company(selected_ticker, "2y")
        benchmark_df = load_benchmark("2y")
except Exception as exc:
    st.error("Market data could not be loaded right now.")
    st.info("Yahoo Finance can temporarily rate-limit requests. Wait a few seconds and retry.")
    st.caption(f"Technical details: {exc}")
    if st.button("Retry"):
        st.cache_data.clear(); st.rerun()
    st.stop()

stock_close = stock_df.set_index("Date")["Close"]
benchmark_close = benchmark_df.set_index("Date")["Close"]
metrics = performance_summary(stock_close, benchmark_close)
latest = stock_df.iloc[-1]
latest_price = float(latest["Close"])
latest_date = pd.to_datetime(latest["Date"])

st.markdown(f'<div class="hero"><h1>{selected_company}</h1><p>{category} • NSE: {selected_ticker} • Historical company analytics</p></div>', unsafe_allow_html=True)

st.subheader("Performance Snapshot")
cols = st.columns(5)
items = [("Annual Return",f"{metrics['Annual Return']:.2%}","Historical annualized return"),("Annual Volatility",f"{metrics['Annual Volatility']:.2%}","Historical price fluctuation"),("Sharpe Ratio",f"{metrics['Sharpe Ratio']:.2f}","Return earned per unit of risk"),("Maximum Drawdown",f"{metrics['Maximum Drawdown']:.2%}","Largest historical peak-to-trough fall"),("Beta",f"{metrics['Beta']:.2f}","Sensitivity versus NIFTY 50")]
for c,(label,value,desc) in zip(cols,items):
    with c: st.markdown(f'<div class="card"><div class="small">{label}</div><div class="big">{value}</div><div style="color:#94a3b8;font-size:.75rem">{desc}</div></div>',unsafe_allow_html=True)
st.markdown(f'<div class="explain"><b>Quick read:</b> {selected_company} is at <b>₹{latest_price:,.2f}</b>. Its historical annual return is <b>{metrics["Annual Return"]:.2%}</b>, volatility is <b>{metrics["Annual Volatility"]:.2%}</b>, and Sharpe ratio is <b>{metrics["Sharpe Ratio"]:.2f}</b>. These are historical measures, not guarantees of future performance.</div>',unsafe_allow_html=True)

# 1. TECHNICALS
st.header("1. Price & Technical Analysis")
st.markdown('<div class="section-note">Price trend, moving averages, volatility bands and momentum indicators.</div>',unsafe_allow_html=True)
fig=go.Figure()
fig.add_trace(go.Candlestick(x=stock_df.Date,open=stock_df.Open,high=stock_df.High,low=stock_df.Low,close=stock_df.Close,name=selected_company))
for col,name,dash in [("SMA_50","50-Day SMA","solid"),("SMA_200","200-Day SMA","solid"),("BB_Upper","Bollinger Upper","dash"),("BB_Lower","Bollinger Lower","dash")]:
    fig.add_trace(go.Scatter(x=stock_df.Date,y=stock_df[col],mode="lines",name=name,line=dict(width=1.7,dash=dash)))
fig.update_layout(template="plotly_dark",height=560,title="Historical Price with Moving Averages and Bollinger Bands",xaxis_rangeslider_visible=False,hovermode="x unified")
st.plotly_chart(fig,use_container_width=True,theme=None)
st.markdown(f'<div class="explain"><b>What this shows:</b> The candlesticks show daily price movement. The 50-day and 200-day SMAs show short- and long-term trend direction, while Bollinger Bands show a volatility range. <b>For {selected_company}:</b> the current price is ₹{latest_price:,.2f}; it is {"above" if latest_price>float(latest["SMA_50"]) else "below"} its 50-day average and {"above" if latest_price>float(latest["SMA_200"]) else "below"} its 200-day average.</div>',unsafe_allow_html=True)

rsi=float(latest["RSI_14"])
rsi_state="overbought" if rsi>=70 else "oversold" if rsi<=30 else "neutral"
rsi_fig=go.Figure(go.Scatter(x=stock_df.Date,y=stock_df.RSI_14,mode="lines",name="RSI-14"))
rsi_fig.add_hline(y=70,line_dash="dash",annotation_text="70 — overbought"); rsi_fig.add_hline(y=30,line_dash="dash",annotation_text="30 — oversold")
rsi_fig.update_layout(template="plotly_dark",height=320,title="RSI-14 Momentum Indicator",yaxis=dict(range=[0,100]),hovermode="x unified")
st.plotly_chart(rsi_fig,use_container_width=True,theme=None)
st.markdown(f'<div class="explain"><b>What this shows:</b> RSI measures recent momentum from 0 to 100. Readings above 70 are commonly considered overbought and below 30 oversold. <b>For {selected_company}:</b> RSI is <b>{rsi:.1f}</b>, indicating <b>{rsi_state} momentum</b> right now.</div>',unsafe_allow_html=True)

macd=float(latest["MACD"]); signal=float(latest["MACD_Signal"]); macd_state="positive" if macd>signal else "negative" if macd<signal else "neutral"
macd_fig=go.Figure(); macd_fig.add_trace(go.Scatter(x=stock_df.Date,y=stock_df.MACD,mode="lines",name="MACD")); macd_fig.add_trace(go.Scatter(x=stock_df.Date,y=stock_df.MACD_Signal,mode="lines",name="Signal")); macd_fig.add_trace(go.Bar(x=stock_df.Date,y=stock_df.MACD_Histogram,name="Histogram",opacity=.5))
macd_fig.update_layout(template="plotly_dark",height=350,title="MACD Momentum and Signal Line",hovermode="x unified")
st.plotly_chart(macd_fig,use_container_width=True,theme=None)
st.markdown(f'<div class="explain"><b>What this shows:</b> MACD compares short- and long-term price momentum. When MACD is above its signal line, momentum is generally stronger. <b>For {selected_company}:</b> MACD is <b>{macd:.2f}</b> versus a signal value of <b>{signal:.2f}</b>, giving a <b>{macd_state}</b> momentum reading.</div>',unsafe_allow_html=True)

close=float(latest["Close"]); upper=float(latest["BB_Upper"]); lower=float(latest["BB_Lower"])
bb_state="above the upper band" if close>upper else "below the lower band" if close<lower else "inside the bands"
st.markdown(f'<div class="explain"><b>Technical summary:</b> Trend is <b>{"positive" if close>float(latest["SMA_50"]) and close>float(latest["SMA_200") else "weak" if close<float(latest["SMA_50"]) and close<float(latest["SMA_200"]) else "mixed"}</b>, RSI momentum is <b>{rsi_state}</b>, MACD is <b>{macd_state}</b>, and price is <b>{bb_state}</b>. This gives a quick technical view before moving to risk and portfolio analysis.</div>',unsafe_allow_html=True)

# 2. RISK AND RETURN
st.header("2. Risk, Return & Benchmark")
st.markdown('<div class="section-note">How the selected company has historically performed relative to the NIFTY 50 benchmark.</div>',unsafe_allow_html=True)
benchmark_metrics=performance_summary(benchmark_close,benchmark_close)
risk_table=pd.DataFrame({"Metric":["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown","Beta"],selected_company:[metrics["Annual Return"],metrics["Annual Volatility"],metrics["Sharpe Ratio"],metrics["Maximum Drawdown"],metrics["Beta"]],"NIFTY 50 Benchmark":[benchmark_metrics["Annual Return"],benchmark_metrics["Annual Volatility"],benchmark_metrics["Sharpe Ratio"],benchmark_metrics["Maximum Drawdown"],1.0]})
st.dataframe(risk_table.style.format({selected_company:"{:.2%}","NIFTY 50 Benchmark":"{:.2%}"},subset=["Annual Return","Annual Volatility","Maximum Drawdown"]),use_container_width=True,hide_index=True)
st.markdown(f'<div class="explain"><b>How to read it:</b> Sharpe asks how much historical return was achieved per unit of volatility. Beta measures sensitivity to the NIFTY 50. Maximum drawdown measures the worst peak-to-trough decline. <b>{selected_company}</b> has a Sharpe of <b>{metrics["Sharpe Ratio"]:.2f}</b> and Beta of <b>{metrics["Beta"]:.2f}</b>.</div>',unsafe_allow_html=True)

stock_norm=stock_close/stock_close.iloc[0]*100; bench_norm=benchmark_close/benchmark_close.iloc[0]*100
fig=go.Figure(); fig.add_trace(go.Scatter(x=stock_norm.index,y=stock_norm,mode="lines",name=selected_company)); fig.add_trace(go.Scatter(x=bench_norm.index,y=bench_norm,mode="lines",name="NIFTY 50"))
fig.update_layout(template="plotly_dark",height=420,title="Growth of ₹100: Company vs NIFTY 50",yaxis_title="Indexed Value",hovermode="x unified")
st.plotly_chart(fig,use_container_width=True,theme=None)
relative="outperformed" if stock_norm.iloc[-1]>bench_norm.iloc[-1] else "underperformed"
st.markdown(f'<div class="explain"><b>What this shows:</b> Both series start at 100, so their relative movement is easy to compare. <b>{selected_company}</b> has historically <b>{relative}</b> the NIFTY 50 over the displayed period based on the indexed endpoints.</div>',unsafe_allow_html=True)

# 3. PEERS
st.header("3. Insurance Peer Comparison")
st.markdown('<div class="section-note">Compare the selected company with the separately listed pure-insurance peers available in the dataset.</div>',unsafe_allow_html=True)
pure_insurance={k:v for k,v in INSURANCE_STOCKS.items() if COMPANY_CATEGORIES.get(k)=="Pure Insurance"}
peer_rows=[]
with st.spinner("Loading insurance peers..."):
    for company,ticker in pure_insurance.items():
        try:
            df=load_company(ticker,"2y"); c=df.set_index("Date")["Close"]; m=performance_summary(c,benchmark_close)
            peer_rows.append({"Company":company,"Annual Return":m["Annual Return"],"Volatility":m["Annual Volatility"],"Sharpe":m["Sharpe Ratio"],"Max Drawdown":m["Maximum Drawdown"],"Beta":m["Beta"]})
        except Exception: pass
if peer_rows:
    peers=pd.DataFrame(peer_rows).sort_values("Sharpe",ascending=False)
    st.dataframe(peers.style.format({"Annual Return":"{:.2%}","Volatility":"{:.2%}","Sharpe":"{:.2f}","Max Drawdown":"{:.2%}","Beta":"{:.2f}"}),use_container_width=True,hide_index=True)
    fig=go.Figure(go.Bar(x=peers.Company,y=peers.Sharpe,name="Sharpe Ratio")); fig.update_layout(template="plotly_dark",height=380,title="Sharpe Ratio Ranking Across Pure-Insurance Peers",xaxis_title="Company",yaxis_title="Sharpe Ratio")
    st.plotly_chart(fig,use_container_width=True,theme=None)
    rank=int(peers.reset_index(drop=True).index[peers.reset_index(drop=True).Company==selected_company][0]+1) if selected_company in peers.Company.values else None
    rank_text=f"{selected_company} ranks #{rank} of {len(peers)} by Sharpe ratio" if rank else f"{selected_company} is not classified as a pure-insurance peer"
    st.markdown(f'<div class="explain"><b>What this shows:</b> A higher Sharpe indicates stronger historical risk-adjusted performance. <b>Current result:</b> {rank_text} in the available pure-insurance comparison.</div>',unsafe_allow_html=True)
else: st.warning("No peer data is currently available.")

# 4. EFFICIENT FRONTIER
st.header("4. Efficient Frontier")
st.markdown('<div class="section-note">A portfolio-level view of the historical return/risk trade-off among pure-insurance companies.</div>',unsafe_allow_html=True)
returns_df=load_peer_returns(tuple(pure_insurance.items()),"2y")
if returns_df.shape[1]>=3:
    ann_mu=returns_df.mean()*252; cov=returns_df.cov()*252; names=list(returns_df.columns); n=len(names)
    def port_stats(w):
        ret=float(np.dot(w,ann_mu.values)); vol=float(np.sqrt(np.dot(w,np.dot(cov.values,w)))); return ret,vol
    def objective(w):
        ret,vol=port_stats(w); return -((ret-.06)/vol) if vol>0 else 999
    opt=minimize(objective,np.ones(n)/n,bounds=tuple((0,.5) for _ in range(n)),constraints=({"type":"eq","fun":lambda w:np.sum(w)-1},),method="SLSQP")
    rng=np.random.default_rng(42); W=rng.dirichlet(np.ones(n),size=5000); rets=W@ann_mu.values; vols=np.sqrt(np.einsum("ij,jk,ik->i",W,cov.values,W))
    fig=go.Figure(go.Scatter(x=vols,y=rets,mode="markers",marker=dict(size=4,opacity=.3),name="Simulated portfolios"))
    if opt.success:
        oret,ovol=port_stats(opt.x); fig.add_trace(go.Scatter(x=[ovol],y=[oret],mode="markers",marker=dict(size=15,symbol="star"),name="Maximum-Sharpe portfolio"))
    if selected_company in names:
        i=names.index(selected_company); fig.add_trace(go.Scatter(x=[math.sqrt(cov.iloc[i,i])],y=[ann_mu.iloc[i]],mode="markers+text",text=[selected_company],textposition="top center",marker=dict(size=14),name="Selected company"))
    fig.update_layout(template="plotly_dark",height=520,title="Efficient Frontier: Historical Risk vs Return",xaxis_title="Annualized Volatility",yaxis_title="Annualized Return",hovermode="closest")
    st.plotly_chart(fig,use_container_width=True,theme=None)
    if opt.success:
        st.markdown(f'<div class="explain"><b>What this shows:</b> Each dot is a simulated portfolio. Moving right means more historical volatility; moving up means more historical return. The star is the portfolio with the highest estimated Sharpe ratio under the model constraints. <b>Current result:</b> the maximum-Sharpe portfolio has estimated annual return of <b>{oret:.2%}</b> and volatility of <b>{ovol:.2%}</b>. {selected_company} is {"inside the peer opportunity set shown" if selected_company in names else "not itself a pure-insurance peer, so it is not plotted as an individual frontier point"}.</div>',unsafe_allow_html=True)
else: st.warning("Not enough peer histories are available to construct the Efficient Frontier.")

# 5. MONTE CARLO
st.header("5. Monte Carlo Scenario Simulation")
st.markdown('<div class="section-note">A stochastic 1-year simulation using the selected company's historical daily return and volatility.</div>',unsafe_allow_html=True)
simulations=st.slider("Number of simulations",1000,10000,5000,1000,key="mc_simulations")
horizon=252; hist_returns=daily_returns(stock_close).dropna(); mu=float(hist_returns.mean()); sigma=float(hist_returns.std()); rng=np.random.default_rng(42)
shocks=rng.normal(mu,sigma,size=(horizon,simulations)); paths=latest_price*np.exp(np.cumsum(shocks,axis=0)); final_values=paths[-1]; q=np.percentile(final_values,[5,25,50,75,95])
fig=go.Figure(); chosen=rng.choice(simulations,size=min(60,simulations),replace=False)
for idx in chosen: fig.add_trace(go.Scatter(y=paths[:,idx],mode="lines",showlegend=False,opacity=.15))
fig.add_hline(y=latest_price,line_dash="dash",annotation_text="Current price")
fig.update_layout(template="plotly_dark",height=520,title=f"Monte Carlo: {simulations:,} Simulated 1-Year Price Paths",xaxis_title="Trading Days",yaxis_title="Simulated Price (₹)")
st.plotly_chart(fig,use_container_width=True,theme=None)
st.dataframe(pd.DataFrame({"Scenario":["5th percentile","25th percentile","Median","75th percentile","95th percentile"],"Simulated price (₹)":q}),use_container_width=True,hide_index=True)
change=(q[2]/latest_price-1)
st.markdown(f'<div class="explain"><b>What this shows:</b> The simulation repeatedly generates possible 252-trading-day paths using historical mean return and volatility. <b>For {selected_company}:</b> the median simulated endpoint is <b>₹{q[2]:,.2f}</b>, which is {"above" if change>=0 else "below"} the current price by about <b>{abs(change):.1%}</b>. The percentile range illustrates uncertainty; it is not a prediction or guarantee.</div>',unsafe_allow_html=True)

# 6. BLACK-SCHOLES
st.header("6. Black-Scholes Option Valuation")
st.markdown('<div class="section-note">A theoretical European call/put valuation using the selected company's current price and historical volatility.</div>',unsafe_allow_html=True)
c1,c2,c3,c4=st.columns(4)
with c1: strike=st.number_input("Strike price (₹)",min_value=1.0,value=float(round(latest_price,0)),step=1.0,key="bs_strike")
with c2: expiry=st.number_input("Time to expiry (years)",min_value=.01,value=1.0,step=.25,key="bs_expiry")
with c3: rf=st.number_input("Risk-free rate",min_value=0.0,max_value=1.0,value=.06,step=.01,format="%.2f",key="bs_rf")
with c4: dividend=st.number_input("Dividend yield",min_value=0.0,max_value=1.0,value=0.0,step=.01,format="%.2f",key="bs_div")
vol=float(metrics["Annual Volatility"]); S=latest_price; K=float(strike); T=float(expiry)
if vol>0 and S>0 and K>0:
    d1=(math.log(S/K)+(rf-dividend+.5*vol**2)*T)/(vol*math.sqrt(T)); d2=d1-vol*math.sqrt(T)
    call=S*math.exp(-dividend*T)*norm.cdf(d1)-K*math.exp(-rf*T)*norm.cdf(d2); put=K*math.exp(-rf*T)*norm.cdf(-d2)-S*math.exp(-dividend*T)*norm.cdf(-d1)
    o1,o2,o3=st.columns(3); o1.metric("Theoretical Call Value",f"₹{call:,.2f}"); o2.metric("Theoretical Put Value",f"₹{put:,.2f}"); o3.metric("Historical Volatility Used",f"{vol:.2%}")
    st.dataframe(pd.DataFrame({"Input":["Current price","Strike","Volatility","Risk-free rate","Dividend yield","Time to expiry","d1","d2"],"Value":[f"₹{S:,.2f}",f"₹{K:,.2f}",f"{vol:.2%}",f"{rf:.2%}",f"{dividend:.2%}",f"{T:.2f} years",f"{d1:.4f}",f"{d2:.4f}"]}),use_container_width=True,hide_index=True)
    st.markdown(f'<div class="explain"><b>What this shows:</b> Black-Scholes estimates theoretical option values from price, strike, time, interest rate, dividends and volatility. <b>For {selected_company}:</b> the model gives a theoretical call value of <b>₹{call:,.2f}</b> and put value of <b>₹{put:,.2f}</b> under the inputs above. This is an option-pricing model, not a recommendation to buy an option.</div>',unsafe_allow_html=True)
else: st.warning("Black-Scholes requires positive price, strike, time and volatility.")

st.markdown("---")
st.caption(f"InsureInvest • {selected_company} • Data through {latest_date.strftime('%d %B %Y')} • Historical analytics prototype; not financial advice.")
