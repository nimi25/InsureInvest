import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import performance_summary
from data import BENCHMARK, INSURANCE_STOCKS, COMPANY_CATEGORIES, get_price_data
from technicals import add_technical_indicators

# ---------------------------------------------------------------------------
# HTML rendering helper
# ---------------------------------------------------------------------------
def _collapse_html(markup):
    return re.sub(r"\n\s*", "", markup.strip())


def render_html(markup):
    st.markdown(_collapse_html(markup), unsafe_allow_html=True)


def render_sidebar_html(markup):
    st.sidebar.markdown(_collapse_html(markup), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chart theming
# ---------------------------------------------------------------------------
DARK_CHART = dict(template="plotly_dark", paper_bgcolor="#111827", plot_bgcolor="#111827", font=dict(family="Arial", color="#ffffff"))
DARK_GRID = "#374151"
LEGEND_TOP = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color="#ffffff"))
LEGEND_TOP_RIGHT = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#ffffff"))

st.set_page_config(page_title="InsureInvest — Company Analysis", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

render_html("""
<style>
.stApp { background-color:#f6f8fb; }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1400px; }
section[data-testid="stSidebar"] { background-color:#111827; }
section[data-testid="stSidebar"] * { color:#f9fafb !important; }
.hero { background:linear-gradient(135deg,#111827 0%,#1e3a5f 100%); padding:2rem 2.5rem; border-radius:18px; margin-bottom:1.5rem; }
.hero-title { color:#fff !important; font-size:2.4rem; font-weight:700; margin:0; }
.hero-subtitle { color:#cbd5e1 !important; font-size:1rem; margin-top:.5rem; }
.company-name { font-size:2rem; font-weight:700; color:#111827 !important; margin-bottom:.1rem; }
.company-meta { color:#64748b !important; font-size:.9rem; margin-bottom:1.5rem; }
.section-title { color:#111827 !important; font-size:1.35rem; font-weight:700; margin-top:1.8rem; margin-bottom:.7rem; }
.metric-card { background:#fff; border:1px solid #e5e7eb; border-radius:14px; padding:1.2rem 1.3rem; min-height:125px; box-shadow:0 4px 15px rgba(15,23,42,.05); }
.metric-label { color:#64748b !important; font-size:.82rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }
.metric-value { color:#111827 !important; font-size:1.8rem; font-weight:700; margin-top:.45rem; }
.metric-description { color:#94a3b8 !important; font-size:.75rem; margin-top:.3rem; }
.interpretation-card { background:#fff !important; border:1px solid #e2e8f0; border-left:5px solid #2563eb; border-radius:14px; padding:1.5rem 1.7rem; margin-top:.8rem; box-shadow:0 4px 15px rgba(15,23,42,.05); }
.interpretation-card * { color:#334155 !important; }
.interpretation-overall { color:#111827 !important; font-size:1.1rem; font-weight:700; margin-bottom:.4rem; }
.interpretation-overall-text { color:#475569 !important; font-size:.92rem; line-height:1.6; margin-bottom:1rem; }
.interpretation-grid { display:grid; grid-template-columns:1fr 1fr; gap:1.2rem 2rem; margin-top:1rem; }
.interpretation-item { background:#f8fafc !important; border:1px solid #e2e8f0; border-radius:10px; padding:1rem; }
.interpretation-label { color:#1e3a5f !important; font-weight:700; font-size:.9rem; margin-bottom:.35rem; }
.interpretation-text { color:#475569 !important; font-size:.84rem; line-height:1.55; }
.interpretation-note { color:#64748b !important; font-size:.75rem; margin-top:1.2rem; padding-top:.8rem; border-top:1px solid #e2e8f0; }
.footer { text-align:center; color:#94a3b8 !important; font-size:.75rem; margin-top:3rem; padding-top:1rem; border-top:1px solid #e2e8f0; }
@media (max-width:768px) { .interpretation-grid { grid-template-columns:1fr; } .hero-title { font-size:1.8rem; } }
</style>
""")

render_sidebar_html('<div style="font-size:1.7rem;font-weight:700;color:#fff !important;">InsureInvest</div><div style="color:#94a3b8 !important;font-size:.85rem;margin-top:5px;">Company Analysis</div>')
st.sidebar.markdown("---")
company_options = list(INSURANCE_STOCKS.keys())
_default_company = st.session_state.get("selected_company", company_options[0])
_default_index = company_options.index(_default_company) if _default_company in company_options else 0
selected_company = st.sidebar.selectbox("Select a company", company_options, index=_default_index)
st.session_state["selected_company"] = selected_company
selected_ticker = INSURANCE_STOCKS[selected_company]
render_sidebar_html(f'<div style="background:#1f2937;padding:10px;border-radius:8px;margin-top:10px;color:#cbd5e1 !important;font-size:.85rem;">NSE Ticker<br><strong style="color:#fff !important;">{selected_ticker}</strong><br>{COMPANY_CATEGORIES.get(selected_company, "Investment")}</div>')
st.sidebar.markdown("---")
if st.sidebar.button("← Back to Investment Planner", use_container_width=True):
    st.switch_page("app.py")

render_html('<div class="hero"><div class="hero-title">Company Analysis</div><div class="hero-subtitle">Detailed historical, technical and risk analysis</div></div>')

with st.spinner("Loading market data..."):
    try:
        stock_df = get_price_data(selected_ticker, "2y")
        stock_df = add_technical_indicators(stock_df)
        benchmark_df = get_price_data(BENCHMARK, "2y")
    except Exception as exc:
        st.error("Couldn\'t load market data from Yahoo Finance right now. This is usually temporary rate-limiting on their side — wait a minute and retry.")
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

render_html(f'<div class="company-name">{selected_company}</div><div class="company-meta">{COMPANY_CATEGORIES.get(selected_company, "Investment")} &nbsp; • &nbsp; NSE: {selected_ticker} &nbsp; • &nbsp; Latest price: ₹{latest_price:,.2f} &nbsp; • &nbsp; Data through {latest_date.strftime("%d %B %Y")}</div>')

# Performance snapshot
render_html('<div class="section-title">Performance Snapshot</div>')
col1,col2,col3,col4,col5=st.columns(5)
metric_cards=[(col1,"Annual Return",f"{metrics['Annual Return']:.2%}","Historical annualized return"),(col2,"Annual Volatility",f"{metrics['Annual Volatility']:.2%}","Historical price fluctuation"),(col3,"Sharpe Ratio",f"{metrics['Sharpe Ratio']:.2f}","Risk-adjusted return"),(col4,"Maximum Drawdown",f"{metrics['Maximum Drawdown']:.2%}","Worst peak-to-trough fall"),(col5,"Beta",f"{metrics['Beta']:.2f}","Sensitivity to NIFTY 50")]
for column,label,value,description in metric_cards:
    with column:
        render_html(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-description">{description}</div></div>')

# Price chart
st.markdown('<div class="section-title">Historical Price & Technical Indicators</div>',unsafe_allow_html=True)
st.caption("Daily price movement with 50-day and 200-day moving averages and Bollinger Bands.")
fig=go.Figure()
fig.add_trace(go.Candlestick(x=stock_df["Date"],open=stock_df["Open"],high=stock_df["High"],low=stock_df["Low"],close=stock_df["Close"],name=selected_company))
fig.add_trace(go.Scatter(x=stock_df["Date"],y=stock_df["SMA_50"],mode="lines",name="50-Day SMA",line=dict(width=2)))
fig.add_trace(go.Scatter(x=stock_df["Date"],y=stock_df["SMA_200"],mode="lines",name="200-Day SMA",line=dict(width=2)))
fig.add_trace(go.Scatter(x=stock_df["Date"],y=stock_df["BB_Upper"],mode="lines",name="Bollinger Upper",line=dict(width=1,dash="dash")))
fig.add_trace(go.Scatter(x=stock_df["Date"],y=stock_df["BB_Lower"],mode="lines",name="Bollinger Lower",line=dict(width=1,dash="dash")))
fig.update_layout(**DARK_CHART,height=560,margin=dict(l=20,r=20,t=20,b=20),xaxis=dict(title="Date",rangeslider=dict(visible=False),showgrid=False),yaxis=dict(title="Price (₹)",showgrid=True,gridcolor=DARK_GRID),hovermode="x unified",legend=LEGEND_TOP)
st.plotly_chart(fig,use_container_width=True,theme=None)

# RSI
st.markdown('<div class="section-title">Momentum Indicator — RSI</div>',unsafe_allow_html=True)
st.caption("14-day Relative Strength Index. Values above 70 may indicate overbought conditions, while values below 30 may indicate oversold conditions.")
rsi_fig=go.Figure(); rsi_fig.add_trace(go.Scatter(x=stock_df["Date"],y=stock_df["RSI_14"],mode="lines",name="RSI (14)",line=dict(width=2))); rsi_fig.add_hline(y=70,line_dash="dash",annotation_text="Overbought (70)"); rsi_fig.add_hline(y=30,line_dash="dash",annotation_text="Oversold (30)"); rsi_fig.add_hline(y=50,line_dash="dot",annotation_text="50")
rsi_fig.update_layout(**DARK_CHART,height=320,margin=dict(l=20,r=20,t=20,b=20),xaxis=dict(title="Date",showgrid=False),yaxis=dict(title="RSI",range=[0,100],showgrid=True,gridcolor=DARK_GRID),hovermode="x unified")
st.plotly_chart(rsi_fig,use_container_width=True,theme=None)

# MACD
st.markdown('<div class="section-title">Trend & Momentum Indicator — MACD</div>',unsafe_allow_html=True)
st.caption("Moving Average Convergence Divergence using 12-day, 26-day and 9-day exponential moving averages.")
macd_fig=go.Figure(); macd_fig.add_trace(go.Scatter(x=stock_df["Date"],y=stock_df["MACD"],mode="lines",name="MACD",line=dict(width=2))); macd_fig.add_trace(go.Scatter(x=stock_df["Date"],y=stock_df["MACD_Signal"],mode="lines",name="Signal Line",line=dict(width=2))); macd_fig.add_trace(go.Bar(x=stock_df["Date"],y=stock_df["MACD_Histogram"],name="Histogram",opacity=.6)); macd_fig.add_hline(y=0,line_dash="dash")
macd_fig.update_layout(**DARK_CHART,height=360,margin=dict(l=20,r=20,t=20,b=20),xaxis=dict(title="Date",showgrid=False),yaxis=dict(title="MACD",showgrid=True,gridcolor=DARK_GRID),hovermode="x unified",legend=LEGEND_TOP)
st.plotly_chart(macd_fig,use_container_width=True,theme=None)

# Technical summary
render_html('<div class="section-title">Technical Signal Summary</div><div style="color:#64748b;font-size:.9rem;margin-bottom:1rem;">Latest technical signals for the selected company.</div>')
latest_close=stock_df["Close"].iloc[-1]; latest_sma_50=stock_df["SMA_50"].iloc[-1]; latest_sma_200=stock_df["SMA_200"].iloc[-1]; latest_rsi=stock_df["RSI_14"].iloc[-1]; latest_macd=stock_df["MACD"].iloc[-1]; latest_macd_signal=stock_df["MACD_Signal"].iloc[-1]; latest_bb_upper=stock_df["BB_Upper"].iloc[-1]; latest_bb_lower=stock_df["BB_Lower"].iloc[-1]
if latest_close>latest_sma_50 and latest_close>latest_sma_200: trend_signal="Positive"; trend_text="The current price is above both the 50-day and 200-day moving averages, indicating a positive trend relative to these indicators."
elif latest_close<latest_sma_50 and latest_close<latest_sma_200: trend_signal="Weak"; trend_text="The current price is below both the 50-day and 200-day moving averages, indicating a relatively weak price trend."
else: trend_signal="Mixed"; trend_text="The current price is between the 50-day and 200-day moving averages, indicating mixed trend signals."
if latest_rsi>=70: rsi_signal="Overbought"; rsi_text=f"RSI is {latest_rsi:.1f}, above 70. This may indicate strong upward momentum or an overbought condition."
elif latest_rsi<=30: rsi_signal="Oversold"; rsi_text=f"RSI is {latest_rsi:.1f}, below 30. This may indicate strong downward momentum or an oversold condition."
else: rsi_signal="Neutral"; rsi_text=f"RSI is {latest_rsi:.1f}, within the 30–70 neutral range."
if latest_macd>latest_macd_signal: macd_signal="Positive"; macd_text="The MACD line is above the signal line, indicating positive momentum."
elif latest_macd<latest_macd_signal: macd_signal="Negative"; macd_text="The MACD line is below the signal line, indicating negative momentum."
else: macd_signal="Neutral"; macd_text="The MACD line is approximately equal to the signal line."
if latest_close>latest_bb_upper: bollinger_signal="Above Upper Band"; bollinger_text="The current price is above the upper Bollinger Band, indicating unusually strong recent price movement."
elif latest_close<latest_bb_lower: bollinger_signal="Below Lower Band"; bollinger_text="The current price is below the lower Bollinger Band, indicating unusually weak recent price movement."
else: bollinger_signal="Within Bands"; bollinger_text="The current price is currently within the Bollinger Bands."
render_html(f'<div class="interpretation-card"><div class="interpretation-overall">Trend: {trend_signal}</div><div class="interpretation-overall-text">{trend_text}</div><div class="interpretation-grid"><div class="interpretation-item"><div class="interpretation-label">Current Price</div><div class="interpretation-text">₹{latest_close:,.2f}</div></div><div class="interpretation-item"><div class="interpretation-label">Moving Averages</div><div class="interpretation-text">50-Day SMA: ₹{latest_sma_50:,.2f}<br>200-Day SMA: ₹{latest_sma_200:,.2f}</div></div><div class="interpretation-item"><div class="interpretation-label">RSI — {rsi_signal}</div><div class="interpretation-text">{rsi_text}</div></div><div class="interpretation-item"><div class="interpretation-label">MACD — {macd_signal}</div><div class="interpretation-text">{macd_text}<br><br>MACD: {latest_macd:.2f}<br>Signal: {latest_macd_signal:.2f}</div></div><div class="interpretation-item"><div class="interpretation-label">Bollinger Bands — {bollinger_signal}</div><div class="interpretation-text">{bollinger_text}<br><br>Upper Band: ₹{latest_bb_upper:,.2f}<br>Lower Band: ₹{latest_bb_lower:,.2f}</div></div><div class="interpretation-item"><div class="interpretation-label">Technical Overview</div><div class="interpretation-text">Trend: {trend_signal}<br>Momentum: {rsi_signal}<br>MACD: {macd_signal}<br>Volatility Position: {bollinger_signal}</div></div></div><div class="interpretation-note">These are technical indicators generated from historical market data, not buy/sell guarantees.</div></div>')

# Benchmark
render_html('<div class="section-title">Benchmark Comparison</div><div style="color:#64748b;font-size:.9rem;margin-bottom:1rem;">Selected company versus the NIFTY 50 over the same historical period.</div>')
comparison_df=pd.concat([stock_close.rename(selected_company),benchmark_close.rename("NIFTY 50")],axis=1).dropna(); normalized_comparison=comparison_df/comparison_df.iloc[0]*100
stock_total_return=(comparison_df[selected_company].iloc[-1]/comparison_df[selected_company].iloc[0])-1; benchmark_total_return=(comparison_df["NIFTY 50"].iloc[-1]/comparison_df["NIFTY 50"].iloc[0])-1; trading_days=len(comparison_df); stock_annualized_return=(1+stock_total_return)**(252/trading_days)-1; benchmark_annualized_return=(1+benchmark_total_return)**(252/trading_days)-1
daily_returns_comparison=comparison_df.pct_change().dropna(); stock_volatility=daily_returns_comparison[selected_company].std()*np.sqrt(252); benchmark_volatility=daily_returns_comparison["NIFTY 50"].std()*np.sqrt(252); relative_total_return=stock_total_return-benchmark_total_return; relative_annualized_return=stock_annualized_return-benchmark_annualized_return
if relative_annualized_return>.05: benchmark_assessment="Strong Outperformance"; benchmark_text=f"{selected_company} generated an annualized return {relative_annualized_return:.2%} higher than the NIFTY 50."
elif relative_annualized_return>0: benchmark_assessment="Outperformed"; benchmark_text=f"{selected_company} generated a higher annualized return than the NIFTY 50 by {relative_annualized_return:.2%}."
elif relative_annualized_return>-.05: benchmark_assessment="Slight Underperformance"; benchmark_text=f"{selected_company} generated a lower annualized return than the NIFTY 50 by {abs(relative_annualized_return):.2%}."
else: benchmark_assessment="Underperformed"; benchmark_text=f"{selected_company} generated an annualized return {abs(relative_annualized_return):.2%} below the NIFTY 50."
benchmark_fig=go.Figure(); benchmark_fig.add_trace(go.Scatter(x=normalized_comparison.index,y=normalized_comparison[selected_company],mode="lines",name=selected_company,line=dict(width=3))); benchmark_fig.add_trace(go.Scatter(x=normalized_comparison.index,y=normalized_comparison["NIFTY 50"],mode="lines",name="NIFTY 50",line=dict(width=3))); benchmark_fig.add_hline(y=100,line_dash="dot",annotation_text="Starting value: ₹100")
benchmark_fig.update_layout(**DARK_CHART,height=480,margin=dict(l=20,r=20,t=20,b=20),xaxis=dict(title="Date",showgrid=False),yaxis=dict(title="Growth of ₹100",showgrid=True,gridcolor=DARK_GRID),hovermode="x unified",legend=LEGEND_TOP); st.plotly_chart(benchmark_fig,use_container_width=True,theme=None)
render_html(f'<div class="interpretation-card"><div class="interpretation-overall">Benchmark Assessment: {benchmark_assessment}</div><div class="interpretation-overall-text">{benchmark_text}</div><div class="interpretation-grid"><div class="interpretation-item"><div class="interpretation-label">{selected_company} Annualized Return</div><div class="interpretation-text">{stock_annualized_return:.2%}</div></div><div class="interpretation-item"><div class="interpretation-label">NIFTY 50 Annualized Return</div><div class="interpretation-text">{benchmark_annualized_return:.2%}</div></div><div class="interpretation-item"><div class="interpretation-label">{selected_company} Volatility</div><div class="interpretation-text">{stock_volatility:.2%}</div></div><div class="interpretation-item"><div class="interpretation-label">NIFTY 50 Volatility</div><div class="interpretation-text">{benchmark_volatility:.2%}</div></div></div><div class="interpretation-note">Benchmark comparison is historical and does not predict future returns.</div></div>')

# Peer comparison
render_html('<div class="section-title">Insurance Peer Comparison</div><div style="color:#64748b;font-size:.9rem;margin-bottom:1rem;">Historical risk and return comparison across the available investment universe.</div>')
@st.cache_data(ttl=3600)
def load_peer_data():
    peer_results=[]
    for company_name,ticker in INSURANCE_STOCKS.items():
        try:
            peer_df=get_price_data(ticker,"2y"); peer_close=peer_df.set_index("Date")["Close"].dropna(); peer_metrics=performance_summary(peer_close,benchmark_close)
            peer_results.append({"Company":company_name,"Ticker":ticker,"Category":COMPANY_CATEGORIES.get(company_name,"Investment"),"Annual Return":peer_metrics["Annual Return"],"Annual Volatility":peer_metrics["Annual Volatility"],"Sharpe Ratio":peer_metrics["Sharpe Ratio"],"Maximum Drawdown":peer_metrics["Maximum Drawdown"],"Beta":peer_metrics["Beta"]})
        except Exception: continue
    return pd.DataFrame(peer_results)
with st.spinner("Comparing companies..."): peer_df=load_peer_data()
if peer_df.empty: st.warning("Peer comparison data could not be loaded right now. Please refresh.")
else:
    numeric_columns=["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown","Beta"]
    for column in numeric_columns: peer_df[column]=pd.to_numeric(peer_df[column],errors="coerce")
    peer_df=peer_df.dropna(subset=["Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown"])
    peer_df["Peer Score"]=peer_df["Annual Return"].rank(pct=True)*100*.30+peer_df["Sharpe Ratio"].rank(pct=True)*100*.30+peer_df["Annual Volatility"].rank(pct=True,ascending=False)*100*.20+peer_df["Maximum Drawdown"].rank(pct=True)*100*.20
    peer_df["Rank"]=peer_df["Peer Score"].rank(ascending=False,method="min").astype(int); peer_df=peer_df.sort_values("Peer Score",ascending=False).reset_index(drop=True)
    peer_chart_df=peer_df.sort_values("Peer Score",ascending=True); other_peers=peer_chart_df[peer_chart_df["Company"]!=selected_company]; selected_peer=peer_chart_df[peer_chart_df["Company"]==selected_company]
    peer_fig=go.Figure(); peer_fig.add_trace(go.Bar(x=other_peers["Peer Score"],y=other_peers["Company"],orientation="h",name="Other Companies",marker=dict(color="#CBD5E1"),text=[f"{v:.1f}" for v in other_peers["Peer Score"]],textposition="outside"))
    if not selected_peer.empty: peer_fig.add_trace(go.Bar(x=selected_peer["Peer Score"],y=selected_peer["Company"],orientation="h",name=f"Selected: {selected_company}",marker=dict(color="#3B82F6"),text=[f"{v:.1f}" for v in selected_peer["Peer Score"]],textposition="outside"))
    peer_fig.update_layout(**DARK_CHART,height=max(420,30*len(peer_df)),margin=dict(l=20,r=80,t=40,b=20),xaxis=dict(title="Peer Score",range=[0,max(100,peer_df["Peer Score"].max()+10)]),yaxis=dict(title="",categoryorder="array",categoryarray=peer_chart_df["Company"].tolist()),legend=LEGEND_TOP_RIGHT,bargap=.25); st.plotly_chart(peer_fig,use_container_width=True,theme=None)
    display_peer_df=peer_df[["Rank","Company","Category","Annual Return","Annual Volatility","Sharpe Ratio","Maximum Drawdown","Beta","Peer Score"]].copy(); display_peer_df.columns=["Rank","Company","Category","Annual Return","Volatility","Sharpe Ratio","Maximum Drawdown","Beta","Peer Score"]
    st.dataframe(display_peer_df.style.format({"Annual Return":"{:.2%}","Volatility":"{:.2%}","Sharpe Ratio":"{:.2f}","Maximum Drawdown":"{:.2%}","Beta":"{:.2f}","Peer Score":"{:.1f}"}),use_container_width=True,hide_index=True)

# Scenario simulation
render_html('<div class="section-title">1-Year Scenario Simulation</div><div style="color:#64748b;font-size:.9rem;margin-bottom:1rem;">Probabilistic price scenarios based on the selected company’s historical return and volatility.</div>')
NUM_SIMULATIONS=10000; TRADING_DAYS=252; DISPLAY_PATHS=100; historical_returns=stock_df["Close"].pct_change().dropna(); daily_mean_return=historical_returns.mean(); daily_volatility=historical_returns.std(); current_price=stock_df["Close"].iloc[-1]; np.random.seed(42); random_returns=np.random.normal(loc=daily_mean_return,scale=daily_volatility,size=(TRADING_DAYS,NUM_SIMULATIONS)); price_paths=np.zeros((TRADING_DAYS+1,NUM_SIMULATIONS)); price_paths[0]=current_price
for day in range(1,TRADING_DAYS+1): price_paths[day]=price_paths[day-1]*(1+random_returns[day-1])
final_prices=price_paths[-1]; median_price=np.percentile(final_prices,50); lower_price=np.percentile(final_prices,5); upper_price=np.percentile(final_prices,95); probability_above_current=np.mean(final_prices>current_price); simulated_median_return=(median_price/current_price)-1
simulation_fig=go.Figure(); days=np.arange(TRADING_DAYS+1)
for i in range(DISPLAY_PATHS): simulation_fig.add_trace(go.Scatter(x=days,y=price_paths[:,i],mode="lines",line=dict(width=1),opacity=.12,showlegend=False,hoverinfo="skip"))
simulation_fig.add_trace(go.Scatter(x=days,y=np.percentile(price_paths,50,axis=1),mode="lines",name="Median Path",line=dict(width=3))); simulation_fig.add_trace(go.Scatter(x=days,y=np.percentile(price_paths,5,axis=1),mode="lines",name="5th Percentile",line=dict(width=2,dash="dash"))); simulation_fig.add_trace(go.Scatter(x=days,y=np.percentile(price_paths,95,axis=1),mode="lines",name="95th Percentile",line=dict(width=2,dash="dash"))); simulation_fig.add_hline(y=current_price,line_dash="dot",annotation_text=f"Current Price: ₹{current_price:,.2f}")
simulation_fig.update_layout(**DARK_CHART,height=520,margin=dict(l=20,r=20,t=20,b=20),xaxis=dict(title="Trading Days Ahead"),yaxis=dict(title="Simulated Price (₹)",showgrid=True,gridcolor=DARK_GRID),hovermode="x unified",legend=LEGEND_TOP_RIGHT); st.plotly_chart(simulation_fig,use_container_width=True,theme=None)
mc1,mc2,mc3,mc4=st.columns(4)
for c,label,value,desc in [(mc1,"Current Price",f"₹{current_price:,.2f}","Latest market price"),(mc2,"Median Price",f"₹{median_price:,.2f}","50th percentile after 1 year"),(mc3,"5th–95th Range",f"₹{lower_price:,.0f} – ₹{upper_price:,.0f}","Simulated price range"),(mc4,"Probability Above Current",f"{probability_above_current:.1%}","Simulated paths ending higher")]:
    with c: render_html(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-description">{desc}</div></div>')
render_html(f'<div class="interpretation-card"><div class="interpretation-overall">Scenario result</div><div class="interpretation-overall-text">The median simulated price is ₹{median_price:,.2f}, representing a simulated return of {simulated_median_return:.2%}. This is a probability-based scenario, not a forecast or guarantee.</div></div>')

# Risk-return interpretation
render_html('<div class="section-title">Dynamic Risk & Return Interpretation</div>')
annual_return_value=metrics["Annual Return"]; volatility_value=metrics["Annual Volatility"]; sharpe_value=metrics["Sharpe Ratio"]; drawdown_value=metrics["Maximum Drawdown"]; beta_value=metrics["Beta"]
return_text=f"Historical annualized return: {annual_return_value:.2%}."; volatility_text=f"Historical annual volatility: {volatility_value:.2%}."; sharpe_text=f"Sharpe Ratio: {sharpe_value:.2f}, indicating the historical return relative to risk taken."; drawdown_text=f"Maximum historical drawdown: {drawdown_value:.2%}."; beta_text=f"Beta: {beta_value:.2f}, measuring sensitivity to NIFTY 50 movements."
render_html(f'<div class="interpretation-card"><div class="interpretation-overall">Quick read</div><div class="interpretation-overall-text">This company should be interpreted using return, volatility, risk-adjusted performance and downside risk together rather than any single metric.</div><div class="interpretation-grid"><div class="interpretation-item"><div class="interpretation-label">Return</div><div class="interpretation-text">{return_text}</div></div><div class="interpretation-item"><div class="interpretation-label">Risk</div><div class="interpretation-text">{volatility_text}</div></div><div class="interpretation-item"><div class="interpretation-label">Risk-adjusted Performance</div><div class="interpretation-text">{sharpe_text}</div></div><div class="interpretation-item"><div class="interpretation-label">Downside Risk</div><div class="interpretation-text">{drawdown_text}</div></div><div class="interpretation-item"><div class="interpretation-label">Market Sensitivity</div><div class="interpretation-text">{beta_text}</div></div></div></div>')

render_html('<div class="section-title">Historical Market Data</div>')
st.caption("Most recent 20 trading days for the selected company.")
st.dataframe(stock_df.tail(20),use_container_width=True,hide_index=True)
render_html('<div class="footer">InsureInvest • Historical market analytics prototype<br>Data sourced through Yahoo Finance via yfinance.</div>')
