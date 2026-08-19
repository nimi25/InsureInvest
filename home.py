import pandas as pd
import streamlit as st

from data import BENCHMARK, INSURANCE_STOCKS, COMPANY_CATEGORIES
from recommendation import analyze_universe, build_portfolio, recommendation_text

st.markdown("""
<style>
.stApp { background-color:#f6f8fb; }
.block-container { max-width:1400px; padding-top:2rem; padding-bottom:3rem; }
.hero { background:linear-gradient(135deg,#111827 0%,#1e3a5f 100%); padding:3rem; border-radius:20px; margin-bottom:1.5rem; }
.hero-title {color:#fff;font-size:3rem;font-weight:750;margin:0;}
.hero-subtitle {color:#cbd5e1;font-size:1.05rem;margin-top:.5rem;}
.card { background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:1.4rem;box-shadow:0 4px 15px rgba(15,23,42,.05); }
.title {color:#111827;font-size:1.4rem;font-weight:700;margin:1.5rem 0 .7rem;}
.muted {color:#64748b;font-size:.9rem;}
.big-score {font-size:2.2rem;font-weight:750;color:#111827;}
.small-label {color:#64748b;font-size:.78rem;font-weight:650;text-transform:uppercase;letter-spacing:.04em;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## InsureInvest")
st.sidebar.caption("Insurance Investment Analytics")
st.sidebar.markdown("---")
st.sidebar.info("Use Home to build a portfolio. Use Company Analysis to inspect an individual company in detail.")

st.markdown("""
<div class="hero">
    <div class="hero-title">InsureInvest</div>
    <div class="hero-subtitle">Data-driven insurance investment decision support</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Build your investment plan</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Tell InsureInvest how much you want to invest and your risk appetite. The model will rank the available companies and build a diversified historical-data-based portfolio.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    investment_amount = st.number_input("Investment amount (₹)", min_value=1000.0, value=100000.0, step=5000.0, format="%.0f")
with col2:
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Moderate", "Aggressive"], index=1)

if st.button("Analyze My Investment", type="primary", use_container_width=True):
    with st.spinner("Analyzing the investment universe..."):
        try:
            results, failures = analyze_universe(INSURANCE_STOCKS, BENCHMARK, risk_profile, "2y")
            if results.empty:
                st.error("No usable company data was returned. Please try again shortly.")
            else:
                st.session_state["home_results"] = results
                st.session_state["home_failures"] = failures
                st.session_state["home_profile"] = risk_profile
                st.session_state["home_amount"] = investment_amount
        except Exception as exc:
            st.error(f"The recommendation model could not complete the analysis: {exc}")

results = st.session_state.get("home_results")
stored_profile = st.session_state.get("home_profile", risk_profile)
stored_amount = st.session_state.get("home_amount", investment_amount)

if isinstance(results, pd.DataFrame) and not results.empty:
    portfolio = build_portfolio(results, stored_amount, max_companies=min(5, len(results)))
    st.markdown('<div class="title">Your recommended portfolio</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="muted">₹{stored_amount:,.0f} • {stored_profile} risk • Based on successfully loaded historical market data</div>', unsafe_allow_html=True)

    top_score = float(portfolio["Investment Score"].max())
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="card"><div class="small-label">Top company score</div><div class="big-score">{top_score:.1f}/100</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="card"><div class="small-label">Portfolio companies</div><div class="big-score">{len(portfolio)}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="card"><div class="small-label">Allocated</div><div class="big-score">{portfolio["Allocation %"].sum():.0f}%</div></div>', unsafe_allow_html=True)

    display = portfolio[["Investment Score", "Allocation %", "Recommended Amount"]].copy()
    display.insert(0, "Company", portfolio.index)
    display.insert(1, "Category", [COMPANY_CATEGORIES.get(company, "Investment") for company in portfolio.index])
    display["Investment Score"] = display["Investment Score"].map(lambda x: f"{x:.1f}/100")
    display["Allocation %"] = display["Allocation %"].map(lambda x: f"{x:.1f}%")
    display["Recommended Amount"] = display["Recommended Amount"].map(lambda x: f"₹{x:,.0f}")
    display.columns = ["Company", "Category", "Score", "Allocation", "Recommended Amount"]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown('<div class="title">Why these companies?</div>', unsafe_allow_html=True)
    for company, row in portfolio.iterrows():
        st.markdown(f'<div class="card" style="margin-bottom:.7rem;"><div class="small-label">{COMPANY_CATEGORIES.get(company, "Investment")}</div><h4 style="margin:.35rem 0;color:#111827;">{company}</h4><div class="muted">{recommendation_text(row)}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="title">Explore a company</div>', unsafe_allow_html=True)
    st.caption("Use the Company Analysis page from the navigation menu to inspect any listed company in detail.")

    with st.expander("How InsureInvest decides"):
        st.markdown("""
        The recommendation model evaluates historical annual return, Sharpe ratio, annual volatility, maximum drawdown and technical momentum.

        Conservative profiles place more weight on risk and downside protection. Moderate profiles balance return and risk. Aggressive profiles place more weight on return and momentum.

        The model ranks the successfully analysed companies and distributes the investment across the top five subject to diversification constraints.

        This is a historical-data-based decision-support model, not a guarantee of future returns.
        """)

    failures = st.session_state.get("home_failures", {})
    if failures:
        st.caption(f"{len(failures)} companies could not be analysed because market data was unavailable. They were excluded from this recommendation.")
else:
    st.markdown('<div class="card" style="margin-top:1.5rem;"><div class="small-label">Ready when you are</div><h3 style="margin:.4rem 0;color:#111827;">Enter an amount and click Analyze My Investment.</h3><div class="muted">The detailed company dashboards are intentionally kept off the homepage so the investment decision stays front and centre.</div></div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#94a3b8;font-size:.75rem;margin-top:3rem;padding-top:1rem;border-top:1px solid #e2e8f0;">InsureInvest • Historical market analytics prototype • Data via Yahoo Finance / yfinance</div>', unsafe_allow_html=True)
