import streamlit as st

st.set_page_config(page_title="InsureInvest", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

home = st.Page("home.py", title="Home", icon="🏠", default=True)
company_analysis = st.Page("pages/Company_Analysis.py", title="Company Analysis", icon="📈")

pg = st.navigation([home, company_analysis], position="sidebar")
pg.run()
