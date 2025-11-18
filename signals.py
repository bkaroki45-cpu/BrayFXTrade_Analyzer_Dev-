# Signals.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Signals & Trades", layout="wide")
st.title("💹 Signals & Performance")

signals_df = st.session_state.get('signals_df', pd.DataFrame())
df = st.session_state.get('market_df', pd.DataFrame())

if signals_df.empty:
    st.info("No signals detected yet. Click **Run Analysis** in Home page.")
else:
    display = signals_df.copy()
    display['Time'] = display['Index'].apply(lambda i: df.index[int(i)].strftime("%Y-%m-%d %H:%M"))
    display = display[['Time','Type','Strategy','Price']]
    st.dataframe(display.sort_values(by='Time', ascending=False).reset_index(drop=True), height=400)
