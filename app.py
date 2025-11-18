# BrayFXTrade Analyzer - Streamlit App (Updated with OHLC check)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="BrayFXTrade Analyzer")

# ---------------------- Utility / Analysis Functions ----------------------

def fetch_data(ticker: str, start: str, end: str, interval: str = '1d') -> pd.DataFrame:
    """Fetch OHLCV data from yfinance and return cleaned DataFrame."""
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if data.empty:
        return data
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [c for c in required_cols if c not in data.columns]
    if missing_cols:
        st.error(f"Missing columns from yfinance: {missing_cols}. Try different ticker or interval.")
        return pd.DataFrame()  # empty to prevent crash

    data = data[required_cols].dropna()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data

# The rest of the functions (detect_order_blocks, detect_fvg, detect_patterns, generate_signals, backtest_signals) remain unchanged.
# Make sure each function has defensive checks like 'if df.empty: return df.copy()' before processing.

# ---------------------- Streamlit UI ----------------------

st.title("BrayFXTrade Analyzer")

with st.sidebar:
    st.header("Settings")
    pair = st.selectbox("Pair / Ticker (Yahoo Finance)", options=["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD", "AAPL"], index=0)
    interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)  # limited to supported
    strategy = st.selectbox("Strategy", options=["OB+FVG", "OB", "FVG", "Patterns"], index=0)
    mode = st.selectbox("Mode", options=["Signal Generation", "Backtest"], index=1)

    today = datetime.utcnow().date()
    default_start = today - timedelta(days=90)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", today)

    look_forward = st.number_input("Backtest look-forward bars", min_value=1, max_value=500, value=48)
    run_button = st.button("Run Analysis")

st.markdown("""
This app downloads OHLCV from Yahoo Finance (via `yfinance`) and performs OB/FVG/pattern detection.
""")

if run_button:
    with st.spinner("Fetching data..."):
        df = fetch_data(pair, start_date.strftime('%Y-%m-%d'), (end_date + timedelta(days=1)).strftime('%Y-%m-%d'), interval=interval)

    if df.empty:
        st.error("No valid data found. Please check your ticker, interval, or date range.")
    else:
        st.success(f"Data fetched successfully with {len(df)} rows.")
        # proceed with analysis safely
        # example: df_ob = detect_order_blocks(df) ... etc.
else:
    st.info("Configure settings in the sidebar and click 'Run Analysis' to begin.")


# End of file