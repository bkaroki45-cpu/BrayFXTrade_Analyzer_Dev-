# BrayFXTrade Analyzer - Streamlit App (Enhanced Demo with Entry, SL, TP & Chart Markers)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(layout="wide", page_title="BrayFXTrade Analyzer")

# ---------------------- Analysis Functions ----------------------

def fetch_data(ticker: str, start: str, end: str, interval: str = '1d') -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if data.empty:
        return data
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Missing required columns from yfinance. Found: {data.columns.tolist()}")
        return pd.DataFrame()
    data = data[required_cols].dropna()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data

# ---------------------- Placeholder Analysis Functions ----------------------

def detect_order_blocks(df, lookback=20): return df.copy()
def detect_fvg(df): return df.copy()
def detect_patterns(df): return df.copy()

def generate_signals(df, strategy='OB+FVG'):
    df = df.copy()
    if 'signal' not in df.columns:
        df['signal'] = np.nan
        df['entry'] = np.nan
        df['sl'] = np.nan
        df['tp'] = np.nan
    # Dummy signals: every 5th bar is a buy
    df.loc[df.index[::5], 'signal'] = 'buy'
    df.loc[df.index[::5], 'entry'] = df['Close'][::5] * 1.001  # example entry slightly above close
    df.loc[df.index[::5], 'sl'] = df['Close'][::5] * 0.995     # example SL slightly below close
    df.loc[df.index[::5], 'tp'] = df['Close'][::5] * 1.01      # example TP slightly above close
    return df

def backtest_signals(df, look_forward=24):
    df = df.copy()
    if 'outcome' not in df.columns:
        df['outcome'] = np.nan
        df['profit'] = np.nan
    # Dummy backtest: 50% chance win for each signal
    np.random.seed(42)
    signals_idx = df.index[df['signal'].notna()]
    outcomes = np.random.choice(['win','loss'], size=len(signals_idx))
    df.loc[signals_idx, 'outcome'] = outcomes
    df.loc[signals_idx, 'profit'] = np.where(df.loc[signals_idx, 'outcome']=='win', df.loc[signals_idx, 'tp']-df.loc[signals_idx, 'entry'], df.loc[signals_idx, 'sl']-df.loc[signals_idx, 'entry'])
    return df

# ---------------------- Streamlit UI ----------------------

st.title("BrayFXTrade Analyzer")

with st.sidebar:
    st.header("Settings")
    pair = st.selectbox("Pair / Ticker", options=["EURUSD=X", "GBPUSD=X", "USDJPY=X", "BTC-USD", "ETH-USD", "AAPL"], index=0)
    interval = st.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
    strategy = st.selectbox("Strategy", options=["OB+FVG", "OB", "FVG", "Patterns"], index=0)
    mode = st.selectbox("Mode", options=["Signal Generation", "Backtest"], index=1)

    today = datetime.utcnow().date()
    default_start = today - timedelta(days=90)
    start_date = st.date_input("Start Date", default_start)
    end_date = st.date_input("End Date", today)

    look_forward = st.number_input("Backtest look-forward bars", min_value=1, max_value=500, value=48)
    run_button = st.button("Run Analysis")

st.markdown("This app fetches OHLCV from Yahoo Finance and performs OB/FVG/pattern detection.")

if run_button:
    with st.spinner("Fetching data..."):
        df = fetch_data(pair, start_date.strftime('%Y-%m-%d'), (end_date + timedelta(days=1)).strftime('%Y-%m-%d'), interval=interval)

    if df.empty:
        st.error("No valid data found. Please check your ticker, interval, or date range.")
    else:
        st.success(f"Data fetched successfully with {len(df)} rows.")

        # ---------------------- Run Analysis ----------------------
        df_ob = detect_order_blocks(df, lookback=20)
        df_fvg = detect_fvg(df_ob)
        df_pat = detect_patterns(df_fvg)
        df_sig = generate_signals(df_pat, strategy=strategy)

        if mode == 'Backtest':
            df_bt = backtest_signals(df_sig, look_forward=look_forward)
        else:
            df_bt = df_sig.copy()

        # ---------------------- Defensive Checks ----------------------
        for col in ['signal','outcome','entry','sl','tp','profit']:
            if col not in df_bt.columns:
                df_bt[col] = np.nan

        # ---------------------- Performance Summary ----------------------
        total_signals = df_bt['signal'].notna().sum()
        wins = (df_bt['outcome'] == 'win').sum()
        losses = (df_bt['outcome'] == 'loss').sum()
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else np.nan

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Signals", int(total_signals))
        col2.metric("Wins", int(wins))
        col3.metric("Losses", int(losses))
        col4.metric("Win Rate", f"{win_rate:.2f}%" if not np.isnan(win_rate) else "N/A")

        # ---------------------- Signals Table ----------------------
        st.subheader("Signals Table")
        display_cols = ['Open','High','Low','Close','Volume','signal','entry','sl','tp','outcome','profit']
        df_display = df_bt.reset_index()
        try:
            cols_to_show = ['index'] + [c for c in display_cols if c in df_display.columns]
            df_display = df_display[cols_to_show].rename(columns={'index':'Datetime'}).sort_values('Datetime', ascending=False)
        except KeyError:
            st.warning("Some columns are missing in the signals table. Showing available columns.")
            df_display = df_display.sort_values(df_display.columns[0], ascending=False)
        st.dataframe(df_display)

        # ---------------------- Chart & Markers ----------------------
        st.subheader("Chart & Signals")
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        fig.add_trace(go.Candlestick(x=df_bt.index, open=df_bt['Open'], high=df_bt['High'], low=df_bt['Low'], close=df_bt['Close'], name='Price'))

        # Add markers for signals
        buy_signals = df_bt[df_bt['signal']=='buy']
        sell_signals = df_bt[df_bt['signal']=='sell'] if 'sell' in df_bt['signal'].values else pd.DataFrame()
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['entry'], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['entry'], mode='markers', marker=dict(color='red', size=10, symbol='triangle-down'), name='Sell'))

        st.plotly_chart(fig, use_container_width=True)
        st.success("Analysis complete with Entry, SL, TP and chart markers.")

else:
    st.info("Configure settings in the sidebar and click 'Run Analysis' to begin.")

st.markdown("---")
st.markdown("Built by BrayFXTrade Analyzer — demo heuristics with signals and chart markers.")