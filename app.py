# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import plotly.graph_objects as go

st.set_page_config(page_title="BrayFXTrade Analyzer", layout="wide")

st.title("BrayFXTrade Analyzer")
st.write("This app fetches OHLCV from Yahoo Finance and detects OB/FVG zones and chart patterns.")

# ---------------------- User Inputs ----------------------
ticker = st.text_input("Ticker", value="EURUSD=X")
timeframe = st.selectbox(
    "Timeframe",
    options=["1mo", "1wk", "1d", "4h", "2h", "1h", "30m", "15m", "5m", "3m", "1m"],
    index=2
)
start_date = st.date_input("Start Date", value=datetime(2023,1,1))
end_date = st.date_input("End Date", value=datetime.now(timezone.utc).date())
mode = st.selectbox("Mode", options=["Signal Generation", "Backtest"], index=1)

# ---------------------- Fetch Data ----------------------
@st.cache_data
def fetch_data(ticker, start_date, end_date, interval):
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    if df.empty:
        st.error("No data found. Check ticker, interval, or date range.")
        return None
    # Ensure columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Missing OHLCV columns. Available columns: {df.columns.tolist()}")
        return None
    df = df.dropna(subset=required_cols)
    df.reset_index(inplace=True)
    return df

df = fetch_data(ticker, start_date, end_date, timeframe)
if df is None:
    st.stop()
st.success(f"Data fetched successfully with {len(df)} rows.")

# ---------------------- OB/FVG Detection ----------------------
def detect_order_blocks(df, lookback=20):
    df = df.copy()
    df['OB_type'] = None
    df['OB_top'] = np.nan
    df['OB_bottom'] = np.nan

    for i in range(lookback, len(df)):
        recent = df.iloc[i-lookback:i]
        recent_close = recent['Close'].dropna()
        if recent_close.empty:
            continue
        recent_max = recent_close.max()
        recent_min = recent_close.min()
        current_close = df['Close'].iloc[i]

        if current_close > recent_max:
            df.loc[df.index[i], 'OB_type'] = 'bullish'
            df.loc[df.index[i], 'OB_top'] = df['High'].iloc[i]
            df.loc[df.index[i], 'OB_bottom'] = df['Low'].iloc[i]
        elif current_close < recent_min:
            df.loc[df.index[i], 'OB_type'] = 'bearish'
            df.loc[df.index[i], 'OB_top'] = df['High'].iloc[i]
            df.loc[df.index[i], 'OB_bottom'] = df['Low'].iloc[i]
    return df

def detect_fvg(df):
    df = df.copy()
    df['FVG_top'] = np.nan
    df['FVG_bottom'] = np.nan

    for i in range(2, len(df)):
        if pd.notna(df['OB_type'].iloc[i-2]):
            if df['OB_type'].iloc[i-2] == 'bullish':
                if df['Low'].iloc[i] > df['High'].iloc[i-2]:
                    df.loc[df.index[i], 'FVG_top'] = df['High'].iloc[i-2]
                    df.loc[df.index[i], 'FVG_bottom'] = df['Low'].iloc[i]
            elif df['OB_type'].iloc[i-2] == 'bearish':
                if df['High'].iloc[i] < df['Low'].iloc[i-2]:
                    df.loc[df.index[i], 'FVG_top'] = df['High'].iloc[i]
                    df.loc[df.index[i], 'FVG_bottom'] = df['Low'].iloc[i-2]
    return df

# ---------------------- Simple Pattern Detection ----------------------
def detect_patterns(df):
    df = df.copy()
    df['Pattern'] = None
    # For demo, we detect simple bullish/bearish rectangles
    for i in range(5, len(df)-5):
        window = df['Close'].iloc[i-5:i+5]
        if window.max() - window.min() < (window.mean() * 0.005):  # tiny range = rectangle
            df.loc[df.index[i], 'Pattern'] = 'Rectangle'
    return df

# ---------------------- Apply Detection ----------------------
df = detect_order_blocks(df)
df = detect_fvg(df)
df = detect_patterns(df)

# ---------------------- Generate Signals ----------------------
df['signal'] = None
for i in range(len(df)):
    if df['OB_type'].iloc[i] == 'bullish' and pd.notna(df['FVG_bottom'].iloc[i]):
        df.loc[df.index[i], 'signal'] = 'buy'
    elif df['OB_type'].iloc[i] == 'bearish' and pd.notna(df['FVG_top'].iloc[i]):
        df.loc[df.index[i], 'signal'] = 'sell'

# ---------------------- Backtest ----------------------
if mode == "Backtest":
    df_bt = df.copy()
    df_bt['outcome'] = np.nan
    for i in range(len(df_bt)):
        signal = df_bt['signal'].iloc[i]
        if signal == 'buy':
            sl = df_bt['OB_bottom'].iloc[i] * 0.98
            tp = df_bt['OB_top'].iloc[i] + 3*(df_bt['OB_top'].iloc[i]-sl)
            if (df_bt['High'].iloc[i:] >= tp).any():
                df_bt.loc[df_bt.index[i], 'outcome'] = 'win'
            elif (df_bt['Low'].iloc[i:] <= sl).any():
                df_bt.loc[df_bt.index[i], 'outcome'] = 'loss'
        elif signal == 'sell':
            sl = df_bt['OB_top'].iloc[i] * 1.02
            tp = df_bt['OB_bottom'].iloc[i] - 3*(sl - df_bt['OB_bottom'].iloc[i])
            if (df_bt['Low'].iloc[i:] <= tp).any():
                df_bt.loc[df_bt.index[i], 'outcome'] = 'win'
            elif (df_bt['High'].iloc[i:] >= sl).any():
                df_bt.loc[df_bt.index[i], 'outcome'] = 'loss'

    total_signals = df_bt['signal'].notna().sum()
    wins = (df_bt['outcome'] == 'win').sum()
    losses = (df_bt['outcome'] == 'loss').sum()
    win_rate = (wins / (wins + losses)*100) if (wins+losses)>0 else np.nan

    st.subheader("Performance Summary")
    st.write(f"Total Signals: {total_signals}")
    st.write(f"Wins: {wins}")
    st.write(f"Losses: {losses}")
    st.write(f"Win Rate: {win_rate:.2f}%" if not np.isnan(win_rate) else "Win Rate: N/A")

# ---------------------- Display Signals Table ----------------------
st.subheader("Signals & Patterns Table")
display_cols = ['Datetime','signal','OB_type','OB_top','OB_bottom','FVG_top','FVG_bottom','Pattern','outcome']
df_display = df.reset_index().rename(columns={'index':'Datetime'})
available_cols = [c for c in display_cols if c in df_display.columns]
st.dataframe(df_display[available_cols].sort_values('Datetime', ascending=False))

# ---------------------- Plotly Chart ----------------------
st.subheader("Chart & Signals")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df['Date'] if 'Date' in df.columns else df['Datetime'],
    open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name='Price'
))
# OB/FVG Markers
for i, row in df.iterrows():
    if pd.notna(row['OB_top']):
        fig.add_shape(type='rect',
                      x0=row['Datetime'], x1=row['Datetime'],
                      y0=row['OB_bottom'], y1=row['OB_top'],
                      line=dict(color='green' if row['OB_type']=='bullish' else 'red'),
                      opacity=0.3)
    if pd.notna(row['FVG_top']):
        fig.add_shape(type='rect',
                      x0=row['Datetime'], x1=row['Datetime'],
                      y0=row['FVG_bottom'], y1=row['FVG_top'],
                      line=dict(color='blue'),
                      opacity=0.2)
st.plotly_chart(fig, use_container_width=True)

st.markdown("**Built by BrayFXTrade Analyzer — demo heuristics for OB/FVG and chart patterns.**")
