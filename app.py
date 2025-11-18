# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="BrayFXTrade Analyzer", layout="wide", page_icon="💹")

# ----------------- Sidebar -----------------
st.sidebar.title("BrayFXTrade Analyzer Settings")
mode = st.sidebar.selectbox("Mode", options=["Signal Generation", "Backtest"], index=1)

pairs = st.sidebar.multiselect(
    "Select Forex Pairs",
    options=["EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X","USDCHF=X","NZDUSD=X"],
    default=["EURUSD=X"]
)

timeframe = st.sidebar.selectbox(
    "Timeframe",
    options=["1M","1W","1D","4H","2H","1H","30min","15min","5min","3min","1min"],
    index=2
)

start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-11-18"))

st.sidebar.markdown("---")

# ----------------- Data Fetching -----------------
@st.cache_data
def fetch_data(ticker, start, end, interval):
    interval_map = {
        "1M":"1mo","1W":"1wk","1D":"1d","4H":"4h","2H":"2h","1H":"1h",
        "30min":"30m","15min":"15m","5min":"5m","3min":"3m","1min":"1m"
    }
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval_map[interval])
        df = df.reset_index()
        if not all(col in df.columns for col in ['Open','High','Low','Close']):
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

# ----------------- OB/FVG Detection -----------------
def detect_order_blocks(df):
    df['OB_type'] = None
    df['OB_top'] = np.nan
    df['OB_bottom'] = np.nan
    for i in range(2, len(df)):
        # Simple bullish OB heuristic
        if df['Close'].iloc[i] > df['Close'].iloc[i-1] and df['Open'].iloc[i-1] > df['Close'].iloc[i-1]:
            df.at[i, 'OB_type'] = 'Bullish'
            df.at[i, 'OB_top'] = df['High'].iloc[i-1]
            df.at[i, 'OB_bottom'] = df['Low'].iloc[i-1]
        # Simple bearish OB heuristic
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1] and df['Open'].iloc[i-1] < df['Close'].iloc[i-1]:
            df.at[i, 'OB_type'] = 'Bearish'
            df.at[i, 'OB_top'] = df['High'].iloc[i-1]
            df.at[i, 'OB_bottom'] = df['Low'].iloc[i-1]
    return df

# ----------------- Pattern Detection -----------------
def detect_patterns(df):
    patterns = []
    # Dummy placeholders: implement actual detection algorithms here
    # For demonstration, mark a rising wedge if last 3 candles close higher than open
    for i in range(3, len(df)):
        if all(df['Close'].iloc[i-3:i] > df['Open'].iloc[i-3:i]):
            patterns.append((i, "Rising Wedge"))
        elif all(df['Close'].iloc[i-3:i] < df['Open'].iloc[i-3:i]):
            patterns.append((i, "Falling Wedge"))
    return df, patterns

# ----------------- Signal Generation / Backtest -----------------
def generate_signals(df):
    signals = []
    for i in range(len(df)):
        if df['OB_type'].iloc[i] == 'Bullish':
            entry = df['OB_bottom'].iloc[i]
            sl = entry - (df['OB_top'].iloc[i] - df['OB_bottom'].iloc[i])*0.02
            tp = entry + (df['OB_top'].iloc[i] - df['OB_bottom'].iloc[i])*3
            signals.append({'Datetime':df['Datetime'].iloc[i],'Type':'Buy','Entry':entry,'SL':sl,'TP':tp})
        elif df['OB_type'].iloc[i] == 'Bearish':
            entry = df['OB_top'].iloc[i]
            sl = entry + (df['OB_top'].iloc[i] - df['OB_bottom'].iloc[i])*0.02
            tp = entry - (df['OB_top'].iloc[i] - df['OB_bottom'].iloc[i])*3
            signals.append({'Datetime':df['Datetime'].iloc[i],'Type':'Sell','Entry':entry,'SL':sl,'TP':tp})
    return pd.DataFrame(signals)

# ----------------- Main Chart -----------------
st.title("💹 BrayFXTrade Analyzer")

combined_fig = go.Figure()
colors = ["cyan","magenta","orange","lime","blue","pink","yellow"]

for idx, pair in enumerate(pairs):
    with st.spinner(f"Fetching and analyzing {pair}..."):
        df = fetch_data(pair, start_date, end_date, timeframe)
        if df.empty:
            st.warning(f"No data for {pair}")
            continue
        
        df = detect_order_blocks(df)
        df, patterns = detect_patterns(df)
        
        color = colors[idx % len(colors)]
        
        # Candlestick
        combined_fig.add_trace(go.Candlestick(
            x=df['Datetime'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name=pair, increasing_line_color=color, decreasing_line_color=color, opacity=0.5
        ))
        
        # OB/FVG zones
        for i in range(len(df)):
            if not pd.isna(df['OB_top'].iloc[i]):
                zone_color = 'green' if df['OB_type'].iloc[i]=='Bullish' else 'red'
                combined_fig.add_shape(type="rect",
                                       x0=df['Datetime'].iloc[i], x1=df['Datetime'].iloc[i]+pd.Timedelta(minutes=1),
                                       y0=df['OB_bottom'].iloc[i], y1=df['OB_top'].iloc[i],
                                       line=dict(color=zone_color), fillcolor=zone_color, opacity=0.2)
        
        # Patterns
        for idx_pat, pat in patterns:
            combined_fig.add_trace(go.Scatter(
                x=[df['Datetime'].iloc[idx_pat]], y=[df['High'].iloc[idx_pat]+0.0005],
                mode="markers+text", marker=dict(size=8,color=color),
                text=[pat], textposition="top center", name=f"{pair} Pattern"
            ))
        
        # Signals / Backtest
        if mode == "Backtest":
            df_signals = generate_signals(df)
            for _, row in df_signals.iterrows():
                # Entry
                combined_fig.add_trace(go.Scatter(
                    x=[row['Datetime']], y=[row['Entry']],
                    mode="markers", marker=dict(size=10, symbol='triangle-up' if row['Type']=='Buy' else 'triangle-down', color=color),
                    name=f"{pair} Entry"
                ))
                # SL
                combined_fig.add_trace(go.Scatter(
                    x=[row['Datetime']], y=[row['SL']],
                    mode="markers", marker=dict(size=8, symbol='x', color='red'),
                    name=f"{pair} SL"
                ))
                # TP
                combined_fig.add_trace(go.Scatter(
                    x=[row['Datetime']], y=[row['TP']],
                    mode="markers", marker=dict(size=8, symbol='circle', color='lime'),
                    name=f"{pair} TP"
                ))

combined_fig.update_layout(template="plotly_dark",
                           title="Multi-Pair Candlestick Chart with OB/FVG, Patterns & Backtest Signals",
                           xaxis_title="Datetime", yaxis_title="Price",
                           legend=dict(orientation="h", y=-0.2))
st.plotly_chart(combined_fig, use_container_width=True)

st.markdown("**Built by BrayFXTrade Analyzer — demo heuristics for OB/FVG and chart patterns.**")

