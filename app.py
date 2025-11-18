import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="BrayFXTrade Analyzer", layout="wide", page_icon="💹", initial_sidebar_state="expanded")

# --- Sidebar ---
st.sidebar.title("BrayFXTrade Analyzer")
ticker = st.sidebar.text_input("Ticker (Forex pair, e.g., EURUSD=X)", value="EURUSD=X")
timeframe = st.sidebar.selectbox("Timeframe", ["1mo","1wk","1d","4h","2h","1h","30m","15m","5m","3m","1m"], index=2)
start_date = st.sidebar.date_input("Start Date", datetime(2023,1,1))
end_date = st.sidebar.date_input("End Date", datetime.today())
mode = st.sidebar.selectbox("Mode", ["Signal Generation", "Backtest"], index=1)

st.sidebar.markdown("**Analysis Dashboard**")

# --- Fetch Data ---
@st.cache_data
def fetch_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df = df.reset_index()
    required_cols = ['Open','High','Low','Close','Volume']
    existing_cols = [c for c in required_cols if c in df.columns]
    if len(existing_cols)<5:
        st.error("Missing required OHLCV columns in fetched data")
        return pd.DataFrame()
    df = df.dropna(subset=existing_cols)
    df.rename(columns={"Date":"Datetime"}, inplace=True)
    return df

df = fetch_data(ticker, start_date, end_date, timeframe)
if df.empty:
    st.stop()
st.success(f"Data fetched successfully with {len(df)} rows.")

# --- OB/FVG Detection ---
def detect_order_blocks(df):
    df['OB_type'] = None
    df['OB_top'] = np.nan
    df['OB_bottom'] = np.nan
    df['FVG_top'] = np.nan
    df['FVG_bottom'] = np.nan

    for i in range(2,len(df)):
        # Simplified Bullish OB
        if df['Close'].iloc[i-2] > df['Open'].iloc[i-2] and df['Close'].iloc[i-1] < df['Open'].iloc[i-1]:
            df.at[i,'OB_type'] = 'Bullish'
            df.at[i,'OB_top'] = df['High'].iloc[i-1]
            df.at[i,'OB_bottom'] = df['Low'].iloc[i-1]
            # FVG zone after OB
            df.at[i,'FVG_top'] = df['High'].iloc[i-1]
            df.at[i,'FVG_bottom'] = df['Low'].iloc[i-1]
        # Bearish OB
        elif df['Close'].iloc[i-2] < df['Open'].iloc[i-2] and df['Close'].iloc[i-1] > df['Open'].iloc[i-1]:
            df.at[i,'OB_type'] = 'Bearish'
            df.at[i,'OB_top'] = df['High'].iloc[i-1]
            df.at[i,'OB_bottom'] = df['Low'].iloc[i-1]
            df.at[i,'FVG_top'] = df['High'].iloc[i-1]
            df.at[i,'FVG_bottom'] = df['Low'].iloc[i-1]

    return df

# --- Pattern Detection ---
def detect_patterns(df):
    df['Pattern'] = None
    patterns = []
    for i in range(3,len(df)-3):
        highs = df['High'].iloc[i-3:i+1].values
        lows = df['Low'].iloc[i-3:i+1].values

        # Double Top
        if highs[0]<highs[1]>highs[2] and abs(highs[1]-highs[2])/highs[1]<0.003:
            df.at[i,'Pattern'] = 'Double Top'
            patterns.append((i,'Double Top'))
        # Double Bottom
        elif lows[0]>lows[1]<lows[2] and abs(lows[1]-lows[2])/lows[1]<0.003:
            df.at[i,'Pattern'] = 'Double Bottom'
            patterns.append((i,'Double Bottom'))
        # Rising Wedge
        elif all(np.diff(lows[-3:])>0) and all(np.diff(highs[-3:])>0) and (highs[-1]-lows[-3])<0.01:
            df.at[i,'Pattern'] = 'Rising Wedge'
            patterns.append((i,'Rising Wedge'))
        # Falling Wedge
        elif all(np.diff(lows[-3:])<0) and all(np.diff(highs[-3:])<0) and (highs[-3]-lows[-1])<0.01:
            df.at[i,'Pattern'] = 'Falling Wedge'
            patterns.append((i,'Falling Wedge'))
        # Symmetrical Triangle
        elif max(highs)-min(lows)<0.01:
            df.at[i,'Pattern'] = 'Symmetrical Triangle'
            patterns.append((i,'Symmetrical Triangle'))
        # Ascending Triangle
        elif abs(highs[-1]-highs[-2])<0.0003 and all(np.diff(lows[-3:])>0):
            df.at[i,'Pattern'] = 'Ascending Triangle'
            patterns.append((i,'Ascending Triangle'))
        # Descending Triangle
        elif abs(lows[-1]-lows[-2])<0.0003 and all(np.diff(highs[-3:])<0):
            df.at[i,'Pattern'] = 'Descending Triangle'
            patterns.append((i,'Descending Triangle'))
        # Rectangle
        elif abs(highs[-1]-highs[-3])<0.0005 and abs(lows[-1]-lows[-3])<0.0005:
            df.at[i,'Pattern'] = 'Rectangle'
            patterns.append((i,'Rectangle'))
    return df, patterns

# --- Run Analysis ---
df = detect_order_blocks(df)
df, patterns = detect_patterns(df)

# --- Generate Signals ---
df['signal'] = None
df['entry'] = np.nan
df['stoploss'] = np.nan
df['takeprofit'] = np.nan

for i in range(len(df)):
    if df['OB_type'].iloc[i]=='Bullish':
        df.at[i,'signal'] = 'Buy'
        df.at[i,'entry'] = df['FVG_bottom'].iloc[i]
        df.at[i,'stoploss'] = df['FVG_bottom'].iloc[i] - 0.02*(df['FVG_top'].iloc[i]-df['FVG_bottom'].iloc[i])
        df.at[i,'takeprofit'] = df['entry'].iloc[i] + 3*(df['entry'].iloc[i]-df['stoploss'].iloc[i])
    elif df['OB_type'].iloc[i]=='Bearish':
        df.at[i,'signal'] = 'Sell'
        df.at[i,'entry'] = df['FVG_top'].iloc[i]
        df.at[i,'stoploss'] = df['FVG_top'].iloc[i] + 0.02*(df['FVG_top'].iloc[i]-df['FVG_bottom'].iloc[i])
        df.at[i,'takeprofit'] = df['entry'].iloc[i] - 3*(df['stoploss'].iloc[i]-df['entry'].iloc[i])

# --- Performance Summary ---
if mode=="Backtest":
    df_bt = df.copy()
    total_signals = df_bt['signal'].notna().sum()
    st.write(f"Total Signals: {total_signals}")
else:
    df_bt = df[df['signal'].notna()]

# --- Plot Candlestick Chart ---
fig = go.Figure(go.Candlestick(
    x=df['Datetime'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name="Candlestick"
))

# Overlay OB/FVG
for i in range(len(df)):
    if not pd.isna(df['OB_top'].iloc[i]):
        color='green' if df['OB_type'].iloc[i]=='Bullish' else 'red'
        fig.add_shape(type="rect",
                      x0=df['Datetime'].iloc[i], x1=df['Datetime'].iloc[i],
                      y0=df['OB_bottom'].iloc[i], y1=df['OB_top'].iloc[i],
                      line=dict(color=color), fillcolor=color, opacity=0.3)

# Overlay Patterns
for idx, pat in patterns:
    fig.add_trace(go.Scatter(
        x=[df['Datetime'].iloc[idx]],
        y=[df['High'].iloc[idx]+0.0005],
        mode="markers+text",
        marker=dict(size=10,color='yellow'),
        text=[pat],
        textposition="top center",
        name="Pattern"
    ))

fig.update_layout(
    template="plotly_dark",
    title=f"{ticker} Candlestick Chart with OB/FVG & Patterns",
    xaxis_title="Datetime",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)
st.write("Analysis complete.")

# --- Signals Table ---
st.subheader("Signals Table")
display_cols = ['Datetime','signal','entry','stoploss','takeprofit','OB_type','Pattern']
st.dataframe(df[display_cols].dropna(subset=['signal']), height=400)
