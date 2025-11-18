import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# ----------------------------
st.set_page_config(page_title="BrayFXTrade Analyzer — Pro", layout="wide")
st.title("💹 BrayFXTrade Analyzer — Pro (TradingView-style)")

# ----------------------------
# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Backtest", "Live Analysis"])
    pair_clean = st.selectbox("Select Forex Pair", ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","GBPJPY","BTC","ETH"])
    timeframe = st.selectbox("Timeframe", ["1H","4H","1D","15m","5m"])
    today = datetime.today()
    start_date = st.date_input("Start Date", value=today - timedelta(days=365))
    end_date = st.date_input("End Date", value=today)
    st.markdown("---")
    st.subheader("Strategies")
    use_ob = st.checkbox("Order Blocks (OB)", value=True)
    use_fvg = st.checkbox("Fair Value Gaps (FVG)", value=True)
    use_patterns = st.checkbox("Chart Patterns (Engulfing)", value=True)
    st.markdown("---")
    st.subheader("Run / Live Options")
    auto_refresh = st.checkbox("Auto-refresh (Live only)", value=False)
    refresh_interval = st.number_input("Refresh interval (s)", min_value=10, value=60, step=10)

# Backend mapping for Yahoo Finance symbols
YF_MAP = {"EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X","AUDUSD":"AUDUSD=X",
          "USDCAD":"USDCAD=X","USDCHF":"USDCHF=X","NZDUSD":"NZDUSD=X","GBPJPY":"GBPJPY=X",
          "BTC":"BTC-USD","ETH":"ETH-USD"}
yf_symbol = YF_MAP[pair_clean]

# ----------------------------
# Fetch data
@st.cache_data(ttl=60)
def fetch_data(symbol,start,end,interval):
    try:
        df = yf.download(symbol, start=start, end=end + timedelta(days=1), interval=interval, progress=False)
        df = df.dropna(subset=['Open','High','Low','Close'])
        df.index = df.index.tz_localize(None)
        return df
    except:
        return pd.DataFrame()

TF_MAP = {"1H":"60m","4H":"60m","1D":"1d","15m":"15m","5m":"5m"}
raw_df = fetch_data(yf_symbol, start_date, end_date, TF_MAP.get(timeframe,"60m"))
# Resample 4H
if timeframe=="4H" and not raw_df.empty:
    df = raw_df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
else:
    df = raw_df.copy()

if df.empty:
    st.error(f"No data for {pair_clean} in {timeframe}. Try different range.")
    st.stop()
st.session_state['market_df'] = df

# ----------------------------
# Run Analysis button
st.markdown("---")
run_now = st.button("▶️ Run Analysis")

# ----------------------------
# OB/FVG/Patterns detection functions (same as your previous code)
def detect_ob(data):
    rows=[]
    for i in range(1,len(data)-1):
        prev, cur, nxt = data.iloc[i-1], data.iloc[i], data.iloc[i+1]
        body_cur = abs(cur.Close - cur.Open)
        body_prev = abs(prev.Close - prev.Open)
        body_next = abs(nxt.Close - nxt.Open)
        if cur.Close<cur.Open and body_cur>body_prev*1.0 and body_cur>body_next*0.8:
            rows.append({"Index":i,"Type":"Bearish OB","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"OB"})
        if cur.Close>cur.Open and body_cur>body_prev*1.0 and body_cur>body_next*0.8:
            rows.append({"Index":i,"Type":"Bullish OB","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"OB"})
    return pd.DataFrame(rows)

def detect_fvg(data):
    rows=[]
    for i in range(1,len(data)-1):
        cur, nxt = data.iloc[i], data.iloc[i+1]
        if cur.High<nxt.Low:
            rows.append({"Index":i,"Type":"Bullish FVG","Price":(cur.High+nxt.Low)/2,"Top":nxt.Low,"Bottom":cur.High,"Strategy":"FVG"})
        if cur.Low>nxt.High:
            rows.append({"Index":i,"Type":"Bearish FVG","Price":(cur.Low+nxt.High)/2,"Top":cur.Low,"Bottom":nxt.High,"Strategy":"FVG"})
    return pd.DataFrame(rows)

def detect_patterns(data):
    rows=[]
    for i in range(1,len(data)):
        prev, cur = data.iloc[i-1], data.iloc[i]
        if prev.Close<prev.Open and cur.Close>cur.Open and cur.Close>prev.Open and cur.Open<prev.Close:
            rows.append({"Index":i,"Type":"Bullish Engulfing","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"Pattern"})
        if prev.Close>prev.Open and cur.Close<cur.Open and cur.Open>prev.Close and cur.Close<prev.Open:
            rows.append({"Index":i,"Type":"Bearish Engulfing","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"Pattern"})
    return pd.DataFrame(rows)

# ----------------------------
# Run analysis logic
if run_now:
    df = st.session_state['market_df']
    signals_df=pd.DataFrame()
    if use_ob:
        s=detect_ob(df); s['StrategyName']='OB Strategy'; signals_df=pd.concat([signals_df,s],ignore_index=True)
    if use_fvg:
        s=detect_fvg(df); s['StrategyName']='FVG Strategy'; signals_df=pd.concat([signals_df,s],ignore_index=True)
    if use_patterns:
        s=detect_patterns(df); s['StrategyName']='Pattern Recognition'; signals_df=pd.concat([signals_df,s],ignore_index=True)
    st.session_state['signals_df']=signals_df

# ----------------------------
# Show results on Plotly chart
signals_df = st.session_state.get('signals_df', pd.DataFrame())
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Interactive Plotly Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        increasing_line_color='green', decreasing_line_color='red', name='Price'
    )])
    # Draw OB/FVG/patterns
    if not signals_df.empty:
        for _, s in signals_df.iterrows():
            idx=int(s['Index'])
            if idx>=len(df): continue
            t0 = df.index[idx]
            top, bottom = s['Top'], s['Bottom']
            color = 'rgba(0,255,0,0.2)' if 'Bull' in s['Type'] else 'rgba(255,0,0,0.15)'
            fig.add_shape(type="rect", x0=t0, x1=t0 + pd.Timedelta(hours=12), y0=bottom, y1=top,
                          fillcolor=color,line=dict(width=0), layer="below")
            fig.add_trace(go.Scatter(x=[df.index[idx]], y=[s['Price']], mode='markers+text',
                                     marker=dict(color='green' if 'Bull' in s['Type'] else 'red', size=10),
                                     text=[s['Type']], textposition='top center', name=s['Type']))
    fig.update_layout(height=700, margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Signals Table")
    if signals_df.empty:
        st.info("Click ▶️ Run Analysis to detect OB/FVG/patterns.")
    else:
        display=signals_df[['Index','Type','StrategyName','Price']].copy()
        display['Time'] = display['Index'].apply(lambda i: df.index[int(i)].strftime("%Y-%m-%d %H:%M"))
        st.dataframe(display.sort_values(by='Time', ascending=False).reset_index(drop=True), height=300)
