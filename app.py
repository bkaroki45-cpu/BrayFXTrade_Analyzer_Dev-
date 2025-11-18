# brayfxtrade_pro.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import streamlit.components.v1 as components

# ---------------------------
# App config
st.set_page_config(page_title="BrayFXTrade Analyzer - Pro (TradingView-style)", layout="wide")
st.title("💹 BrayFXTrade Analyzer — Pro (TradingView-style)")

# ---------------------------
# Utilities & mappings
TF_TO_YF_INTERVAL = {
    "1H": "60m",
    "4H": "60m",   # we'll resample to 4H from 60m
    "1D": "1d",
    "15m": "15m",
    "5m": "5m"
}

# Frontend pairs (without =X)
FRONT_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD",
    "USDCHF","NZDUSD","GBPJPY","BTC","ETH"
]

# Map frontend pair to backend yfinance symbol
PAIR_MAP = { "EURUSD":"EURUSD=X", "GBPUSD":"GBPUSD=X", "USDJPY":"USDJPY=X",
             "AUDUSD":"AUDUSD=X", "USDCAD":"USDCAD=X", "USDCHF":"USDCHF=X",
             "NZDUSD":"NZDUSD=X", "GBPJPY":"GBPJPY=X", "BTC":"BTC-USD","ETH":"ETH-USD" }

# Simple helper to convert dataframe index to timezone-naive datetime
def to_localized_index(df):
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None) if df.index.tzinfo is not None else df.index
    return df

# ---------------------------
# Sidebar - controls
with st.sidebar:
    st.header("BrayFXTrade Analyzer Settings")
    mode = st.selectbox("Mode", ["Backtest", "Live Analysis"])
    pair_front = st.selectbox("Select Forex Pair", FRONT_PAIRS, index=0)
    pair_backend = PAIR_MAP[pair_front]  # backend uses Yahoo format
    timeframe = st.selectbox("Timeframe", ["1H","4H","1D","15m","5m"], index=0)
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
    st.markdown("---")
    export_csv = st.checkbox("Show Export Buttons", value=True)
    st.markdown("Built by BrayFXTrade Analyzer — demo heuristics for OB/FVG and chart patterns.")

# ---------------------------
# Data loader with caching
@st.cache_data(ttl=60)
def fetch_data(pair, start, end, yf_interval):
    try:
        # Convert date to datetime if needed
        if isinstance(start, datetime) == False:
            start = datetime.combine(start, time.min)
        if isinstance(end, datetime) == False:
            end = datetime.combine(end, time.min)
        # For intraday, limit to last 60 days
        if yf_interval in ["5m","15m","60m"]:
            start = max(start, datetime.today() - timedelta(days=60))
        df = yf.download(pair, start=start, end=end + timedelta(days=1), interval=yf_interval, progress=False)
        if df.empty:
            return df
        df = df.dropna(subset=['Open','High','Low','Close'])
        df = to_localized_index(df)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Map timeframe to yf interval
yf_interval = TF_TO_YF_INTERVAL.get(timeframe, "60m")
raw_df = fetch_data(pair_backend, start_date, end_date, yf_interval)

# Handle 4H resample if requested
if not raw_df.empty and timeframe == "4H":
    df = raw_df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
else:
    df = raw_df.copy()

if df.empty:
    st.error(f"No data for {pair_front} in {timeframe}. Try a different range.")
    st.stop()

st.session_state['market_df'] = df

# ---------------------------
# Analysis functions
def detect_ob(data):
    rows=[]
    for i in range(1,len(data)-1):
        prev, cur, nxt = data.iloc[i-1], data.iloc[i], data.iloc[i+1]
        body_cur, body_prev, body_next = abs(cur.Close-cur.Open), abs(prev.Close-prev.Open), abs(nxt.Close-nxt.Open)
        if cur.Close < cur.Open and body_cur > body_prev*1.0 and body_cur > body_next*0.8:
            rows.append({"Index":i,"Type":"Bearish OB","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"OB"})
        if cur.Close > cur.Open and body_cur > body_prev*1.0 and body_cur > body_next*0.8:
            rows.append({"Index":i,"Type":"Bullish OB","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"OB"})
    return pd.DataFrame(rows)

def detect_fvg(data):
    rows=[]
    for i in range(1,len(data)-1):
        cur,nxt = data.iloc[i], data.iloc[i+1]
        if cur.High < nxt.Low:
            rows.append({"Index":i,"Type":"Bullish FVG","Price":(cur.High+nxt.Low)/2,"Top":nxt.Low,"Bottom":cur.High,"Strategy":"FVG"})
        if cur.Low > nxt.High:
            rows.append({"Index":i,"Type":"Bearish FVG","Price":(cur.Low+nxt.High)/2,"Top":cur.Low,"Bottom":nxt.High,"Strategy":"FVG"})
    return pd.DataFrame(rows)

def detect_patterns(data):
    rows=[]
    for i in range(1,len(data)):
        prev, cur = data.iloc[i-1], data.iloc[i]
        if prev.Close < prev.Open and cur.Close > cur.Open and cur.Close > prev.Open and cur.Open < prev.Close:
            rows.append({"Index":i,"Type":"Bullish Engulfing","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"Pattern"})
        if prev.Close > prev.Open and cur.Close < cur.Open and cur.Open > prev.Close and cur.Close < prev.Open:
            rows.append({"Index":i,"Type":"Bearish Engulfing","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"Pattern"})
    return pd.DataFrame(rows)

# ---------------------------
# Analysis runner
def run_analysis(df,use_ob,use_fvg,use_patterns):
    signals = pd.DataFrame()
    if use_ob:
        s = detect_ob(df)
        if not s.empty:
            s['StrategyName']='OB Strategy'
            signals = pd.concat([signals,s],ignore_index=True)
    if use_fvg:
        s = detect_fvg(df)
        if not s.empty:
            s['StrategyName']='FVG Strategy'
            signals = pd.concat([signals,s],ignore_index=True)
    if use_patterns:
        s = detect_patterns(df)
        if not s.empty:
            s['StrategyName']='Pattern Recognition'
            signals = pd.concat([signals,s],ignore_index=True)
    if not signals.empty:
        signals['Index']=signals['Index'].astype(int)
    return signals

# ---------------------------
# Session controls & Run Analysis button
st.markdown("---")
run_col1, run_col2 = st.columns([1,1])

with run_col1:
    run_now = st.button("▶️ Run Analysis")

with run_col2:
    clear_state = st.button("🧹 Clear Results")

if clear_state:
    for k in ['signals_df','trades_df']:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# ---------------------------
# Run analysis when user clicks
if run_now:
    df = st.session_state['market_df']
    signals_df = run_analysis(df,use_ob,use_fvg,use_patterns)
    st.session_state['signals_df'] = signals_df

signals_df = st.session_state.get('signals_df', pd.DataFrame())

# ---------------------------
# Show TradingView widget
tv_html = f"""
<div id="tradingview_abc" style="height:520px;"></div>
<script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
<script type="text/javascript">
new TradingView.widget({{
  "width": "100%",
  "height": 520,
  "symbol": "{pair_front}",
  "interval": "{timeframe}",
  "timezone": "Etc/UTC",
  "theme": "dark",
  "style": "1",
  "locale": "en",
  "toolbar_bg": "#f1f3f6",
  "enable_publishing": false,
  "allow_symbol_change": true,
  "container_id": "tradingview_abc"
}});
</script>
"""
components.html(tv_html, height=540)

# ---------------------------
# Show signals on side table
st.subheader("Signals Detected")
if signals_df.empty:
    st.info("No signals detected yet. Click ▶️ Run Analysis to scan the range.")
else:
    display = signals_df.copy()
    display['Time'] = display['Index'].apply(lambda i: df.index[int(i)].strftime("%Y-%m-%d %H:%M"))
    display = display[['Time','Type','StrategyName','Price']]
    st.dataframe(display.sort_values(by='Time',ascending=False).reset_index(drop=True),height=300)

st.caption("BrayFXTrade Analyzer — demo heuristics. Use for research/backtesting only; not trading advice.")
