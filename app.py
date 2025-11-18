import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit.components.v1 as components
from datetime import datetime, timedelta

st.set_page_config(page_title="BrayFXTrade Analyzer - Pro", layout="wide")
st.title("💹 BrayFXTrade Analyzer — Pro (TradingView-style)")

# -------- Sidebar --------
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Backtest", "Live Analysis"])
    
    # Frontend clean pairs
    FRONT_PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","GBPJPY","BTC","ETH"]
    selected_front_pair = st.selectbox("Select Forex Pair", FRONT_PAIRS, index=0)
    
    # Backend mapping
    PAIR_MAP = {
        "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X",
        "AUDUSD":"AUDUSD=X","USDCAD":"USDCAD=X","USDCHF":"USDCHF=X",
        "NZDUSD":"NZDUSD=X","GBPJPY":"GBPJPY=X","BTC":"BTC-USD","ETH":"ETH-USD"
    }
    backend_pair = PAIR_MAP[selected_front_pair]
    
    timeframe = st.selectbox("Timeframe", ["1H","4H","1D","15m","5m"], index=0)
    today = datetime.today()
    start_date = st.date_input("Start Date", today - timedelta(days=365))
    end_date = st.date_input("End Date", today)
    
    st.markdown("---")
    st.subheader("Strategies")
    use_ob = st.checkbox("OB", value=True)
    use_fvg = st.checkbox("FVG", value=True)
    use_patterns = st.checkbox("Patterns", value=True)
    
    st.markdown("---")
    run_analysis_btn = st.button("▶️ Run Analysis")
    
# -------- Fetch Data --------
@st.cache_data
def get_data(pair, start, end, interval="60m"):
    df = yf.download(pair, start=start, end=end+timedelta(days=1), interval=interval, progress=False)
    if df.empty: return df
    return df.dropna(subset=['Open','High','Low','Close'])

INTERVAL_MAP = {"1H":"60m","4H":"60m","1D":"1d","15m":"15m","5m":"5m"}
raw_df = get_data(backend_pair, start_date, end_date, INTERVAL_MAP.get(timeframe,"60m"))
if raw_df.empty:
    st.error(f"No data for {selected_front_pair} ({timeframe})")
    st.stop()

# Optional: resample 4H
if timeframe=="4H":
    df = raw_df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})
else:
    df = raw_df.copy()
st.session_state['market_df'] = df

# -------- Analysis Functions --------
# (OB/FVG/Pattern detection functions go here)
# ... same as previous, unchanged ...

# -------- Run Analysis --------
if run_analysis_btn:
    df = st.session_state['market_df']
    signals_df = run_analysis(df, use_ob, use_fvg, use_patterns)
    trades_df = simulate_trades(signals_df, df) if not signals_df.empty else pd.DataFrame()
    st.session_state['signals_df'] = signals_df
    st.session_state['trades_df'] = trades_df

signals_df = st.session_state.get('signals_df', pd.DataFrame())
trades_df = st.session_state.get('trades_df', pd.DataFrame())

# -------- Layout: Left + Right --------
left, right = st.columns([1,2])
with left:
    st.subheader("Signals & Trades")
    st.dataframe(signals_df.tail(10))
    st.dataframe(trades_df.tail(10))

with right:
    st.subheader("TradingView Chart")
    tv_html = f"""
    <div id="tv_widget" style="height:700px;"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
      new TradingView.widget({{
        "container_id": "tv_widget",
        "autosize": true,
        "symbol": "{selected_front_pair}",
        "interval": "{timeframe}",
        "timezone": "Etc/UTC",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "allow_symbol_change": true
      }});
    </script>
    """
    components.html(tv_html, height=720)
