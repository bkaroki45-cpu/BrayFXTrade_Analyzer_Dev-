# brayfxtrade_pro.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit.components.v1 as components
from io import BytesIO

# ---------------------------
# App config
st.set_page_config(page_title="BrayFXTrade Analyzer - Pro (TradingView-style)", layout="wide")
st.title("💹 BrayFXTrade Analyzer — Pro (TradingView-style)")

# ---------------------------
# Frontend / Backend currency mapping
FRONTEND_PAIRS = [
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD",
    "USDCHF","NZDUSD","GBPJPY","BTCUSD","ETHUSD"
]

PAIR_TO_YF = {
    "EURUSD":"EURUSD=X","GBPUSD":"GBPUSD=X","USDJPY":"USDJPY=X",
    "AUDUSD":"AUDUSD=X","USDCAD":"USDCAD=X","USDCHF":"USDCHF=X",
    "NZDUSD":"NZDUSD=X","GBPJPY":"GBPJPY=X","BTCUSD":"BTC-USD","ETHUSD":"ETH-USD"
}

# ---------------------------
# Utilities & mappings
TF_TO_YF_INTERVAL = {
    "1H": "60m",
    "4H": "60m",   # we'll resample to 4H from 60m
    "1D": "1d",
    "15m": "15m",
    "5m": "5m"
}

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
    # frontend pair selection
    pair_front = st.selectbox("Select Forex Pair", FRONTEND_PAIRS, index=2)
    pair = PAIR_TO_YF[pair_front]  # backend uses =X internally
    timeframe = st.selectbox("Timeframe", ["1H","4H","1D","15m","5m"], index=0)
    today = datetime.today()
    start_date = st.date_input("Start Date", value=today - timedelta(days=365*2))
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
# Top-row: TradingView widget + quick stats
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("TradingView Widget (interactive)")
    # TradingView widget embed (online)
    tv_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div id="tradingview_abc" style="height:520px;"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget(
    {{
      "width": "100%",
      "height": 520,
      "symbol": "{pair.replace('=','') if '=' in pair else pair}",
      "interval": "{timeframe}",
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#f1f3f6",
      "enable_publishing": false,
      "allow_symbol_change": true,
      "container_id": "tradingview_abc"
    }}
    );
    </script>
    <!-- TradingView Widget END -->
    """
    components.html(tv_html, height=540)

with col2:
    st.subheader("Quick Stats")
    st.write(f"Mode: **{mode}**")
    st.write(f"Pair: **{pair_front}**")
    st.write(f"Timeframe: **{timeframe}**")
    st.write(f"Range: **{start_date.isoformat()}** → **{end_date.isoformat()}**")
    st.write("Strategies enabled:")
    st.write(f"- OB: {'✅' if use_ob else '❌'}  ")
    st.write(f"- FVG: {'✅' if use_fvg else '❌'}  ")
    st.write(f"- Patterns: {'✅' if use_patterns else '❌'}  ")

# ---------------------------
# Data loader with caching
@st.cache_data(ttl=60)
def fetch_data(pair, start, end, yf_interval):
    """
    Fetch with yfinance; for multi-hour TF like 4H we fetch 1h and resample.
    """
    try:
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
raw_df = fetch_data(pair, start_date, end_date, yf_interval)

# Handle 4H resample if requested
if not raw_df.empty and timeframe == "4H":
    df = raw_df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
else:
    df = raw_df.copy()

if df.empty:
    st.error(f"No data for {pair_front} — try a different timeframe or shorter date range.")
    st.stop()

st.session_state['market_df'] = df

# ---------------------------
# Analysis functions (OB/FVG/Patterns)
# (same as your existing code; unchanged)

# ---------------------------
# Analysis runner
def run_analysis(df, use_ob, use_fvg, use_patterns):
    signals = pd.DataFrame()
    if use_ob:
        s = detect_ob(df)
        if not s.empty:
            s['StrategyName'] = 'OB Strategy'
            signals = pd.concat([signals, s], ignore_index=True)
    if use_fvg:
        s = detect_fvg(df)
        if not s.empty:
            s['StrategyName'] = 'FVG Strategy'
            signals = pd.concat([signals, s], ignore_index=True)
    if use_patterns:
        s = detect_patterns(df)
        if not s.empty:
            s['StrategyName'] = 'Pattern Recognition'
            signals = pd.concat([signals, s], ignore_index=True)
    if not signals.empty:
        signals['Index'] = signals['Index'].astype(int)
    return signals

# ---------------------------
# Trade simulation
# (same as your existing code; unchanged)

# ---------------------------
# Session controls & Run Analysis button
st.markdown("---")
run_col1, run_col2, run_col3 = st.columns([1,1,2])

with run_col1:
    run_now = st.button("▶️ Run Analysis")  # Added button

with run_col2:
    st.write("")  # spacer
    clear_state = st.button("🧹 Clear Results")

with run_col3:
    pass  # manual download handled below

if clear_state:
    for k in ['signals_df','trades_df','summary_df']:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# Auto-refresh for Live Mode
if mode == "Live Analysis" and auto_refresh:
    st.experimental_rerun()

# Run analysis when button is clicked
if run_now:
    df = st.session_state['market_df']
    signals_df = run_analysis(df, use_ob, use_fvg, use_patterns)
    st.session_state['signals_df'] = signals_df
    trades_df = simulate_trades(signals_df, df) if not signals_df.empty else pd.DataFrame()
    st.session_state['trades_df'] = trades_df

# Retrieve session state
signals_df = st.session_state.get('signals_df', pd.DataFrame())
trades_df = st.session_state.get('trades_df', pd.DataFrame())

# ---------------------------
# Left: Analysis outputs, Right: Charts
# (same as your existing code; unchanged, just showing results if run_now clicked)

# The rest of your code remains 100% the same, including charts, alerts, equity curves, export buttons, etc.

