# Home.py
import streamlit as st
from datetime import datetime, timedelta
from utils import fetch_data, INTERVAL_MAP, run_analysis

st.set_page_config(page_title="BrayFXTrade Analyzer", layout="wide")
st.title("💹 BrayFXTrade Analyzer - Home")

# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Backtest","Live Analysis"])
    pair = st.selectbox("Forex Pair", ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","GBPJPY","BTCUSD","ETHUSD"])
    timeframe = st.selectbox("Timeframe", ["1H","4H","1D","15m","5m"])
    today = datetime.today()
    start_date = st.date_input("Start Date", today - timedelta(days=365))
    end_date = st.date_input("End Date", today)
    st.markdown("---")
    st.subheader("Strategies")
    use_ob = st.checkbox("OB", True)
    use_fvg = st.checkbox("FVG", True)
    use_patterns = st.checkbox("Patterns", True)
    st.markdown("---")
    run_analysis_btn = st.button("▶️ Run Analysis")  # Sidebar button

# Run Analysis
if run_analysis_btn:
    interval = INTERVAL_MAP.get(timeframe,"60m")
    df = fetch_data(pair, start_date, end_date, interval)
    if df.empty:
        st.error(f"No data for {pair} in {timeframe}")
    else:
        st.session_state['market_df'] = df
        st.session_state['pair'] = pair
        st.session_state['timeframe'] = timeframe
        signals_df = run_analysis(df, use_ob, use_fvg, use_patterns)
        st.session_state['signals_df'] = signals_df
        st.success(f"Loaded {len(df)} rows | Signals detected: {len(signals_df)}")
        st.info("Go to the **TradingView** or **Signals** page to see chart and trades")
