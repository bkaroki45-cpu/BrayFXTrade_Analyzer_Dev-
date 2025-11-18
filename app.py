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
# Utilities & mappings
TF_TO_YF_INTERVAL = {
    "1H": "60m",
    "4H": "60m",   # we'll resample to 4H from 60m
    "1D": "1d",
    "15m": "15m",
    "5m": "5m"
}

ALL_PAIRS = [
    "EURUSD=X","GBPUSD=X","USDJPY=X","AUDUSD=X","USDCAD=X",
    "USDCHF=X","NZDUSD=X","GBPJPY=X","BTC-USD","ETH-USD"
]

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
    pair = st.selectbox("Select Forex Pair", ALL_PAIRS, index=2)
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
    st.write(f"Pair: **{pair}**")
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
        # yfinance uses periods for intraday with small windows; when user supplies long start->end for 15m,
        # yfinance may fail — but we'll try the basic download and later handle resampling
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
    # ensure datetime index
    df = raw_df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
else:
    df = raw_df.copy()

# Basic no-data handling
if df.empty:
    st.error(f"No data for {pair} — try a different timeframe or shorter date range.")
    st.stop()

# Keep df in session_state so Run Analysis can reuse
st.session_state['market_df'] = df

# ---------------------------
# Analysis functions (heuristics)
def detect_ob(data):
    rows = []
    # Order block heuristic: large bullish/bearish candle followed by smaller retracement.
    for i in range(1, len(data)-1):
        prev, cur, nxt = data.iloc[i-1], data.iloc[i], data.iloc[i+1]
        body_cur = abs(cur.Close - cur.Open)
        body_prev = abs(prev.Close - prev.Open)
        body_next = abs(nxt.Close - nxt.Open)
        # Bearish OB: large bearish candle then smaller corrective move
        if cur.Close < cur.Open and body_cur > body_prev * 1.0 and body_cur > body_next * 0.8:
            rows.append({"Index": i, "Type":"Bearish OB", "Price": cur.Close,
                         "Top": cur.High, "Bottom": cur.Low, "Strategy":"OB"})
        # Bullish OB
        if cur.Close > cur.Open and body_cur > body_prev * 1.0 and body_cur > body_next * 0.8:
            rows.append({"Index": i, "Type":"Bullish OB", "Price": cur.Close,
                         "Top": cur.High, "Bottom": cur.Low, "Strategy":"OB"})
    return pd.DataFrame(rows)

def detect_fvg(data):
    rows=[]
    # FVG heuristic: gap between consecutive candles where there's a price range not overlapped
    for i in range(1, len(data)-1):
        cur = data.iloc[i]
        nxt = data.iloc[i+1]
        # Bullish FVG: current high < next low (gap up on close->open)
        if cur.High < nxt.Low:
            rows.append({"Index":i, "Type":"Bullish FVG", "Price": (cur.High+nxt.Low)/2,
                         "Top": nxt.Low, "Bottom": cur.High, "Strategy":"FVG"})
        # Bearish FVG: current low > next high
        if cur.Low > nxt.High:
            rows.append({"Index":i, "Type":"Bearish FVG", "Price": (cur.Low+nxt.High)/2,
                         "Top": cur.Low, "Bottom": nxt.High, "Strategy":"FVG"})
    return pd.DataFrame(rows)

def detect_patterns(data):
    rows=[]
    # Simple engulfing pattern detector
    for i in range(1, len(data)):
        prev, cur = data.iloc[i-1], data.iloc[i]
        # Bullish engulfing
        if prev.Close < prev.Open and cur.Close > cur.Open and cur.Close > prev.Open and cur.Open < prev.Close:
            rows.append({"Index": i, "Type":"Bullish Engulfing", "Price": cur.Close, "Top":cur.High, "Bottom":cur.Low, "Strategy":"Pattern"})
        # Bearish engulfing
        if prev.Close > prev.Open and cur.Close < cur.Open and cur.Open > prev.Close and cur.Close < prev.Open:
            rows.append({"Index": i, "Type":"Bearish Engulfing", "Price": cur.Close, "Top":cur.High, "Bottom":cur.Low, "Strategy":"Pattern"})
    return pd.DataFrame(rows)

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
        # normalize Index to int
        signals['Index'] = signals['Index'].astype(int)
    return signals

# ---------------------------
# Trade simulation (entry/SL/TP heuristics)
def simulate_trades(signals, df):
    trades = []
    for _, s in signals.iterrows():
        idx = int(s['Index'])
        typ = s['Type']
        price = s['Price']
        # Basic risk rules: SL/TP by % of price or use high/low of signal candle
        if "Bull" in typ:
            entry = price
            sl = entry - 0.005 * entry
            tp = entry + 0.02 * entry
            direction = "Long"
        else:
            entry = price
            sl = entry + 0.005 * entry
            tp = entry - 0.02 * entry
            direction = "Short"
        # Evaluate result by scanning next N bars (here 20) to see if hit TP/SL
        hit = None
        for j in range(idx+1, min(idx+1+20, len(df))):
            high = df.iloc[j].High
            low = df.iloc[j].Low
            if direction == "Long":
                if low <= sl:
                    hit = "SL"
                    break
                if high >= tp:
                    hit = "TP"
                    break
            else:
                if high >= sl:
                    hit = "SL"
                    break
                if low <= tp:
                    hit = "TP"
                    break
        if hit == "TP":
            profit = abs(tp - entry)
            win = 1
        elif hit == "SL":
            profit = -abs(entry - sl)
            win = 0
        else:
            # neither hit within window: mark as open/neutral (we'll use last close to compute P/L)
            last_close = df.iloc[min(idx+20, len(df)-1)].Close
            profit = (last_close - entry) if direction == "Long" else (entry - last_close)
            win = 1 if profit > 0 else 0
        trades.append({
            "Idx": idx, "Type": typ, "StrategyName": s.get("StrategyName", s.get("Strategy","")), 
            "Entry": entry, "SL": sl, "TP": tp, "Profit": profit, "Win": win
        })
    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['CumulativeProfit'] = trades_df['Profit'].cumsum()
    return trades_df

# ---------------------------
# Session controls & Run Analysis button
st.markdown("---")
run_col1, run_col2, run_col3 = st.columns([1,1,2])

with run_col1:
    run_now = st.button("▶️ Run Analysis")

with run_col2:
    st.write("")  # spacer
    clear_state = st.button("🧹 Clear Results")

with run_col3:
    # allow manual download after results
    pass

if clear_state:
    for k in ['signals_df','trades_df','summary_df']:
        if k in st.session_state:
            del st.session_state[k]
    st.experimental_rerun()

# Auto-refresh logic for live mode (only when auto_refresh is true)
if mode == "Live Analysis" and auto_refresh:
    # use st.experimental_get_query_params hack to trigger rerun every refresh_interval seconds
    st.experimental_rerun()

# Run analysis when user clicks
if run_now:
    df = st.session_state['market_df']
    signals_df = run_analysis(df, use_ob, use_fvg, use_patterns)
    st.session_state['signals_df'] = signals_df
    trades_df = simulate_trades(signals_df, df) if not signals_df.empty else pd.DataFrame()
    st.session_state['trades_df'] = trades_df

# If we have results in session_state, show them
signals_df = st.session_state.get('signals_df', pd.DataFrame())
trades_df = st.session_state.get('trades_df', pd.DataFrame())

# ---------------------------
# Left: Analysis outputs, Right: Charts
left, right = st.columns([1.1,1.9])

with left:
    st.subheader("Signals")
    if signals_df.empty:
        st.info("No signals detected yet. Click **Run Analysis** to scan the chosen range.")
    else:
        # show top signals with types and link to index/time
        display = signals_df.copy()
        display['Time'] = display['Index'].apply(lambda i: df.index[int(i)].strftime("%Y-%m-%d %H:%M"))
        display = display[['Time','Type','StrategyName','Price']]
        st.dataframe(display.sort_values(by='Time', ascending=False).reset_index(drop=True), height=300)

    st.markdown("### Performance Summary")
    if trades_df.empty:
        st.info("No trades simulated yet.")
    else:
        summary = trades_df.groupby('StrategyName').agg(
            Total_Trades=('Win','count'),
            Total_Profit=('Profit','sum'),
            Win_Rate=('Win','mean'),
            Avg_Profit=('Profit','mean')
        ).reset_index()
        summary['Win_Rate'] = (summary['Win_Rate']*100).round(2)
        summary['Total_Profit'] = summary['Total_Profit'].round(6)
        st.table(summary)

    # Alerts (latest signals)
    st.markdown("### Alerts")
    if signals_df.empty:
        st.write("—")
    else:
        latest = signals_df.tail(5).iloc[::-1]
        for _, s in latest.iterrows():
            if "Bull" in s['Type']:
                st.success(f"🚀 {s['Type']} ({s.get('StrategyName','')}) @ {s['Price']:.5f}")
            elif "Bear" in s['Type']:
                st.warning(f"⚠️ {s['Type']} ({s.get('StrategyName','')}) @ {s['Price']:.5f}")
            else:
                st.info(f"ℹ️ {s['Type']} ({s.get('StrategyName','')}) @ {s['Price']:.5f}")

    # Export buttons
    if export_csv:
        if not signals_df.empty:
            csv = signals_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Signals CSV", csv, file_name=f"{pair}_signals.csv", mime="text/csv")
        if not trades_df.empty:
            csv2 = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Trades CSV", csv2, file_name=f"{pair}_trades.csv", mime="text/csv")

with right:
    st.subheader("Programmatic Chart (candles + signals + OB/FVG zones)")

    # Build Plotly candlestick
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green', decreasing_line_color='red',
        name='Price'
    )])

    # Add OB & FVG rectangles and markers if present
    if not signals_df.empty:
        # draw zones
        for _, s in signals_df.iterrows():
            idx = int(s['Index'])
            # safe index check
            if idx < 0 or idx >= len(df):
                continue
            t0 = df.index[idx]
            # show rectangles for OB/FVG using Top/Bottom fields
            top = s.get('Top', None)
            bottom = s.get('Bottom', None)
            typ = s['Type']
            color = 'rgba(0,255,0,0.15)' if 'Bull' in typ else 'rgba(255,0,0,0.12)'
            if top is not None and bottom is not None:
                fig.add_shape(
                    type="rect",
                    x0=t0 - pd.Timedelta(minutes=1),
                    x1=t0 + pd.Timedelta(hours=12),
                    y0=bottom, y1=top,
                    fillcolor=color,
                    line=dict(width=0),
                    layer="below"
                )
            # add marker
            fig.add_trace(go.Scatter(
                x=[df.index[idx]],
                y=[s['Price']],
                mode="markers+text",
                marker=dict(color='green' if 'Bull' in typ else 'red', size=10),
                text=[s['Type']],
                textposition="top center",
                name=f"{s['Type']}"
            ))

    # Add trades equity line if exists
    if not trades_df.empty:
        # show cumulative profit on secondary y-axis
        fig.add_trace(go.Scatter(
            x=[df.index[min(int(r['Idx']), len(df)-1)] for _, r in trades_df.iterrows()],
            y=trades_df['CumulativeProfit'],
            mode="lines+markers",
            name="Equity",
            yaxis="y2"
        ))
        # add yaxis2
        fig.update_layout(
            yaxis2=dict(overlaying="y", side="right", title="Cumulative Profit", showgrid=False)
        )

    fig.update_layout(
        margin=dict(l=10, r=10, t=30, b=10),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Bottom: Extra analytics
st.markdown("---")
st.subheader("Extra Tools & Notes")

col_a, col_b, col_c = st.columns([1,1,1])
with col_a:
    st.markdown("**Multi-timeframe idea**")
    st.write("Embed smaller TF charts in TradingView widget by changing timeframe at top.")

with col_b:
    st.markdown("**Need official TradingView Charting Library?**")
    st.write("Contact TradingView for Charting Library access (paid license) for programmatic drawing in that chart.")

with col_c:
    st.markdown("**Support / Next steps**")
    st.write("- Add Telegram/email alerts  \n- Add position sizing & risk manager  \n- Add more pattern detectors")

st.caption("BrayFXTrade Analyzer — demo heuristics. Use for research/backtesting only; not trading advice.")
