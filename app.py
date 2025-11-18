# brayfxtrade_pro_v2.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit.components.v1 as components

# ---------------------------
# App config
st.set_page_config(page_title="BrayFXTrade Analyzer - Pro", layout="wide")
st.title("💹 BrayFXTrade Analyzer — Pro (TradingView-style)")

# ---------------------------
# Mappings
TF_TO_YF_INTERVAL = {
    "1H": "60m",
    "4H": "60m",  # will resample to 4H
    "1D": "1d",
    "15m": "15m",
    "5m": "5m"
}

# Frontend pairs (no =X)
FRONTEND_PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD",
                  "USDCHF","NZDUSD","GBPJPY","BTC-USD","ETH-USD"]

# Map frontend pair to Yahoo symbol
def yf_symbol(pair):
    if pair in ["BTC-USD","ETH-USD"]:
        return pair
    return pair + "=X"

# ---------------------------
# Sidebar controls
with st.sidebar:
    st.header("BrayFXTrade Analyzer Settings")
    mode = st.selectbox("Mode", ["Backtest", "Live Analysis"])
    pair = st.selectbox("Select Forex Pair", FRONTEND_PAIRS, index=0)
    timeframe = st.selectbox("Timeframe", ["1H","4H","1D","15m","5m"], index=0)
    today = datetime.today()
    start_date = st.date_input("Start Date", today - timedelta(days=365))
    end_date = st.date_input("End Date", today)
    st.markdown("---")
    st.subheader("Strategies")
    use_ob = st.checkbox("Order Blocks (OB)", True)
    use_fvg = st.checkbox("Fair Value Gaps (FVG)", True)
    use_patterns = st.checkbox("Chart Patterns (Engulfing)", True)
    st.markdown("---")
    st.subheader("Run / Live Options")
    auto_refresh = st.checkbox("Auto-refresh (Live only)", False)
    refresh_interval = st.number_input("Refresh interval (s)", 10, 3600, 60)
    st.markdown("---")
    export_csv = st.checkbox("Show Export Buttons", True)
    st.markdown("Built by BrayFXTrade Analyzer — demo heuristics for OB/FVG and chart patterns.")

# ---------------------------
# TradingView Widget
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("TradingView Widget (interactive)")
    tv_html = f"""
    <div id="tradingview_abc" style="height:520px;"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget({{
      "width": "100%",
      "height": 520,
      "symbol": "{pair}",
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

with col2:
    st.subheader("Quick Stats")
    st.write(f"Mode: **{mode}**")
    st.write(f"Pair: **{pair}**")
    st.write(f"Timeframe: **{timeframe}**")
    st.write(f"Range: {start_date} → {end_date}")
    st.write("Strategies enabled:")
    st.write(f"- OB: {'✅' if use_ob else '❌'}")
    st.write(f"- FVG: {'✅' if use_fvg else '❌'}")
    st.write(f"- Patterns: {'✅' if use_patterns else '❌'}")

# ---------------------------
# Fetch data with caching
@st.cache_data(ttl=60)
def fetch_data(pair, start, end, interval):
    try:
        # Intraday <1D limited to 60 days
        if interval != "1d":
            max_start = datetime.today() - timedelta(days=60)
            if start < max_start:
                st.warning(f"Intraday data limited to last 60 days. Adjusting start date.")
                start = max_start
        df = yf.download(yf_symbol(pair), start=start, end=end+timedelta(days=1), interval=interval, progress=False)
        df = df.dropna(subset=['Open','High','Low','Close'])
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

yf_interval = TF_TO_YF_INTERVAL.get(timeframe, "60m")
raw_df = fetch_data(pair, start_date, end_date, yf_interval)

# Resample 4H if needed
if not raw_df.empty and timeframe == "4H":
    df = raw_df.resample('4H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
else:
    df = raw_df.copy()

if df.empty:
    st.error(f"No data for {pair} in {timeframe}. Try shorter range.")
    st.stop()

st.session_state['market_df'] = df

# ---------------------------
# Analysis functions
def detect_ob(data):
    rows = []
    for i in range(1,len(data)-1):
        prev, cur, nxt = data.iloc[i-1], data.iloc[i], data.iloc[i+1]
        body_cur = abs(cur.Close - cur.Open)
        if cur.Close < cur.Open:
            rows.append({"Index":i,"Type":"Bearish OB","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"OB"})
        if cur.Close > cur.Open:
            rows.append({"Index":i,"Type":"Bullish OB","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"OB"})
    return pd.DataFrame(rows)

def detect_fvg(data):
    rows=[]
    for i in range(1,len(data)-1):
        cur, nxt = data.iloc[i], data.iloc[i+1]
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

def run_analysis(df, use_ob, use_fvg, use_patterns):
    signals = pd.DataFrame()
    if use_ob: signals = pd.concat([signals, detect_ob(df)], ignore_index=True)
    if use_fvg: signals = pd.concat([signals, detect_fvg(df)], ignore_index=True)
    if use_patterns: signals = pd.concat([signals, detect_patterns(df)], ignore_index=True)
    if not signals.empty:
        signals['Index'] = signals['Index'].astype(int)
    return signals

def simulate_trades(signals, df):
    trades=[]
    for _, s in signals.iterrows():
        idx=int(s['Index']); typ=s['Type']; price=s['Price']
        if "Bull" in typ:
            entry = price; sl=entry-0.005*entry; tp=entry+0.02*entry; direction="Long"
        else:
            entry=price; sl=entry+0.005*entry; tp=entry-0.02*entry; direction="Short"
        hit=None
        for j in range(idx+1, min(idx+21, len(df))):
            high=df.iloc[j].High; low=df.iloc[j].Low
            if direction=="Long":
                if low<=sl: hit="SL"; break
                if high>=tp: hit="TP"; break
            else:
                if high>=sl: hit="SL"; break
                if low<=tp: hit="TP"; break
        if hit=="TP": profit=abs(tp-entry); win=1
        elif hit=="SL": profit=-abs(entry-sl); win=0
        else: last_close=df.iloc[min(idx+20,len(df)-1)].Close
        trades.append({"Idx":idx,"Type":typ,"StrategyName":s.get("StrategyName",s.get("Strategy","")),
                       "Entry":entry,"SL":sl,"TP":tp,"Profit":profit,"Win":win})
    trades_df=pd.DataFrame(trades)
    if not trades_df.empty: trades_df['CumulativeProfit']=trades_df['Profit'].cumsum()
    return trades_df

# ---------------------------
# Run Analysis button
st.markdown("---")
run_col1, run_col2 = st.columns([1,1])
with run_col1:
    run_now = st.button("▶️ Run Analysis")
with run_col2:
    clear_state = st.button("🧹 Clear Results")
if clear_state:
    for k in ['signals_df','trades_df']: 
        if k in st.session_state: del st.session_state[k]; st.experimental_rerun()
if run_now:
    df = st.session_state['market_df']
    signals_df = run_analysis(df,use_ob,use_fvg,use_patterns)
    st.session_state['signals_df']=signals_df
    trades_df = simulate_trades(signals_df,df) if not signals_df.empty else pd.DataFrame()
    st.session_state['trades_df']=trades_df

signals_df = st.session_state.get('signals_df', pd.DataFrame())
trades_df = st.session_state.get('trades_df', pd.DataFrame())

# ---------------------------
# Left: Signals, summary, alerts
left, right = st.columns([1,2])
with left:
    st.subheader("Signals")
    if signals_df.empty:
        st.info("No signals detected yet. Click **Run Analysis**.")
    else:
        display=signals_df.copy()
        display['Time']=display['Index'].apply(lambda i: df.index[int(i)].strftime("%Y-%m-%d %H:%M"))
        display=display[['Time','Type','StrategyName','Price']]
        st.dataframe(display.sort_values('Time',ascending=False), height=300)

    st.markdown("### Performance Summary")
    if trades_df.empty:
        st.info("No trades simulated yet.")
    else:
        summary=trades_df.groupby('StrategyName').agg(Total_Trades=('Win','count'),
                                                      Total_Profit=('Profit','sum'),
                                                      Win_Rate=('Win','mean')).reset_index()
        summary['Win_Rate']=(summary['Win_Rate']*100).round(2)
        summary['Total_Profit']=summary['Total_Profit'].round(6)
        st.table(summary)

    st.markdown("### Alerts")
    if signals_df.empty: st.write("—")
    else:
        latest = signals_df.tail(5).iloc[::-1]
        for _, s in latest.iterrows():
            if "Bull" in s['Type']: st.success(f"🚀 {s['Type']} ({s.get('StrategyName','')}) @ {s['Price']:.5f}")
            elif "Bear" in s['Type']: st.warning(f"⚠️ {s['Type']} ({s.get('StrategyName','')}) @ {s['Price']:.5f}")
            else: st.info(f"ℹ️ {s['Type']} ({s.get('StrategyName','')}) @ {s['Price']:.5f}")

    if export_csv:
        if not signals_df.empty:
            csv = signals_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Signals CSV", csv, file_name=f"{pair}_signals.csv", mime="text/csv")
        if not trades_df.empty:
            csv2 = trades_df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Trades CSV", csv2, file_name=f"{pair}_trades.csv", mime="text/csv")

# ---------------------------
# Right: Plotly chart with OB/FVG/patterns
with right:
    st.subheader("Programmatic Chart (candles + signals)")
    fig = go.Figure(data=[go.Candlestick(x=df.index,open=df['Open'],high=df['High'],low=df['Low'],close=df['Close'],
                                         increasing_line_color='green', decreasing_line_color='red', name='Price')])
    if not signals_df.empty:
        for _, s in signals_df.iterrows():
            idx=int(s['Index'])
            t0=df.index[idx]
            top=s.get('Top',None)
            bottom=s.get('Bottom',None)
            typ=s['Type']
            color='rgba(0,255,0,0.15)' if 'Bull' in typ else 'rgba(255,0,0,0.12)'
            if top and bottom:
                fig.add_shape(type="rect", x0=t0-pd.Timedelta(minutes=1), x1=t0+pd.Timedelta(hours=12),
                              y0=bottom, y1=top, fillcolor=color, line=dict(width=0), layer="below")
            fig.add_trace(go.Scatter(x=[df.index[idx]], y=[s['Price']], mode="markers+text",
                                     marker=dict(color='green' if 'Bull' in typ else 'red', size=10),
                                     text=[s['Type']], textposition="top center", name=f"{s['Type']}"))

    if not trades_df.empty:
        fig.add_trace(go.Scatter(x=[df.index[min(int(r['Idx']), len(df)-1)] for _, r in trades_df.iterrows()],
                                 y=trades_df['CumulativeProfit'], mode="lines+markers", name="Equity", yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="Cumulative Profit", showgrid=False))

    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=700,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

st.caption("BrayFXTrade Analyzer — demo heuristics. Use for research/backtesting only; not trading advice.")
