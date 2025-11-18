# brayfxtrade_pro_full.py
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
# Utilities
TF_TO_YF_INTERVAL = {"1H":"60m","4H":"60m","1D":"1d","15m":"15m","5m":"5m"}
ALL_PAIRS = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","GBPJPY","BTC","ETH"]

def to_localized_index(df):
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None) if df.index.tzinfo is not None else df.index
    return df

# ---------------------------
# Sidebar: settings + run button
with st.sidebar:
    st.header("BrayFXTrade Analyzer Settings")
    mode = st.selectbox("Mode", ["Backtest","Live Analysis"])
    pair = st.selectbox("Select Forex Pair", ALL_PAIRS, index=0)
    timeframe = st.selectbox("Timeframe", ["1H","4H","1D","15m","5m"], index=0)
    today = datetime.today()
    start_date = st.date_input("Start Date", today - timedelta(days=365))
    end_date = st.date_input("End Date", today)
    st.markdown("---")
    st.subheader("Strategies")
    use_ob = st.checkbox("Order Blocks (OB)", True)
    use_fvg = st.checkbox("Fair Value Gaps (FVG)", True)
    use_patterns = st.checkbox("Chart Patterns", True)
    st.markdown("---")
    st.subheader("Run / Live Options")
    auto_refresh = st.checkbox("Auto-refresh (Live only)", False)
    refresh_interval = st.number_input("Refresh interval (s)", 10, 60, 10)
    st.markdown("---")
    export_csv = st.checkbox("Show Export Buttons", True)
    st.markdown("Built by BrayFXTrade Analyzer — demo heuristics for OB/FVG and chart patterns.")
    
    # ✅ Run Analysis Button
    run_now = st.button("▶️ Run Analysis")

# ---------------------------
# Backend: add =X for Yahoo
backend_pair = pair + "=X" if pair not in ["BTC","ETH"] else pair
yf_interval = TF_TO_YF_INTERVAL.get(timeframe,"60m")

@st.cache_data(ttl=60)
def fetch_data(pair,start,end,interval):
    try:
        df = yf.download(pair,start=start,end=end+timedelta(days=1),interval=interval,progress=False)
        if df.empty: return df
        df = df.dropna(subset=['Open','High','Low','Close'])
        df = to_localized_index(df)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

raw_df = fetch_data(backend_pair,start_date,end_date,yf_interval)
if not raw_df.empty and timeframe=="4H":
    df = raw_df.resample("4H").agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
else:
    df = raw_df.copy()
if df.empty:
    st.error(f"No data for {pair} in {timeframe}. Try a different range or timeframe.")
    st.stop()

st.session_state['market_df'] = df

# ---------------------------
# Analysis functions
def detect_ob(data):
    rows=[]
    for i in range(1,len(data)-1):
        prev, cur, nxt = data.iloc[i-1], data.iloc[i], data.iloc[i+1]
        body_cur = abs(cur.Close - cur.Open)
        body_prev = abs(prev.Close - prev.Open)
        body_next = abs(nxt.Close - nxt.Open)
        if cur.Close<cur.Open and body_cur>body_prev and body_cur>body_next*0.8:
            rows.append({"Index":i,"Type":"Bearish OB","Price":cur.Close,"Top":cur.High,"Bottom":cur.Low,"Strategy":"OB"})
        if cur.Close>cur.Open and body_cur>body_prev and body_cur>body_next*0.8:
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

def run_analysis(df,ob,fvg,pat):
    signals=pd.DataFrame()
    if ob: s=detect_ob(df); s['StrategyName']='OB Strategy'; signals=pd.concat([signals,s],ignore_index=True) if not s.empty else signals
    if fvg: s=detect_fvg(df); s['StrategyName']='FVG Strategy'; signals=pd.concat([signals,s],ignore_index=True) if not s.empty else signals
    if pat: s=detect_patterns(df); s['StrategyName']='Pattern Recognition'; signals=pd.concat([signals,s],ignore_index=True) if not s.empty else signals
    if not signals.empty: signals['Index']=signals['Index'].astype(int)
    return signals

# ---------------------------
# Trade simulation
def simulate_trades(signals,df):
    trades=[]
    for _,s in signals.iterrows():
        idx=int(s['Index']); typ=s['Type']; price=s['Price']
        if "Bull" in typ: entry=price; sl=entry-0.005*entry; tp=entry+0.02*entry; direction="Long"
        else: entry=price; sl=entry+0.005*entry; tp=entry-0.02*entry; direction="Short"
        hit=None
        for j in range(idx+1,min(idx+21,len(df))):
            high,low=df.iloc[j].High,df.iloc[j].Low
            if direction=="Long":
                if low<=sl: hit="SL"; break
                if high>=tp: hit="TP"; break
            else:
                if high>=sl: hit="SL"; break
                if low<=tp: hit="TP"; break
        if hit=="TP": profit=abs(tp-entry); win=1
        elif hit=="SL": profit=-abs(entry-sl); win=0
        else: last_close=df.iloc[min(idx+20,len(df)-1)].Close; profit=(last_close-entry) if direction=="Long" else (entry-last_close); win=1 if profit>0 else 0
        trades.append({"Idx":idx,"Type":typ,"StrategyName":s.get("StrategyName",""),"Entry":entry,"SL":sl,"TP":tp,"Profit":profit,"Win":win})
    trades_df=pd.DataFrame(trades)
    if not trades_df.empty: trades_df['CumulativeProfit']=trades_df['Profit'].cumsum()
    return trades_df

# ---------------------------
# Run analysis
if run_now:
    df = st.session_state['market_df']
    signals_df = run_analysis(df,use_ob,use_fvg,use_patterns)
    st.session_state['signals_df']=signals_df
    trades_df = simulate_trades(signals_df,df) if not signals_df.empty else pd.DataFrame()
    st.session_state['trades_df']=trades_df

signals_df = st.session_state.get('signals_df',pd.DataFrame())
trades_df = st.session_state.get('trades_df',pd.DataFrame())

# ---------------------------
# Layout: Left (results) / Right (TradingView)
left,right = st.columns([1,2])

with right:
    st.subheader("TradingView Chart")
    tv_html = f"""
    <div id="tradingview_abc" style="height:700px;"></div>
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget({{
      "width":"100%","height":700,"symbol":"{pair}","interval":"{timeframe}","timezone":"Etc/UTC",
      "theme":"dark","style":"1","locale":"en","toolbar_bg":"#f1f3f6","enable_publishing":false,
      "allow_symbol_change":true,"container_id":"tradingview_abc"
    }});
    </script>
    """
    components.html(tv_html,height=720)

with left:
    st.subheader("Signals")
    if signals_df.empty: st.info("No signals detected yet. Click ▶️ Run Analysis")
    else:
        display=signals_df.copy(); display['Time']=display['Index'].apply(lambda i: df.index[int(i)].strftime("%Y-%m-%d %H:%M"))
        display=display[['Time','Type','StrategyName','Price']]
        st.dataframe(display.sort_values('Time',ascending=False).reset_index(drop=True),height=300)
    st.subheader("Performance Summary")
    if trades_df.empty: st.info("No trades simulated yet.")
    else:
        summary=trades_df.groupby('StrategyName').agg(Total_Trades=('Win','count'),Total_Profit=('Profit','sum'),
                                                     Win_Rate=('Win','mean'),Avg_Profit=('Profit','mean')).reset_index()
        summary['Win_Rate']=(summary['Win_Rate']*100).round(2)
        summary['Total_Profit']=summary['Total_Profit'].round(6)
        st.table(summary)

# Export CSV
if export_csv and not signals_df.empty:
    st.download_button("⬇️ Download Signals CSV",signals_df.to_csv(index=False).encode('utf-8'),file_name=f"{pair}_signals.csv",mime="text/csv")
if export_csv and not trades_df.empty:
    st.download_button("⬇️ Download Trades CSV",trades_df.to_csv(index=False).encode('utf-8'),file_name=f"{pair}_trades.csv",mime="text/csv")

st.caption("BrayFXTrade Analyzer — demo heuristics. For research/backtesting only.")
