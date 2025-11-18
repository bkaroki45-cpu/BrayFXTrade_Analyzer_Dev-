# brayfxtrade_pro_realtime.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# ------------------------------
st.set_page_config(page_title="BrayFXTrade Analyzer — Real-Time", layout="wide")
st.title("💹 BrayFXTrade Analyzer — Real-Time Multi-Timeframe")

# ------------------------------
# Frontend Symbols
PAIR_MAP = {
    "EURUSD":"EURUSD=X", "GBPUSD":"GBPUSD=X", "USDJPY":"USDJPY=X",
    "USDCAD":"USDCAD=X", "USDCHF":"USDCHF=X", "AUDUSD":"AUDUSD=X",
    "NZDUSD":"NZDUSD=X", "GBPJPY":"GBPJPY=X", "XAUUSD":"XAUUSD=X",
    "XAGUSD":"XAGUSD=X", "BTCUSD":"BTC-USD", "ETHUSD":"ETH-USD"
}

CATEGORY_MAP = {
    "Forex":["EURUSD","GBPUSD","USDJPY","USDCAD","USDCHF","AUDUSD","NZDUSD","GBPJPY"],
    "Crypto":["BTCUSD","ETHUSD"],
    "Commodities":["XAUUSD","XAGUSD"]
}

TF_MAP = {"5m":"5m","15m":"15m","1H":"60m","4H":"240m","1D":"1d"}

# ------------------------------
# Sidebar
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Backtest","Live Analysis"])
    category = st.selectbox("Category", list(CATEGORY_MAP.keys()))
    selected_symbol = st.selectbox("Select Symbol", CATEGORY_MAP[category])
    selected_timeframes = st.multiselect("Select Timeframe(s)", ["5m","15m","1H","4H","1D"], default=["1H"])
    start_date = st.date_input("Start Date", datetime.today()-timedelta(days=90))
    end_date = st.date_input("End Date", datetime.today())
    st.subheader("Strategies")
    use_ob = st.checkbox("Order Blocks (OB)", True)
    use_fvg = st.checkbox("Fair Value Gap (FVG)", True)
    use_patterns = st.checkbox("Engulfing Pattern", True)
    st.subheader("Extras")
    show_ema = st.checkbox("Show EMA overlay", True)
    ema_period = st.number_input("EMA period", 5, 200, 34)
    auto_refresh = st.checkbox("Auto-refresh Live", False)
    refresh_interval = st.number_input("Refresh Interval (s)", 5, 300, 60)

yf_symbol = PAIR_MAP.get(selected_symbol, selected_symbol)

# ------------------------------
# Fetch data with caching
@st.cache_data(ttl=60)
def fetch_data(symbol, start, end, interval):
    try:
        df = yf.download(symbol, start=start, end=end+timedelta(days=1), interval=interval, progress=False)
        if df.empty: return pd.DataFrame()
        df = df.dropna(subset=['Open','High','Low','Close'])
        df.index = pd.to_datetime(df.index)
        return df
    except:
        return pd.DataFrame()

# ------------------------------
# Analysis Functions
def detect_ob(data):
    out=[]
    for i in range(1,len(data)-1):
        prev,cur,nxt=data.iloc[i-1],data.iloc[i],data.iloc[i+1]
        body_cur=abs(cur.Close-cur.Open)
        body_next=abs(nxt.Close-nxt.Open)
        if body_cur>body_next:
            typ="Bullish OB" if cur.Close>cur.Open else "Bearish OB"
            out.append({"Index":i,"Type":typ,"Price":cur.Close,"Strategy":"OB"})
    return pd.DataFrame(out)

def detect_fvg(data):
    out=[]
    for i in range(1,len(data)-1):
        cur,nxt=data.iloc[i],data.iloc[i+1]
        if cur.High<nxt.Low: out.append({"Index":i,"Type":"Bullish FVG","Price":(cur.High+nxt.Low)/2,"Strategy":"FVG"})
        if cur.Low>nxt.High: out.append({"Index":i,"Type":"Bearish FVG","Price":(cur.Low+nxt.High)/2,"Strategy":"FVG"})
    return pd.DataFrame(out)

def detect_patterns(data):
    out=[]
    for i in range(1,len(data)):
        prev,cur=data.iloc[i-1],data.iloc[i]
        if prev.Close<prev.Open and cur.Close>cur.Open and cur.Close>prev.Open and cur.Open<prev.Close:
            out.append({"Index":i,"Type":"Bullish Engulfing","Price":cur.Close,"Strategy":"Pattern"})
        if prev.Close>prev.Open and cur.Close<cur.Open and cur.Open>prev.Close and cur.Close<prev.Open:
            out.append({"Index":i,"Type":"Bearish Engulfing","Price":cur.Close,"Strategy":"Pattern"})
    return pd.DataFrame(out)

def simulate_trades(signals,data,lookahead=20):
    trades=[]
    for _,s in signals.iterrows():
        idx=int(s['Index']);typ=s['Type'];price=float(s['Price'])
        direction="Long" if "Bull" in typ else "Short"
        entry=price;sl=entry-0.005*entry if direction=="Long" else entry+0.005*entry
        tp=entry+0.02*entry if direction=="Long" else entry-0.02*entry
        hit=None
        for j in range(idx+1,min(idx+1+lookahead,len(data))):
            high=data.iloc[j].High;low=data.iloc[j].Low
            if direction=="Long":
                if low<=sl: hit="SL"; break
                if high>=tp: hit="TP"; break
            else:
                if high>=sl: hit="SL"; break
                if low<=tp: hit="TP"; break
        if hit=="TP": profit=abs(tp-entry);win=1
        elif hit=="SL": profit=-abs(entry-sl);win=0
        else:
            last_close=data.iloc[min(idx+lookahead,len(data)-1)].Close
            profit=(last_close-entry) if direction=="Long" else (entry-last_close)
            win=1 if profit>0 else 0
        trades.append({"Idx":idx,"Type":typ,"Strategy":s.get("Strategy",""),"Entry":entry,"SL":sl,"TP":tp,"Profit":profit,"Win":win})
    df_trades=pd.DataFrame(trades)
    if not df_trades.empty:
        df_trades['CumulativeProfit']=df_trades['Profit'].cumsum()
    return df_trades

# ------------------------------
# Run Analysis Function
def run_analysis():
    all_signals = {}
    all_trades = {}
    for tf in selected_timeframes:
        df_tf = fetch_data(yf_symbol, start_date, end_date, TF_MAP.get(tf,"60m"))
        if df_tf.empty:
            st.warning(f"No data for {selected_symbol} on {tf}")
            continue
        signals = pd.DataFrame()
        if use_ob: signals = pd.concat([signals, detect_ob(df_tf)])
        if use_fvg: signals = pd.concat([signals, detect_fvg(df_tf)])
        if use_patterns: signals = pd.concat([signals, detect_patterns(df_tf)])
        if not signals.empty: signals = signals.sort_values("Index").reset_index(drop=True)
        trades = simulate_trades(signals, df_tf) if not signals.empty else pd.DataFrame()
        all_signals[tf] = (df_tf, signals)
        all_trades[tf] = trades
    return all_signals, all_trades

# ------------------------------
# Analysis Trigger
if st.button("▶️ Run Analysis") or (mode=="Live Analysis" and auto_refresh):
    all_signals, all_trades = run_analysis()
    st.session_state['all_signals'] = all_signals
    st.session_state['all_trades'] = all_trades
    st.session_state['last_run'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# ------------------------------
# Display Results
all_signals = st.session_state.get('all_signals', {})
all_trades = st.session_state.get('all_trades', {})

for tf in selected_timeframes:
    st.markdown(f"### ⏱ Timeframe: {tf}")
    df_tf, signals = all_signals.get(tf, (pd.DataFrame(), pd.DataFrame()))
    trades = all_trades.get(tf, pd.DataFrame())
    
    if df_tf.empty:
        st.info(f"No data for {selected_symbol} on {tf}")
        continue
    
    col1,col2 = st.columns([1,2])
    with col1:
        st.subheader("Signals")
        if signals.empty: st.info("No signals detected")
        else:
            disp = signals.copy()
            disp['Time'] = disp['Index'].apply(lambda i: df_tf.index[int(i)].strftime("%Y-%m-%d %H:%M"))
            st.dataframe(disp[['Time','Type','Strategy','Price']].sort_values('Time', ascending=False))

        st.subheader("Performance")
        if not trades.empty:
            summ = trades.groupby('Strategy').agg(
                Total_Trades=('Win','count'),
                Total_Profit=('Profit','sum'),
                Win_Rate=('Win','mean')
            ).reset_index()
            summ['Win_Rate'] = (summ['Win_Rate']*100).round(2)
            summ['Total_Profit'] = summ['Total_Profit'].round(6)
            st.table(summ)

        st.subheader("Alerts")
        if not signals.empty:
            latest = signals.tail(5).iloc[::-1]
            for _,s in latest.iterrows():
                txt = f"{s['Type']} ({s.get('Strategy','')}) @ {s['Price']:.6f} - {df_tf.index[int(s['Index'])].strftime('%Y-%m-%d %H:%M')}"
                st.success(txt) if "Bull" in s['Type'] else st.warning(txt) if "Bear" in s['Type'] else st.info(txt)

    with col2:
        st.subheader("Candlestick Chart + Signals")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_tf.index, open=df_tf['Open'], high=df_tf['High'], low=df_tf['Low'], close=df_tf['Close'],
            increasing_line_color='green', decreasing_line_color='red', name='Price'
        ))
        if show_ema: fig.add_trace(go.Scatter(
            x=df_tf.index, y=df_tf['Close'].ewm(span=ema_period).mean(), name=f"EMA{ema_period}", line=dict(dash='dash')
        ))
        if not signals.empty:
            for _,s in signals.iterrows():
                idx = int(s['Index']); price = s['Price']; typ = s['Type']
                fig.add_trace(go.Scatter(
                    x=[df_tf.index[idx]], y=[price], mode='markers+text',
                    marker=dict(color='green' if 'Bull' in typ else 'red', size=10),
                    text=[typ], textposition="top center", name=typ
                ))
        if not trades.empty:
            fig.add_trace(go.Scatter(
                x=[df_tf.index[min(int(r['Idx']), len(df_tf)-1)] for _,r in trades.iterrows()],
                y=trades['CumulativeProfit'], mode='lines+markers', name='Equity',
                yaxis='y2'
            ))
            fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Cumulative Profit', showgrid=False))
        fig.update_layout(height=720, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True)

st.caption("BrayFXTrade Analyzer — Real-Time Pro. Demo heuristics only; not financial advice.")

# ------------------------------
# Auto-refresh loop in Live Analysis
if mode=="Live Analysis" and auto_refresh:
    time.sleep(refresh_interval)
    st.experimental_rerun()
