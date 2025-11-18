# utils.py
import yfinance as yf
import pandas as pd

PAIR_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
    "GBPJPY": "GBPJPY=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD"
}

INTERVAL_MAP = {
    "1H": "60m",
    "4H": "60m",
    "1D": "1d",
    "15m": "15m",
    "5m": "5m"
}

def get_backend_pair(front_pair):
    return PAIR_MAP.get(front_pair, front_pair)

def fetch_data(pair, start, end, interval):
    backend_pair = get_backend_pair(pair)
    df = yf.download(backend_pair, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=['Open','High','Low','Close'])
    return df

# ---------------------------
# Signal detectors
def detect_ob(data):
    rows = []
    for i in range(1,len(data)-1):
        prev, cur, nxt = data.iloc[i-1], data.iloc[i], data.iloc[i+1]
        body_cur, body_prev, body_next = abs(cur.Close-cur.Open), abs(prev.Close-prev.Open), abs(nxt.Close-nxt.Open)
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

def run_analysis(df, use_ob=True, use_fvg=True, use_patterns=True):
    signals=[]
    if use_ob:
        signals.append(detect_ob(df))
    if use_fvg:
        signals.append(detect_fvg(df))
    if use_patterns:
        signals.append(detect_patterns(df))
    return pd.concat(signals,ignore_index=True) if signals else pd.DataFrame()
