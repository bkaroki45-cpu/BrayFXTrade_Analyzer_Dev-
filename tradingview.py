# TradingView.py
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="TradingView Chart", layout="wide")
st.title("💹 TradingView Chart")

pair = st.session_state.get('pair','EURUSD')
timeframe = st.session_state.get('timeframe','1H')

tv_html=f"""
<div id="tv-widget" style="height:700px;"></div>
<script src="https://s3.tradingview.com/tv.js"></script>
<script>
new TradingView.widget({{
  "width": "100%",
  "height": 700,
  "symbol": "{pair}",
  "interval": "{timeframe}",
  "timezone": "Etc/UTC",
  "theme": "dark",
  "style": "1",
  "locale": "en",
  "toolbar_bg": "#f1f3f6",
  "enable_publishing": false,
  "allow_symbol_change": true,
  "container_id": "tv-widget"
}});
</script>
"""

components.html(tv_html, height=720)
st.write("OB/FVG/Pattern signals will be drawn programmatically in next update (Plotly overlay)")
