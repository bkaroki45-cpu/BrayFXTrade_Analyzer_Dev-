import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("📊 Signals & Performance")

signals_df = st.session_state.get("signals_df", pd.DataFrame())
trades_df = st.session_state.get("trades_df", pd.DataFrame())

if signals_df.empty:
    st.info("No signals detected yet. Run Analysis from Home page.")
else:
    st.subheader("Signals")
    st.dataframe(signals_df)

if not trades_df.empty:
    st.subheader("Trades Summary")
    summary = trades_df.groupby("StrategyName").agg(
        Total_Trades=("Win","count"),
        Total_Profit=("Profit","sum"),
        Win_Rate=("Win","mean")
    ).reset_index()
    summary['Win_Rate'] = (summary['Win_Rate']*100).round(2)
    st.table(summary)

    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trades_df.index, y=trades_df['CumulativeProfit'], mode="lines+markers", name="Equity"))
    st.plotly_chart(fig, use_container_width=True)
