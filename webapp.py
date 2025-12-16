from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# -------------------------------------------------------------------
# Environment safety
# -------------------------------------------------------------------
os.environ["YFINANCE_NO_SCI"] = "1"
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# -------------------------------------------------------------------
# Streamlit config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Commodities Dashboard",
    layout="wide",
)

st.title("📈 Commodities Dashboard")
st.caption("Yahoo Finance (via yfinance) • Interactive charts via Plotly")

# -------------------------------------------------------------------
# Tickers
# -------------------------------------------------------------------
COMMODITIES: Dict[str, str] = {
    "Gold (GC=F)": "GC=F",
    "Silver (SI=F)": "SI=F",
    "WTI Crude (CL=F)": "CL=F",
    "Brent Crude (BZ=F)": "BZ=F",
    "Natural Gas (NG=F)": "NG=F",
    "Copper (HG=F)": "HG=F",
    "Corn (ZC=F)": "ZC=F",
    "Wheat (ZW=F)": "ZW=F",
    "Soybeans (ZS=F)": "ZS=F",
    "Coffee (KC=F)": "KC=F",
    "Cocoa (CC=F)": "CC=F",
    "Cotton (CT=F)": "CT=F",
}

PERIOD_MAP = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
}

INTERVAL_MAP = {
    "Daily": "1d",
    "Hourly": "1h",
}

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")

    label = st.selectbox("Commodity", list(COMMODITIES.keys()))
    period_ui = st.selectbox("Period", list(PERIOD_MAP.keys()), index=2)
    interval_ui = st.selectbox("Interval", list(INTERVAL_MAP.keys()), index=0)

    show_ohlc = st.toggle("Candlesticks (OHLC)", value=True)
    show_volume = st.toggle("Show Volume", value=False)

ticker = COMMODITIES[label]
period = PERIOD_MAP[period_ui]
interval = INTERVAL_MAP[interval_ui]

# -------------------------------------------------------------------
# Data fetch
# -------------------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    return df.dropna()

with st.spinner(f"Loading {label}…"):
    df = fetch_history(ticker, period, interval)

if df.empty:
    st.error("No data returned from Yahoo Finance.")
    st.stop()

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
close = df["Close"].dropna()

last = float(close.iloc[-1])
prev = float(close.iloc[-2]) if len(close) > 1 else None

if prev is not None and prev != 0:
    chg = last - prev
    chg_pct = (chg / prev) * 100
else:
    chg = None
    chg_pct = None

hi = float(df["High"].max())
lo = float(df["Low"].min())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last", f"{last:,.2f}")
c2.metric("Change", f"{chg:,.2f}" if chg is not None else "—")
c3.metric("Change %", f"{chg_pct:.2f}%" if chg_pct is not None else "—")
c4.metric("Period High", f"{hi:,.2f}")
c5.metric("Period Low", f"{lo:,.2f}")

st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

# -------------------------------------------------------------------
# Chart (THIS IS THE KEY FIX)
# -------------------------------------------------------------------
fig = go.Figure()

if show_ohlc:
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=label,
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )
    )
else:
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name=label,
            line=dict(color="#42a5f5", width=2),
        )
    )

fig.update_layout(
    template="plotly_dark",
    height=600,
    hovermode="x unified",
    dragmode="zoom",
    margin=dict(l=30, r=30, t=40, b=40),
    xaxis=dict(
        title="Date",
        type="date",
        showgrid=False,
        rangeslider=dict(visible=False),
        tickformat="%b %d\n%Y",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    ),
    yaxis=dict(
        title="Price",
        side="right",
        showgrid=False,
        tickformat=",.2f",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
    ),
)

st.plotly_chart(
    fig,
    width="stretch",
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
    },
)

# -------------------------------------------------------------------
# Volume + Stats
# -------------------------------------------------------------------
colA, colB = st.columns([1, 1])

with colA:
    if show_volume and "Volume" in df.columns:
        st.subheader("Volume")
        st.bar_chart(df.set_index("Date")["Volume"])

with colB:
    st.subheader("Stats")
    stats = df["Close"].describe().to_frame(name="Close")
    st.dataframe(stats, width="stretch")