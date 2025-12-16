from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# --- Tickers (same idea as your terminal app) ---
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
    "15 min": "15m",
}

st.set_page_config(page_title="Commodities Dashboard", layout="wide")

st.title("📈 Commodities Dashboard")
st.caption("Data: Yahoo Finance (via yfinance) • Interactive charts via Plotly")

# Quiet noisy yfinance logs in the UI/server logs
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


@st.cache_data(ttl=60)
def fetch_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        out = pd.DataFrame()
        out.attrs["error"] = str(e)
        return out

    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(how="all")
    return df


def metrics_from_df(df: pd.DataFrame) -> dict:
    if df.empty or "Close" not in df.columns:
        return {"last": None, "chg": None, "chg_pct": None, "hi": None, "lo": None, "vol": None}

    close = df["Close"].dropna()
    if len(close) == 0:
        return {"last": None, "chg": None, "chg_pct": None, "hi": None, "lo": None, "vol": None}

    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else None

    chg = (last - prev) if prev is not None else None
    chg_pct = (chg / prev * 100.0) if (prev not in (None, 0.0) and chg is not None) else None

    hi = float(df["High"].max()) if "High" in df.columns else None
    lo = float(df["Low"].min()) if "Low" in df.columns else None
    vol = float(df["Volume"].iloc[-1]) if "Volume" in df.columns and not df["Volume"].empty else None

    return {"last": last, "chg": chg, "chg_pct": chg_pct, "hi": hi, "lo": lo, "vol": vol}


# ---- Sidebar controls ----
with st.sidebar:
    st.header("Controls")
    label = st.selectbox("Commodity", list(COMMODITIES.keys()), index=0)
    period_ui = st.selectbox("Period", list(PERIOD_MAP.keys()), index=2)  # 6M
    interval_ui = st.selectbox("Interval", list(INTERVAL_MAP.keys()), index=0)  # Daily

    show_ohlc = st.toggle("Candlesticks (OHLC)", value=True)
    show_volume = st.toggle("Show Volume", value=False)

ticker = COMMODITIES[label]
period = PERIOD_MAP[period_ui]
interval = INTERVAL_MAP[interval_ui]

# Yahoo limitation: intraday intervals (e.g. 15m/1h) only support recent windows.
if interval in {"15m", "1h"} and period not in {"1mo", "3mo"}:
    st.warning("Intraday intervals only work for recent windows on Yahoo. Switching Period to 3M.")
    period_ui = "3M"
    period = PERIOD_MAP[period_ui]

st.info(f"Loading {label} ({ticker}) • period={period_ui} • interval={interval_ui}")
with st.spinner("Fetching data from Yahoo Finance..."):
    df = fetch_history(ticker, period, interval)

if df.empty:
    err = df.attrs.get("error") if hasattr(df, "attrs") else None
    if err:
        st.error(
            "Data fetch failed. This is usually a dependency or Yahoo limitation.\n\n"
            f"**Details:** {err}\n\n"
            "Try: switch Interval to Daily, switch Period to 1M/3M, or refresh. "
            "If deploying: ensure `requirements.txt` includes `scipy` if you want repair-mode."
        )
    else:
        st.error(
            f"No data returned for {label} ({ticker}).\n\n"
            "This can happen with futures tickers on Yahoo (temporary throttling/outages).\n"
            "Try: (1) switch Interval to Daily, (2) switch Period to 1M/3M, or (3) refresh."
        )
    st.stop()

m = metrics_from_df(df)

# ---- Top metrics row ----
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Last", "—" if m["last"] is None else f"{m['last']:,.2f}")
c2.metric("Change", "—" if m["chg"] is None else f"{m['chg']:,.2f}")
c3.metric("Change %", "—" if m["chg_pct"] is None else f"{m['chg_pct']:.2f}%")
c4.metric("Period High", "—" if m["hi"] is None else f"{m['hi']:,.2f}")
c5.metric("Period Low", "—" if m["lo"] is None else f"{m['lo']:,.2f}")

st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

# ---- Chart ----
fig = go.Figure()

if show_ohlc and {"Open", "High", "Low", "Close"}.issubset(df.columns):
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )
else:
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close"))

fig.update_layout(
    height=550,
    margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
)

st.plotly_chart(fig, use_container_width=True)

# ---- Optional volume + stats table ----
colA, colB = st.columns([1, 1])

with colA:
    if show_volume and "Volume" in df.columns:
        st.subheader("Volume")
        st.bar_chart(df["Volume"].fillna(0))

with colB:
    st.subheader("Stats")
    stats = pd.DataFrame(
        {
            "Close": df["Close"].describe() if "Close" in df.columns else pd.Series(dtype=float),
        }
    )
    st.dataframe(stats, use_container_width=True)