from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

import json
import streamlit.components.v1 as components

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

def _normalize_yf_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance output across versions/edge-cases.

    - Flattens MultiIndex columns like ('Open','GC=F') -> 'Open'
    - Resets index to a 'Date' column (handles 'Date'/'Datetime'/'index')
    - Forces OHLCV columns to numeric
    """
    df = raw.copy()

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]) for c in df.columns]

    # Bring the datetime index into a column
    df = df.reset_index()

    # Identify the datetime column name after reset_index
    if "Date" in df.columns:
        dt_col = "Date"
    elif "Datetime" in df.columns:
        dt_col = "Datetime"
    elif "index" in df.columns:
        dt_col = "index"
    else:
        # Fall back: pick the first column that looks datetime-like
        dt_col = df.columns[0]

    df = df.rename(columns={dt_col: "Date"})

    # Ensure timezone-naive datetimes for Plotly candlesticks
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.tz_localize(None)

    # Force OHLCV columns to numeric if they exist
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no datetime or no close
    df = df.dropna(subset=["Date"])
    if "Close" in df.columns:
        df = df.dropna(subset=["Close"])

    return df

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
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = _normalize_yf_df(raw)
    return df.dropna(how="any")

with st.spinner(f"Loading {label}…"):
    df = fetch_history(ticker, period, interval)

if df.empty:
    st.error("No data returned from Yahoo Finance.")
    st.stop()

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
close = df["Close"].dropna()

last = float(close.iloc[-1].item())
prev = float(close.iloc[-2].item()) if len(close) > 1 else None

if prev is not None and prev != 0:
    chg = last - prev
    chg_pct = (chg / prev) * 100
else:
    chg = None
    chg_pct = None

hi = float(df["High"].max().item())
lo = float(df["Low"].min().item())

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
            x=df["Date"].astype("datetime64[ns]"),
            open=df["Open"].astype(float),
            high=df["High"].astype(float),
            low=df["Low"].astype(float),
            close=df["Close"].astype(float),
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

# Format x-axis labels based on interval
if interval == "1h":
    x_tickformat = "%b %d\n%H:%M"
    x_hoverformat = "%Y-%m-%d %H:%M"
else:
    x_tickformat = "%b %d\n%Y"
    x_hoverformat = "%Y-%m-%d"

fig.update_layout(
    template="plotly_dark",
    uirevision=f"{ticker}-{period}-{interval}-{show_ohlc}",
    height=600,
    hovermode="x unified",
    dragmode="pan",
    margin=dict(l=30, r=30, t=40, b=40),
    xaxis=dict(
        title="Date",
        type="date",
        autorange=True,
        showgrid=False,
        rangeslider=dict(visible=False),
        tickformat=x_tickformat,
        hoverformat=x_hoverformat,
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
        autorange=True,
        fixedrange=False,
    ),
)

# Basic chart (stable Streamlit render)
st.plotly_chart(
    fig,
    width="stretch",
    config={
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "zoom2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
        ],
    },
)

st.divider()
st.subheader("Pro Chart (Yahoo-style)")
st.caption("Hard range locks • Auto y-axis refit • Click-drag Δ measurement (drag on blank space pans)")

# Serialize Plotly figure for the JS component
fig_json = fig.to_json()

# Use data bounds for hard clamping
x_min = df["Date"].min()
x_max = df["Date"].max()

# Build a clean array of OHLC for y-refit (falls back to Close)
series_close = df["Close"].astype(float).tolist()
series_open = df["Open"].astype(float).tolist() if "Open" in df.columns else None
series_high = df["High"].astype(float).tolist() if "High" in df.columns else None
series_low = df["Low"].astype(float).tolist() if "Low" in df.columns else None
series_x = [d.isoformat() for d in pd.to_datetime(df["Date"]).tolist()]

# Pass everything into JS safely
payload = {
    "fig": json.loads(fig_json),
    "xMin": pd.to_datetime(x_min).isoformat(),
    "xMax": pd.to_datetime(x_max).isoformat(),
    "x": series_x,
    "close": series_close,
    "open": series_open,
    "high": series_high,
    "low": series_low,
    "interval": interval,
    "label": label,
}

html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    html, body {{ margin:0; padding:0; background: transparent; }}
    #wrap {{ width: 100%; }}
    #toolbar {{ display:flex; gap:8px; align-items:center; margin: 6px 0 10px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; }}
    #pill {{ padding:6px 10px; border-radius: 999px; background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.9); font-size: 12px; }}
    button {{ cursor:pointer; border: 1px solid rgba(255,255,255,0.15); background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.9); padding: 6px 10px; border-radius: 8px; font-size: 12px; }}
    button:hover {{ background: rgba(255,255,255,0.10); }}
    #delta {{ margin-left:auto; padding:6px 10px; border-radius: 8px; background: rgba(0,0,0,0.25); color: rgba(255,255,255,0.92); font-size: 12px; white-space: nowrap; }}
    #chart {{ width: 100%; height: 620px; }}
  </style>
</head>
<body>
<div id="wrap">
  <div id="toolbar">
    <span id="pill">Drag blank space = pan • Scroll = zoom • Drag on candle/line = Δ</span>
    <button id="btnReset">Reset view</button>
    <button id="btnClear">Clear Δ</button>
    <div id="delta">Δ: —</div>
  </div>
  <div id="chart"></div>
</div>

<script>
(function() {{
  const payload = {json.dumps(payload)};

  const gd = document.getElementById('chart');
  const deltaBox = document.getElementById('delta');
  const btnReset = document.getElementById('btnReset');
  const btnClear = document.getElementById('btnClear');

  const xMin = new Date(payload.xMin).getTime();
  const xMax = new Date(payload.xMax).getTime();

  // --- helpers ---
  function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}

  function getVisibleIndexRange(x0, x1) {{
    // x array is sorted by time; find nearest indices
    const xs = payload.x;
    let i0 = 0, i1 = xs.length - 1;
    // linear scan is fine for <= ~10k points; keep it simple
    for (let i = 0; i < xs.length; i++) {{
      const t = new Date(xs[i]).getTime();
      if (t >= x0) {{ i0 = i; break; }}
    }}
    for (let i = xs.length - 1; i >= 0; i--) {{
      const t = new Date(xs[i]).getTime();
      if (t <= x1) {{ i1 = i; break; }}
    }}
    if (i1 < i0) {{ i0 = 0; i1 = xs.length - 1; }}
    return [i0, i1];
  }}

  function refitY(x0, x1) {{
    const [i0, i1] = getVisibleIndexRange(x0, x1);
    let lo = Infinity, hi = -Infinity;

    if (payload.low && payload.high) {{
      for (let i = i0; i <= i1; i++) {{
        const L = payload.low[i];
        const H = payload.high[i];
        if (Number.isFinite(L)) lo = Math.min(lo, L);
        if (Number.isFinite(H)) hi = Math.max(hi, H);
      }}
    }} else {{
      for (let i = i0; i <= i1; i++) {{
        const C = payload.close[i];
        if (Number.isFinite(C)) {{ lo = Math.min(lo, C); hi = Math.max(hi, C); }}
      }}
    }}

    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return;
    const pad = (hi - lo) * 0.06;
    const y0 = lo - pad;
    const y1 = hi + pad;
    Plotly.relayout(gd, {{ 'yaxis.range': [y0, y1] }});
  }}

  function setDeltaText(start, end) {{
    if (!start || !end) {{ deltaBox.textContent = 'Δ: —'; return; }}
    const dAbs = end.p - start.p;
    const dPct = start.p !== 0 ? (dAbs / start.p) * 100.0 : NaN;
    const s0 = new Date(start.x).toLocaleString();
    const s1 = new Date(end.x).toLocaleString();
    const absStr = (dAbs >= 0 ? '+' : '') + dAbs.toFixed(2);
    const pctStr = (dPct >= 0 ? '+' : '') + dPct.toFixed(2) + '%';
    deltaBox.textContent = `Δ: ${absStr} (${pctStr}) • ${s0} → ${s1}`;
  }}

  function drawDelta(start, end) {{
    if (!start || !end) return;
    const x0 = new Date(start.x);
    const x1 = new Date(end.x);
    const shapes = [
      {{ type: 'line', xref:'x', yref:'paper', x0:x0, x1:x0, y0:0, y1:1, line: {{ width: 1 }} }},
      {{ type: 'line', xref:'x', yref:'paper', x0:x1, x1:x1, y0:0, y1:1, line: {{ width: 1 }} }},
    ];
    Plotly.relayout(gd, {{ shapes }});
  }}

  function clearDelta() {{
    Plotly.relayout(gd, {{ shapes: [] }});
    setDeltaText(null, null);
    deltaStart = null;
    deltaEnd = null;
    measuring = false;
  }}

  // --- create plot ---
  const fig = payload.fig;

  // Ensure we start in pan mode; our JS will decide when to measure
  fig.layout = fig.layout || {{}};
  fig.layout.dragmode = 'pan';
  fig.layout.xaxis = fig.layout.xaxis || {{}};
  fig.layout.yaxis = fig.layout.yaxis || {{}};
  fig.layout.xaxis.autorange = true;
  fig.layout.yaxis.autorange = true;

  const config = {{
    scrollZoom: true,
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['zoom2d','select2d','lasso2d','zoomIn2d','zoomOut2d'],
  }};

  Plotly.newPlot(gd, fig.data, fig.layout, config).then(() => {{
    // Initial y-fit
    const xr = gd._fullLayout.xaxis.range;
    if (xr && xr.length === 2) {{
      refitY(new Date(xr[0]).getTime(), new Date(xr[1]).getTime());
    }}
  }});

  // --- hard clamp x-range + auto y refit on pan/zoom ---
  gd.on('plotly_relayout', (e) => {{
    const r0 = e['xaxis.range[0]'];
    const r1 = e['xaxis.range[1]'];
    if (!r0 || !r1) return;

    let a = new Date(r0).getTime();
    let b = new Date(r1).getTime();

    const span = b - a;
    const minSpan = 1000 * 60; // 1 minute
    const safeSpan = Math.max(span, minSpan);

    const aC = clamp(a, xMin, xMax - safeSpan);
    const bC = aC + safeSpan;

    // If clamped, relayout back immediately (this creates the “does not move past edge” feel)
    if (aC !== a || bC !== b) {{
      Plotly.relayout(gd, {{ 'xaxis.range': [new Date(aC), new Date(bC)] }});
      refitY(aC, bC);
      return;
    }}

    refitY(a, b);
  }});

  // --- Yahoo-style Δ measurement ---
  let hoverPoint = null;
  let measuring = false;
  let deltaStart = null;
  let deltaEnd = null;

  gd.on('plotly_hover', (ev) => {{
    if (!ev || !ev.points || !ev.points.length) return;
    const p = ev.points[0];
    // p.x can be Date/string, p.y is value
    hoverPoint = {{ x: p.x, p: p.y }};
    if (measuring) {{
      deltaEnd = {{ x: p.x, p: p.y }};
      drawDelta(deltaStart, deltaEnd);
      setDeltaText(deltaStart, deltaEnd);
    }}
  }});

  gd.on('plotly_unhover', () => {{
    hoverPoint = null;
  }});

  // If mouse goes down while we are currently hovering a data point, start measuring.
  // Otherwise allow pan drag.
  gd.addEventListener('mousedown', (evt) => {{
    if (hoverPoint) {{
      measuring = true;
      deltaStart = hoverPoint;
      deltaEnd = hoverPoint;
      drawDelta(deltaStart, deltaEnd);
      setDeltaText(deltaStart, deltaEnd);
      // Prevent Plotly from starting pan when measuring
      evt.preventDefault();
      evt.stopPropagation();
    }}
  }}, true);

  window.addEventListener('mouseup', () => {{
    if (measuring) {{
      measuring = false;
    }}
  }});

  btnClear.addEventListener('click', () => clearDelta());

  btnReset.addEventListener('click', () => {{
    clearDelta();
    Plotly.relayout(gd, {{ 'xaxis.autorange': true, 'yaxis.autorange': true }});
    // After autorange, refit y once the new x-range is known
    setTimeout(() => {{
      const xr = gd._fullLayout.xaxis.range;
      if (xr && xr.length === 2) {{
        refitY(new Date(xr[0]).getTime(), new Date(xr[1]).getTime());
      }}
    }}, 50);
  }});

}})();
</script>
</body>
</html>
"""

components.html(html, height=690, scrolling=False)

# -------------------------------------------------------------------
# Volume + Stats
# -------------------------------------------------------------------
colA, colB = st.columns([1, 1])

with colA:
    if show_volume and "Volume" in df.columns:
        st.subheader("Volume")
        vol = df[["Date", "Volume"]].dropna()
        vol = vol.set_index("Date")["Volume"]
        st.bar_chart(vol)

with colB:
    st.subheader("Stats")

    desc = df["Close"].describe()

    if isinstance(desc, pd.Series):
        stats = desc.to_frame(name="Close")
    else:
        stats = desc.rename(columns={"Close": "Close"})

    st.dataframe(stats, width="stretch")