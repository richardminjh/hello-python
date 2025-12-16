from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# --- Tighten vertical spacing globally ---
st.markdown(
    """
    <style>
      /* Make the app feel more "dashboard" (less vertical whitespace) */
      .block-container { padding-top: 1.2rem; padding-bottom: 1.0rem; }
      h1 { margin-bottom: 0.25rem !important; }
      h2, h3 { margin-top: 0.75rem !important; margin-bottom: 0.35rem !important; }
      .stCaption { margin-top: 0.15rem !important; margin-bottom: 0.35rem !important; }
      /* Slightly tighten default element gaps */
      div[data-testid="stVerticalBlock"] > div { gap: 0.55rem; }
    </style>
    """,
    unsafe_allow_html=True,
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
    # Only require Date + Close; futures often have missing Volume and we don't want to drop price rows.
    return df.dropna(subset=["Date", "Close"]) if "Close" in df.columns else df.dropna(subset=["Date"])

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

# --- colored metrics for change ---
POS_COL = "#6ee7b7"  # light green
NEG_COL = "#fca5a5"  # light red
NEU_COL = "rgba(255,255,255,0.85)"

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Last", f"{last:,.2f}")

# Change (colored)
with c2:
    _chg_txt = f"{chg:+,.2f}" if chg is not None else "—"
    _chg_col = POS_COL if (chg is not None and chg >= 0) else (NEG_COL if chg is not None else NEU_COL)
    st.markdown(
        f"""
        <div style='font-size:14px; opacity:0.85; margin-bottom:4px;'>Change</div>
        <div style='font-size:34px; font-weight:750; color:{_chg_col}; line-height:1.1;'>{_chg_txt}</div>
        """,
        unsafe_allow_html=True,
    )

# Change % (colored)
with c3:
    _pct_txt = f"{chg_pct:+.2f}%" if chg_pct is not None else "—"
    _pct_col = POS_COL if (chg_pct is not None and chg_pct >= 0) else (NEG_COL if chg_pct is not None else NEU_COL)
    st.markdown(
        f"""
        <div style='font-size:14px; opacity:0.85; margin-bottom:4px;'>Change %</div>
        <div style='font-size:34px; font-weight:750; color:{_pct_col}; line-height:1.1;'>{_pct_txt}</div>
        """,
        unsafe_allow_html=True,
    )

c4.metric("Period High", f"{hi:,.2f}")
c5.metric("Period Low", f"{lo:,.2f}")

st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

# -------------------------------------------------------------------
# Chart (THIS IS THE KEY FIX)
# -------------------------------------------------------------------
# Build a 2-row chart when volume is enabled (price on top, volume below)
if show_volume and "Volume" in df.columns:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
    )
else:
    fig = go.Figure()

if show_ohlc:
    candle = go.Candlestick(
        x=df["Date"].astype("datetime64[ns]"),
        open=df["Open"].astype(float),
        high=df["High"].astype(float),
        low=df["Low"].astype(float),
        close=df["Close"].astype(float),
        name=label,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    )
    if show_volume and "Volume" in df.columns:
        fig.add_trace(candle, row=1, col=1)
    else:
        fig.add_trace(candle)
else:
    line = go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name=label,
        line=dict(color="#42a5f5", width=2),
    )
    if show_volume and "Volume" in df.columns:
        fig.add_trace(line, row=1, col=1)
    else:
        fig.add_trace(line)

# Volume panel (pro): embedded in the same Plotly figure
# Note: Yahoo often reports very spiky volume for futures; one huge print can flatten the rest.
# We clip only for *display* (hover still shows the real volume).
if show_volume and "Volume" in df.columns:
    vol_raw = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0).astype(float)

    # Cap extreme spikes so the rest of the bars are visible
    nonzero = vol_raw[vol_raw > 0]
    if len(nonzero) >= 20:
        cap = float(nonzero.quantile(0.99))
    elif len(nonzero) > 0:
        cap = float(nonzero.max())
    else:
        cap = 0.0

    vol_plot = vol_raw.clip(upper=cap) if cap > 0 else vol_raw

    vol_colors = [
        "#26a69a" if float(c) >= float(o) else "#ef5350"
        for o, c in zip(df["Open"].astype(float), df["Close"].astype(float))
    ]

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=vol_plot,
            name="Volume",
            marker=dict(color=vol_colors),
            opacity=0.70,
            customdata=vol_raw,
            hovertemplate="%{x}<br>Volume: %{customdata:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

# Format x-axis labels based on interval
if interval == "1h":
    x_tickformat = "%b %d\n%H:%M"
    x_hoverformat = "%Y-%m-%d %H:%M"
else:
    x_tickformat = "%b %d\n%Y"
    x_hoverformat = "%Y-%m-%d"

# Apply to ALL x-axes (covers subplot xaxis2 when volume is enabled)
fig.update_xaxes(
    tickformat=x_tickformat,
    hoverformat=x_hoverformat,
    showticklabels=True,
    showspikes=True,
    spikemode="across",
    spikesnap="cursor",
    showgrid=False,
)

fig.update_layout(
    template="plotly_dark",
    uirevision=f"{ticker}-{period}-{interval}-{show_ohlc}",
    height=640,
    hovermode="x unified",
    dragmode="pan",
    margin=dict(l=30, r=30, t=40, b=90),
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

# Label subplot axes when volume is enabled
if show_volume and "Volume" in df.columns:
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1, tickformat=".2s", rangemode="tozero")
    fig.update_xaxes(title_text=None, row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

# --- Pro chart title ---
_name = label.split(" (")[0]  # e.g. "Gold"
pro_title = f"{period_ui} {_name} Futures"

st.subheader(pro_title)
st.caption("Hard range locks • Auto y-axis refit • Left-drag pan • Right-drag Δ measurement")

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

html = """
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
    #chart {{ width: 100%; height: 560px; }}
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
(function() {
  const payload = __PAYLOAD__;

  const gd = document.getElementById('chart');
  const deltaBox = document.getElementById('delta');
  const btnReset = document.getElementById('btnReset');
  const btnClear = document.getElementById('btnClear');

  const xMin = new Date(payload.xMin).getTime();
  const xMax = new Date(payload.xMax).getTime();

  // --- helpers ---
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  function getVisibleIndexRange(x0, x1) {
    // x array is sorted by time; find nearest indices
    const xs = payload.x;
    let i0 = 0, i1 = xs.length - 1;
    // linear scan is fine for <= ~10k points; keep it simple
    for (let i = 0; i < xs.length; i++) {
      const t = new Date(xs[i]).getTime();
      if (t >= x0) { i0 = i; break; }
    }
    for (let i = xs.length - 1; i >= 0; i--) {
      const t = new Date(xs[i]).getTime();
      if (t <= x1) { i1 = i; break; }
    }
    if (i1 < i0) { i0 = 0; i1 = xs.length - 1; }
    return [i0, i1];
  }

  function refitY(x0, x1) {
    const [i0, i1] = getVisibleIndexRange(x0, x1);
    let lo = Infinity, hi = -Infinity;

    if (payload.low && payload.high) {
      for (let i = i0; i <= i1; i++) {
        const L = payload.low[i];
        const H = payload.high[i];
        if (Number.isFinite(L)) lo = Math.min(lo, L);
        if (Number.isFinite(H)) hi = Math.max(hi, H);
      }
    } else {
      for (let i = i0; i <= i1; i++) {
        const C = payload.close[i];
        if (Number.isFinite(C)) { lo = Math.min(lo, C); hi = Math.max(hi, C); }
      }
    }

    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) return;
    const pad = (hi - lo) * 0.06;
    const y0 = lo - pad;
    const y1 = hi + pad;
    Plotly.relayout(gd, { 'yaxis.range': [y0, y1] });
  }

  // --- Δ measurement: direction-independent ---
  function orderPointsByX(a, b) {
    const ta = new Date(a.x).getTime();
    const tb = new Date(b.x).getTime();
    return ta <= tb ? [a, b] : [b, a];
  }

  function setDeltaText(start, end) {
    if (!start || !end) { deltaBox.textContent = 'Δ: —'; return; }
    const ordered = orderPointsByX(start, end);
    start = ordered[0];
    end = ordered[1];
    const dAbs = end.p - start.p;
    const dPct = start.p !== 0 ? (dAbs / start.p) * 100.0 : NaN;
    const s0 = new Date(start.x).toLocaleString();
    const s1 = new Date(end.x).toLocaleString();
    const absStr = (dAbs >= 0 ? '+' : '') + dAbs.toFixed(2);
    const pctStr = (dPct >= 0 ? '+' : '') + dPct.toFixed(2) + '%';
    deltaBox.textContent = 'Δ: ' + absStr + ' (' + pctStr + ') • ' + s0 + ' → ' + s1;
  }

  function drawDelta(start, end) {
    if (!start || !end) return;
    const ordered = orderPointsByX(start, end);
    start = ordered[0];
    end = ordered[1];
    const x0 = new Date(start.x);
    const x1 = new Date(end.x);
    const shapes = [
      { type: 'line', xref:'x', yref:'paper', x0:x0, x1:x0, y0:0, y1:1, line: { width: 1 } },
      { type: 'line', xref:'x', yref:'paper', x0:x1, x1:x1, y0:0, y1:1, line: { width: 1 } },
    ];
    Plotly.relayout(gd, { shapes });
  }

  function clearDelta() {
    Plotly.relayout(gd, { shapes: [] });
    setDeltaText(null, null);
    deltaStart = null;
    deltaEnd = null;
    measuring = false;
  }

  // --- create plot ---
  const fig = payload.fig;

  // Disable Plotly default drag interactions; we implement:
  // - Right-click drag = pan
  // - Left-click drag (on data) = Δ measure
  fig.layout = fig.layout || {};
  fig.layout.dragmode = false;
  fig.layout.xaxis = fig.layout.xaxis || {};
  fig.layout.yaxis = fig.layout.yaxis || {};
  fig.layout.xaxis.autorange = true;
  fig.layout.yaxis.autorange = true;

  const config = {
    scrollZoom: true,
    displaylogo: false,
    responsive: true,
    modeBarButtonsToRemove: ['zoom2d','select2d','lasso2d','zoomIn2d','zoomOut2d'],
  };

  Plotly.newPlot(gd, fig.data, fig.layout, config).then(() => {
    // Initial y-fit
    const xr = gd._fullLayout.xaxis.range;
    if (xr && xr.length === 2) {
      refitY(new Date(xr[0]).getTime(), new Date(xr[1]).getTime());
    }
  });

  // Disable context menu so right-drag is usable
  gd.addEventListener('contextmenu', (e) => e.preventDefault());

  // --- hard clamp x-range + auto y refit on pan/zoom ---
  gd.on('plotly_relayout', (e) => {
    const r0 = e['xaxis.range[0]'];
    const r1 = e['xaxis.range[1]'];
    if (!r0 || !r1) return;

    let a = new Date(r0).getTime();
    let b = new Date(r1).getTime();

    // Normalize order
    if (b < a) {
      const tmp = a; a = b; b = tmp;
    }

    const fullSpan = xMax - xMin;
    if (fullSpan <= 0) return;

    const span = b - a;
    const minSpan = 1000 * 60; // 1 minute

    // Prevent infinite zoom-out: cap span to the full available window
    const safeSpan = Math.min(Math.max(span, minSpan), fullSpan);

    // If the user tries to zoom out past the full data window, snap to full range
    if (span >= fullSpan) {
      Plotly.relayout(gd, { 'xaxis.range': [new Date(xMin), new Date(xMax)] });
      refitY(xMin, xMax);
      return;
    }

    // Clamp left edge; right edge follows from span
    const aC = clamp(a, xMin, xMax - safeSpan);
    const bC = aC + safeSpan;

    // If clamped, relayout back immediately (clean “does nothing past bounds” feel)
    if (aC !== a || bC !== b) {
      Plotly.relayout(gd, { 'xaxis.range': [new Date(aC), new Date(bC)] });
      refitY(aC, bC);
      return;
    }

    refitY(a, b);
  });

  // --- Yahoo-style Δ measurement and right-drag pan ---
  let hoverPoint = null;
  let measuring = false;
  let deltaStart = null;
  let deltaEnd = null;
  let panning = false;
  let panStartX = 0;
  let panStartRange = null;

  gd.on('plotly_hover', (ev) => {
    if (!ev || !ev.points || !ev.points.length) return;
    const p = ev.points[0];

    // Ignore volume trace for Δ measurement
    if (p && p.data && p.data.name === 'Volume') {
      hoverPoint = null;
      return;
    }

    // For candlesticks, p.y can be undefined/NaN. Use underlying Close series via pointNumber.
    let price = p.y;
    if ((price === undefined || price === null || Number.isNaN(price)) && (p.pointNumber !== undefined && p.pointNumber !== null)) {
      const idx = p.pointNumber;
      if (payload.close && idx >= 0 && idx < payload.close.length) {
        price = payload.close[idx];
      }
    }

    hoverPoint = { x: p.x, p: price };

    if (measuring) {
      deltaEnd = { x: p.x, p: price };
      drawDelta(deltaStart, deltaEnd);
      setDeltaText(deltaStart, deltaEnd);
    }
  });

  gd.on('plotly_unhover', () => {
    hoverPoint = null;
  });

  gd.addEventListener('mousedown', (evt) => {
    // Left click (button 0): PAN
    if (evt.button === 0) {
      const xr = gd._fullLayout && gd._fullLayout.xaxis && gd._fullLayout.xaxis.range;
      if (!xr || xr.length !== 2) return;
      panning = true;
      panStartX = evt.clientX;
      panStartRange = [new Date(xr[0]).getTime(), new Date(xr[1]).getTime()];
      evt.preventDefault();
      evt.stopPropagation();
      return;
    }

    // Right click (button 2): MEASURE Δ (only if hovering a data point)
    if (evt.button === 2) {
      if (hoverPoint) {
        measuring = true;
        deltaStart = hoverPoint;
        deltaEnd = hoverPoint;
        drawDelta(deltaStart, deltaEnd);
        setDeltaText(deltaStart, deltaEnd);
        evt.preventDefault();
        evt.stopPropagation();
      }
      return;
    }
  }, true);

  window.addEventListener('mousemove', (evt) => {
    if (!panning || !panStartRange) return;
    const rect = gd.getBoundingClientRect();
    const width = Math.max(rect.width, 1);

    const dx = evt.clientX - panStartX;
    const span = panStartRange[1] - panStartRange[0];

    // dx>0 means mouse moved right; we want the window to move left (earlier)
    const dt = -dx / width * span;

    let a = panStartRange[0] + dt;
    let b = panStartRange[1] + dt;

    // Clamp to bounds
    const fullSpan = xMax - xMin;
    const safeSpan = Math.min(Math.max(span, 1000 * 60), fullSpan);
    a = clamp(a, xMin, xMax - safeSpan);
    b = a + safeSpan;

    Plotly.relayout(gd, { 'xaxis.range': [new Date(a), new Date(b)] });
    refitY(a, b);
  });

  window.addEventListener('mouseup', () => {
    if (panning) {
      panning = false;
      panStartRange = null;
    }
    if (measuring) {
      measuring = false;
    }
  });

  btnClear.addEventListener('click', () => clearDelta());

  btnReset.addEventListener('click', () => {
    clearDelta();
    Plotly.relayout(gd, {
      'xaxis.range': [new Date(xMin), new Date(xMax)],
      'yaxis.autorange': true,
    });
    // Ensure y is refit to the full range
    setTimeout(() => {
      refitY(xMin, xMax);
    }, 50);
  });

})();
</script>
</body>
</html>
"""

# Convert f-string escaped braces back to normal JS/CSS braces
html = html.replace("{{", "{").replace("}}", "}")

# Inject the Python payload JSON into the JS placeholder
html = html.replace("__PAYLOAD__", json.dumps(payload))

components.html(html, height=640, scrolling=False)

# -------------------------------------------------------------------
# Stats (prettier + more useful)
# -------------------------------------------------------------------
st.markdown("""
<style>
  .stats-wrap {
    margin-top: 10px;
    padding: 14px 16px 12px 16px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.10);
    background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
  }
  .stats-title { font-size: 22px; font-weight: 850; margin: 0 0 8px 0; }
  .stats-sub { color: rgba(255,255,255,0.70); font-size: 12px; margin: 0 0 10px 0; }
  .stats-spacer { height: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    f"""
<div class="stats-wrap">
  <div class="stats-title">Stats</div>
  <div class="stats-sub">Baked from the currently selected series (period={period_ui}, interval={interval_ui}).</div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="stats-spacer"></div>', unsafe_allow_html=True)

# --- core series ---
_close = df["Close"].astype(float).dropna()
_ret = _close.pct_change().dropna()

if interval == "1h":
    # Rough trading assumption for futures: ~24 hourly bars per day
    ann_factor = 252 * 24
else:
    ann_factor = 252

# --- headline stats ---
period_return = float((_close.iloc[-1] / _close.iloc[0] - 1.0)) if len(_close) >= 2 else float("nan")
vol_ann = float(_ret.std() * (ann_factor ** 0.5)) if len(_ret) >= 2 else float("nan")
ret_ann = float(_ret.mean() * ann_factor) if len(_ret) >= 1 else float("nan")
sharpe_0rf = float(ret_ann / vol_ann) if (pd.notna(ret_ann) and pd.notna(vol_ann) and vol_ann != 0) else float("nan")

# Max drawdown
if len(_close) >= 2:
    _cummax = _close.cummax()
    _dd = _close / _cummax - 1.0
    max_dd = float(_dd.min())
else:
    max_dd = float("nan")

# ATR(14)
atr14 = float("nan")
if all(c in df.columns for c in ["High", "Low", "Close"]):
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1]) if tr.dropna().shape[0] >= 14 else float("nan")

# RSI(14)
rsi14 = float("nan")
if len(_close) >= 15:
    delta = _close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi14 = float(rsi.iloc[-1])

# Range + last volume
rng_abs = float(hi - lo)
rng_pct = float((hi / lo - 1.0)) if lo != 0 else float("nan")
last_vol = float("nan")
if "Volume" in df.columns and not df["Volume"].dropna().empty:
    last_vol = float(pd.to_numeric(df["Volume"], errors="coerce").dropna().iloc[-1])

# --- display: finance-bro friendly tiles ---
row1 = st.columns(6)
row1[0].metric("Period Return", f"{period_return * 100:,.2f}%" if pd.notna(period_return) else "—")
row1[1].metric("Ann. Vol", f"{vol_ann * 100:,.2f}%" if pd.notna(vol_ann) else "—")
row1[2].metric("Ann. Return", f"{ret_ann * 100:,.2f}%" if pd.notna(ret_ann) else "—")
row1[3].metric("Sharpe (0% rf)", f"{sharpe_0rf:,.2f}" if pd.notna(sharpe_0rf) else "—")
row1[4].metric("Max Drawdown", f"{max_dd * 100:,.2f}%" if pd.notna(max_dd) else "—")
row1[5].metric("ATR(14)", f"{atr14:,.2f}" if pd.notna(atr14) else "—")

row2 = st.columns(6)
row2[0].metric("Range", f"{rng_abs:,.2f}")
row2[1].metric("Range %", f"{rng_pct * 100:,.2f}%" if pd.notna(rng_pct) else "—")
row2[2].metric("RSI(14)", f"{rsi14:,.1f}" if pd.notna(rsi14) else "—")
row2[3].metric("Last Volume", f"{last_vol:,.0f}" if pd.notna(last_vol) else "—")
row2[4].metric("Skew (rets)", f"{float(_ret.skew()):,.2f}" if len(_ret) > 5 else "—")
row2[5].metric("Kurtosis (rets)", f"{float(_ret.kurtosis()):,.2f}" if len(_ret) > 5 else "—")

st.markdown("<div style='opacity:0.7; font-size:12px; margin-top:6px;'>Tip: metrics are computed from the displayed series (simple returns).</div>", unsafe_allow_html=True)