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
      .block-container { padding-top: 1.0rem; padding-bottom: 0.15rem; }
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

# --- unified top metrics (same size + bold; only Change colors differ) ---
POS_COL = "#6ee7b7"  # light green
NEG_COL = "#fca5a5"  # light red
NEU_COL = "rgba(255,255,255,0.92)"

_last_txt = f"{last:,.2f}"
_chg_txt = f"{chg:+,.2f}" if chg is not None else "—"
_pct_txt = f"{chg_pct:+.2f}%" if chg_pct is not None else "—"
_hi_txt = f"{hi:,.2f}"
_lo_txt = f"{lo:,.2f}"

_chg_col = POS_COL if (chg is not None and chg >= 0) else (NEG_COL if chg is not None else NEU_COL)
_pct_col = POS_COL if (chg_pct is not None and chg_pct >= 0) else (NEG_COL if chg_pct is not None else NEU_COL)

st.markdown(
    """
    <style>
      .top-metrics { margin-top: 2px; margin-bottom: 2px; }
      .tm-label { font-size: 13px; opacity: 0.80; margin-bottom: 6px; }
      .tm-value { font-size: 34px; font-weight: 800; line-height: 1.08; letter-spacing: 0.2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f"""
        <div class='top-metrics'>
          <div class='tm-label'>Last</div>
          <div class='tm-value' style='color:{NEU_COL};'>{_last_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f"""
        <div class='top-metrics'>
          <div class='tm-label'>Change</div>
          <div class='tm-value' style='color:{_chg_col};'>{_chg_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f"""
        <div class='top-metrics'>
          <div class='tm-label'>Change %</div>
          <div class='tm-value' style='color:{_pct_col};'>{_pct_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f"""
        <div class='top-metrics'>
          <div class='tm-label'>Period High</div>
          <div class='tm-value' style='color:{NEU_COL};'>{_hi_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c5:
    st.markdown(
        f"""
        <div class='top-metrics'>
          <div class='tm-label'>Period Low</div>
          <div class='tm-value' style='color:{NEU_COL};'>{_lo_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

# --- Layout: chart left, stats right (more compact / no-scroll) ---
chart_col, stats_col = st.columns([2.35, 1.0], gap="large")

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
    height=600,
    hovermode="x unified",
    dragmode="pan",
    margin=dict(l=30, r=30, t=40, b=135),
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

with chart_col:
    st.subheader(pro_title)

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
        #delta { margin-left:auto; padding:6px 10px; border-radius: 8px; background: rgba(0,0,0,0.25); color: rgba(255,255,255,0.92); font-size: 12px; white-space: nowrap; opacity: 0; pointer-events:none; }
        #chart {{ width: 100%; height: 520px; }}
      </style>
    </head>
    <body>
    <div id="wrap">
      <div id="toolbar">
        <span id="pill">Drag blank space = pan • Scroll = zoom • Drag on candle/line = Δ</span>
        <button id="btnReset">Reset view</button>
        <button id="btnClear">Clear Δ</button>
        <div id="delta"></div>
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
      function clamp(v, lo, hi) {{ return Math.max(lo, Math.min(hi, v)); }}
      function getVisibleIndexRange(x0, x1) {{
        const xs = payload.x;
        let i0 = 0, i1 = xs.length - 1;
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
      function orderPointsByX(a, b) {{
        const ta = new Date(a.x).getTime();
        const tb = new Date(b.x).getTime();
        return ta <= tb ? [a, b] : [b, a];
      }}
      function setDeltaText(start, end) {{
        if (!start || !end) { deltaBox.textContent = ''; deltaBox.style.opacity = 0; return; }
  deltaBox.style.opacity = 1;
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
      }}
      function drawDelta(start, end) {{
        if (!start || !end) return;
        const ordered = orderPointsByX(start, end);
        start = ordered[0];
        end = ordered[1];
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
      const fig = payload.fig;
      fig.layout = fig.layout || {{}};
      fig.layout.dragmode = false;
      fig.layout.xaxis = fig.layout.xaxis || {{}};
      fig.layout.yaxis = fig.layout.yaxis || {{}};
      fig.layout.xaxis.autorange = true;
      fig.layout.yaxis.autorange = true;
      const config = {{
        scrollZoom: true,
        displaylogo: false,
        responsive: true,
        modeBarButtonsToRemove: ['zoom2d','pan2d','autoScale2d','resetScale2d','select2d','lasso2d','zoomIn2d','zoomOut2d'],
      }};
      Plotly.newPlot(gd, fig.data, fig.layout, config).then(() => {{
        const xr = gd._fullLayout.xaxis.range;
        if (xr && xr.length === 2) {{
          refitY(new Date(xr[0]).getTime(), new Date(xr[1]).getTime());
        }}
      }});
      gd.addEventListener('contextmenu', (e) => e.preventDefault());
      gd.on('plotly_relayout', (e) => {{
        const r0 = e['xaxis.range[0]'];
        const r1 = e['xaxis.range[1]'];
        if (!r0 || !r1) return;
        let a = new Date(r0).getTime();
        let b = new Date(r1).getTime();
        if (b < a) {{
          const tmp = a; a = b; b = tmp;
        }}
        const fullSpan = xMax - xMin;
        if (fullSpan <= 0) return;
        const span = b - a;
        const minSpan = 1000 * 60;
        const safeSpan = Math.min(Math.max(span, minSpan), fullSpan);
        if (span >= fullSpan) {{
          Plotly.relayout(gd, {{ 'xaxis.range': [new Date(xMin), new Date(xMax)] }});
          refitY(xMin, xMax);
          return;
        }}
        const aC = clamp(a, xMin, xMax - safeSpan);
        const bC = aC + safeSpan;
        if (aC !== a || bC !== b) {{
          Plotly.relayout(gd, {{ 'xaxis.range': [new Date(aC), new Date(bC)] }});
          refitY(aC, bC);
          return;
        }}
        refitY(a, b);
      }});
      let hoverPoint = null;
      let measuring = false;
      let deltaStart = null;
      let deltaEnd = null;
      let panning = false;
      let panStartX = 0;
      let panStartRange = null;
      gd.on('plotly_hover', (ev) => {{
        if (!ev || !ev.points || !ev.points.length) return;
        const p = ev.points[0];
        if (p && p.data && p.data.name === 'Volume') {{
          hoverPoint = null;
          return;
        }}
        let price = p.y;
        if ((price === undefined || price === null || Number.isNaN(price)) && (p.pointNumber !== undefined && p.pointNumber !== null)) {{
          const idx = p.pointNumber;
          if (payload.close && idx >= 0 && idx < payload.close.length) {{
            price = payload.close[idx];
          }}
        }}
        hoverPoint = {{ x: p.x, p: price }};
        if (measuring) {{
          deltaEnd = {{ x: p.x, p: price }};
          drawDelta(deltaStart, deltaEnd);
          setDeltaText(deltaStart, deltaEnd);
        }}
      }});
      gd.on('plotly_unhover', () => {{
        hoverPoint = null;
      }});
      gd.addEventListener('mousedown', (evt) => {{
        if (evt.button === 0) {{
          const xr = gd._fullLayout && gd._fullLayout.xaxis && gd._fullLayout.xaxis.range;
          if (!xr || xr.length !== 2) return;
          panning = true;
          panStartX = evt.clientX;
          panStartRange = [new Date(xr[0]).getTime(), new Date(xr[1]).getTime()];
          evt.preventDefault();
          evt.stopPropagation();
          return;
        }}
        if (evt.button === 2) {{
          if (hoverPoint) {{
            measuring = true;
            deltaStart = hoverPoint;
            deltaEnd = hoverPoint;
            drawDelta(deltaStart, deltaEnd);
            setDeltaText(deltaStart, deltaEnd);
            evt.preventDefault();
            evt.stopPropagation();
          }}
          return;
        }}
      }}, true);
      window.addEventListener('mousemove', (evt) => {{
        if (!panning || !panStartRange) return;
        const rect = gd.getBoundingClientRect();
        const width = Math.max(rect.width, 1);
        const dx = evt.clientX - panStartX;
        const span = panStartRange[1] - panStartRange[0];
        const dt = -dx / width * span;
        let a = panStartRange[0] + dt;
        let b = panStartRange[1] + dt;
        const fullSpan = xMax - xMin;
        const safeSpan = Math.min(Math.max(span, 1000 * 60), fullSpan);
        a = clamp(a, xMin, xMax - safeSpan);
        b = a + safeSpan;
        Plotly.relayout(gd, {{ 'xaxis.range': [new Date(a), new Date(b)] }});
        refitY(a, b);
      }});
      window.addEventListener('mouseup', () => {{
        if (panning) {{
          panning = false;
          panStartRange = null;
        }}
        if (measuring) {{
          measuring = false;
        }}
      }});
      btnClear.addEventListener('click', () => clearDelta());
      btnReset.addEventListener('click', () => {{
        clearDelta();
        Plotly.relayout(gd, {{
          'xaxis.range': [new Date(xMin), new Date(xMax)],
          'yaxis.autorange': true,
        }});
        setTimeout(() => {{
          refitY(xMin, xMax);
        }}, 50);
      }});
    }})();
    </script>
    </body>
    </html>
    """

    # Convert f-string escaped braces back to normal JS/CSS braces
    html = html.replace("{{", "{").replace("}}", "}")
    # Inject the Python payload JSON into the JS placeholder
    html = html.replace("__PAYLOAD__", json.dumps(payload))
    components.html(html, height=600, scrolling=False)

# -------------------------------------------------------------------
# Stats (prettier + more useful)
# -------------------------------------------------------------------
with stats_col:
    st.markdown(
        """
        <style>
          .stats-panel {
            height: 640px;                 /* match components.html height */
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            padding-top: 2px;
            overflow: hidden;
          }
          .stats-head {
            font-size: 22px;
            font-weight: 850;
            margin: 0 0 6px 0;
          }
          .stats-sub {
            font-size: 11px;
            color: rgba(255,255,255,0.70);
            margin: 0 0 14px 0;
          }
          .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            column-gap: 22px;
            row-gap: 26px;                 /* more vertical breathing room */
            flex: 1;
            align-content: space-between;  /* spread rows vertically to reduce dead space */
            padding-bottom: 10px;
          }
          .stat-k { font-size: 12px; opacity: 0.85; margin-bottom: 6px; }
          .stat-v { font-size: 22px; font-weight: 800; line-height: 1.05; }
          .stats-tip {
            margin-top: 8px;               /* move tip up slightly */
            font-size: 11px;
            opacity: 0.65;
            padding-bottom: 0px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _close = df["Close"].astype(float).dropna()
    _ret = _close.pct_change().dropna()

    ann_factor = 252 * 24 if interval == "1h" else 252

    period_return = float((_close.iloc[-1] / _close.iloc[0] - 1.0)) if len(_close) >= 2 else float("nan")
    vol_ann = float(_ret.std() * (ann_factor ** 0.5)) if len(_ret) >= 2 else float("nan")
    ret_ann = float(_ret.mean() * ann_factor) if len(_ret) >= 1 else float("nan")
    sharpe_0rf = float(ret_ann / vol_ann) if (pd.notna(ret_ann) and pd.notna(vol_ann) and vol_ann != 0) else float("nan")

    if len(_close) >= 2:
        _cummax = _close.cummax()
        _dd = _close / _cummax - 1.0
        max_dd = float(_dd.min())
    else:
        max_dd = float("nan")

    atr14 = float("nan")
    if all(c in df.columns for c in ["High", "Low", "Close"]):
        h = df["High"].astype(float)
        l = df["Low"].astype(float)
        c = df["Close"].astype(float)
        prev_c = c.shift(1)
        tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr14 = float(tr.rolling(14).mean().iloc[-1]) if tr.dropna().shape[0] >= 14 else float("nan")

    rsi14 = float("nan")
    if len(_close) >= 15:
        delta = _close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, pd.NA)
        rsi = 100 - (100 / (1 + rs))
        rsi14 = float(rsi.iloc[-1])

    rng_abs = float(hi - lo)
    rng_pct = float((hi / lo - 1.0)) if lo != 0 else float("nan")

    last_vol = float("nan")
    if "Volume" in df.columns and not df["Volume"].dropna().empty:
        last_vol = float(pd.to_numeric(df["Volume"], errors="coerce").dropna().iloc[-1])

    skew_rets = float(_ret.skew()) if len(_ret) > 5 else float("nan")
    kurt_rets = float(_ret.kurtosis()) if len(_ret) > 5 else float("nan")

    def _fmt(v, kind: str) -> str:
        if pd.isna(v):
            return "—"
        if kind == "pct2":
            return f"{v * 100:,.2f}%"
        if kind == "num2":
            return f"{v:,.2f}"
        if kind == "num1":
            return f"{v:,.1f}"
        if kind == "int":
            return f"{v:,.0f}"
        return str(v)

    tiles = [
        ("Period Return", _fmt(period_return, "pct2")),
        ("Ann. Vol", _fmt(vol_ann, "pct2")),
        ("Ann. Return", _fmt(ret_ann, "pct2")),
        ("Sharpe (0% rf)", _fmt(sharpe_0rf, "num2")),
        ("Max Drawdown", _fmt(max_dd, "pct2")),
        ("ATR(14)", _fmt(atr14, "num2")),
        ("Range", _fmt(rng_abs, "num2")),
        ("Range %", _fmt(rng_pct, "pct2")),
        ("RSI(14)", _fmt(rsi14, "num1")),
        ("Last Volume", _fmt(last_vol, "int")),
        ("Skew (rets)", _fmt(skew_rets, "num2")),
        ("Kurtosis (rets)", _fmt(kurt_rets, "num2")),
    ]

    html_stats = (
        f"<div class='stats-panel'>"
        f"  <div class='stats-head'>Stats</div>"
        f"  <div class='stats-sub'>Baked from the currently selected series (period={period_ui}, interval={interval_ui}).</div>"
        f"  <div class='stats-grid'>"
    )
    for k, v in tiles:
        html_stats += f"<div><div class='stat-k'>{k}</div><div class='stat-v'>{v}</div></div>"
    html_stats += (
        f"  </div>"
        f"  <div class='stats-tip'>Tip: metrics are computed from the displayed series (simple returns).</div>"
        f"</div>"
    )
    st.markdown(html_stats, unsafe_allow_html=True)