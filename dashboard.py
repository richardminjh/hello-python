from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotext as plt
import yfinance as yf
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Quiet yfinance chatter (it can print 'possibly delisted' messages)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# -----------------------------
# Config: common commodity tickers (Yahoo Finance)
# Notes:
# - "=F" are futures contracts (front month continuous in many cases)
# - Some markets may have delayed data depending on region/time.
# -----------------------------
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

# Default: show ALL commodities in the dashboard
DEFAULT_WATCHLIST = list(COMMODITIES.keys())


@dataclass
class Quote:
    label: str
    ticker: str
    last: Optional[float]
    prev_close: Optional[float]
    change: Optional[float]
    change_pct: Optional[float]
    day_high: Optional[float]
    day_low: Optional[float]
    open_: Optional[float]
    year_high: Optional[float]
    year_low: Optional[float]
    volume: Optional[float]
    currency: Optional[str]
    fetched_at_utc: str
    error: Optional[str] = None


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def fetch_quotes_bulk(watch: List[Tuple[str, str]]) -> List[Quote]:
    """Fetch quotes for a watchlist using ONE Yahoo Finance download call.

    This is much faster and more reliable than calling Ticker().fast_info for each symbol.
    We download 1y daily data and compute:
      - last close
      - previous close
      - day open/high/low/volume (last row)
      - 52w high/low from the 1y window

    If Yahoo returns no data (sometimes happens temporarily for futures tickers),
    the Quote will contain an error message rather than crashing.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    labels = [lbl for lbl, _ in watch]
    tickers = [tkr for _, tkr in watch]

    try:
        hist = yf.download(
            tickers=tickers,
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=True,
        )
    except Exception as e:
        return [
            Quote(
                label=labels[i],
                ticker=tickers[i],
                last=None,
                prev_close=None,
                change=None,
                change_pct=None,
                day_high=None,
                day_low=None,
                open_=None,
                year_high=None,
                year_low=None,
                volume=None,
                currency=None,
                fetched_at_utc=now,
                error=f"Download failed: {e}",
            )
            for i in range(len(watch))
        ]

    if hist is None or hist.empty:
        return [
            Quote(
                label=labels[i],
                ticker=tickers[i],
                last=None,
                prev_close=None,
                change=None,
                change_pct=None,
                day_high=None,
                day_low=None,
                open_=None,
                year_high=None,
                year_low=None,
                volume=None,
                currency=None,
                fetched_at_utc=now,
                error="No price data returned (Yahoo/yfinance sometimes throttles futures tickers). Try refresh.",
            )
            for i in range(len(watch))
        ]

    out: List[Quote] = []

    # yfinance returns either single-index columns (one ticker) or MultiIndex columns (multiple tickers)
    multi = isinstance(hist.columns, pd.MultiIndex)

    for label, ticker in watch:
        try:
            if multi:
                # yfinance MultiIndex can be (ticker, field) OR (field, ticker). Handle both.
                try:
                    df = hist.xs(ticker, axis=1, level=0, drop_level=True)
                except KeyError:
                    df = hist.xs(ticker, axis=1, level=1, drop_level=True)
                df = df.dropna(how="all")
            else:
                # Single ticker case: columns are just OHLCV
                df = hist.dropna(how="all")

            if df is None or df.empty or "Close" not in df.columns:
                out.append(
                    Quote(
                        label=label,
                        ticker=ticker,
                        last=None,
                        prev_close=None,
                        change=None,
                        change_pct=None,
                        day_high=None,
                        day_low=None,
                        open_=None,
                        year_high=None,
                        year_low=None,
                        volume=None,
                        currency="USD",
                        fetched_at_utc=now,
                        error="No data for ticker (may be temporarily unavailable).",
                    )
                )
                continue

            close = df["Close"].dropna()
            if len(close) == 0:
                raise ValueError("Close series empty")

            last = _safe_float(close.iloc[-1])
            prev_close = _safe_float(close.iloc[-2]) if len(close) >= 2 else None

            open_ = _safe_float(df["Open"].iloc[-1]) if "Open" in df.columns else None
            day_high = _safe_float(df["High"].iloc[-1]) if "High" in df.columns else None
            day_low = _safe_float(df["Low"].iloc[-1]) if "Low" in df.columns else None
            volume = _safe_float(df["Volume"].iloc[-1]) if "Volume" in df.columns else None

            year_high = _safe_float(df["High"].max()) if "High" in df.columns else None
            year_low = _safe_float(df["Low"].min()) if "Low" in df.columns else None

            change = None
            change_pct = None
            if last is not None and prev_close not in (None, 0.0):
                change = last - prev_close
                change_pct = (change / prev_close) * 100.0

            out.append(
                Quote(
                    label=label,
                    ticker=ticker,
                    last=last,
                    prev_close=prev_close,
                    change=change,
                    change_pct=change_pct,
                    day_high=day_high,
                    day_low=day_low,
                    open_=open_,
                    year_high=year_high,
                    year_low=year_low,
                    volume=volume,
                    currency="USD",
                    fetched_at_utc=now,
                )
            )
        except Exception as e:
            out.append(
                Quote(
                    label=label,
                    ticker=ticker,
                    last=None,
                    prev_close=None,
                    change=None,
                    change_pct=None,
                    day_high=None,
                    day_low=None,
                    open_=None,
                    year_high=None,
                    year_low=None,
                    volume=None,
                    currency="USD",
                    fetched_at_utc=now,
                    error=f"Parse error: {e}",
                )
            )

    return out


def fetch_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.Series:
    """Returns Close series for charting.

    Yahoo/yfinance can intermittently return empty frames for futures tickers.
    We try a few fallbacks before giving up.
    """

    def _close_from_hist(hist: pd.DataFrame) -> pd.Series:
        if hist is None or hist.empty:
            return pd.Series(dtype=float)
        if isinstance(hist.columns, pd.MultiIndex):
            # Sometimes download returns multi-index columns; grab the first "Close"
            close = hist["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            return close.dropna()
        if "Close" not in hist.columns:
            return pd.Series(dtype=float)
        return hist["Close"].dropna()

    # Attempt 1: bulk downloader (fast)
    try:
        hist = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            actions=False,
            repair=True,
            progress=False,
            threads=True,
        )
        close = _close_from_hist(hist)
        if not close.empty:
            return close
    except Exception:
        pass

    # Attempt 2: Ticker().history() fallback (often succeeds when download() is flaky)
    try:
        t = yf.Ticker(ticker)
        hist2 = t.history(period=period, interval=interval, auto_adjust=False, actions=False)
        close2 = _close_from_hist(hist2)
        if not close2.empty:
            return close2
    except Exception:
        pass

    # Attempt 3: shorter period fallback (sometimes Yahoo serves recent data but not long windows)
    try:
        t = yf.Ticker(ticker)
        hist3 = t.history(period="1mo", interval=interval, auto_adjust=False, actions=False)
        close3 = _close_from_hist(hist3)
        if not close3.empty:
            return close3
    except Exception:
        pass

    return pd.Series(dtype=float)


def fmt_num(x: Optional[float], decimals: int = 2) -> str:
    if x is None:
        return "—"
    return f"{x:,.{decimals}f}"


def fmt_vol(x: Optional[float]) -> str:
    if x is None:
        return "—"
    if x >= 1e9:
        return f"{x/1e9:.2f}B"
    if x >= 1e6:
        return f"{x/1e6:.2f}M"
    if x >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.0f}"


def color_change(x: Optional[float]) -> str:
    if x is None:
        return "white"
    return "green" if x > 0 else ("red" if x < 0 else "yellow")


def render_table(quotes: List[Quote]) -> None:
    table = Table(
        title="Commodities Dashboard (Yahoo Finance via yfinance)",
        box=box.SIMPLE_HEAVY,
        header_style="bold",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Instrument", style="bold")
    table.add_column("Ticker", style="dim")
    table.add_column("Last", justify="right")
    table.add_column("Chg", justify="right")
    table.add_column("Chg%", justify="right")
    table.add_column("Open", justify="right")
    table.add_column("Low", justify="right")
    table.add_column("High", justify="right")
    table.add_column("52W Low", justify="right")
    table.add_column("52W High", justify="right")
    table.add_column("Vol", justify="right")

    for i, q in enumerate(quotes, start=1):
        chg_style = color_change(q.change)
        chg_pct_style = color_change(q.change_pct)

        last_str = fmt_num(q.last)
        if q.currency:
            last_str = f"{last_str} {q.currency}"

        table.add_row(
            str(i),
            q.label,
            q.ticker,
            last_str,
            Text(fmt_num(q.change), style=chg_style),
            Text((fmt_num(q.change_pct) + "%") if q.change_pct is not None else "—", style=chg_pct_style),
            fmt_num(q.open_),
            fmt_num(q.day_low),
            fmt_num(q.day_high),
            fmt_num(q.year_low),
            fmt_num(q.year_high),
            fmt_vol(q.volume),
        )

    fetched_times = sorted({q.fetched_at_utc for q in quotes})
    subtitle = fetched_times[-1] if fetched_times else ""
    console.print(
        Panel.fit(
            f"[bold]Last refresh:[/bold] {subtitle}    |    [dim]Select a row number to chart[/dim]"
        )
    )
    console.print(table)

    any_errors = [q for q in quotes if q.error]
    if any_errors:
        console.print(Panel.fit("[yellow]Note:[/yellow] Some tickers returned errors. Try refresh."))
        for q in any_errors:
            console.print(f"[dim]- {q.label} ({q.ticker}):[/dim] {q.error}")


def render_chart(label: str, ticker: str, period: str = "6mo") -> None:
    close = fetch_history(ticker, period=period, interval="1d")
    if close.empty:
        console.print(Panel.fit(f"[red]No history returned for {label} ({ticker}).[/red]"))
        return

    # plotext can't plot pandas Timestamp/DatetimeIndex directly.
    # Convert dates to strings and values to plain floats.
    try:
        x = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(close.index).to_pydatetime()]
    except Exception:
        x = [str(d) for d in close.index]
    y = [float(v) for v in close.values]

    plt.clear_figure()
    plt.date_form("Y-m-d")
    plt.title(f"{label} — Close ({period})")
    plt.xlabel("Date")
    plt.ylabel("Price")

    plt.plot(x, y)
    if len(x) > 20:
        step = max(1, len(x) // 10)
        plt.xticks(x[::step])

    plt.canvas_color(None)
    plt.axes_color(None)
    plt.ticks_color(None)
    plt.show()


def choose_watchlist() -> List[Tuple[str, str]]:
    # Default: show everything (sorted for consistent row numbers)
    names = sorted(DEFAULT_WATCHLIST)
    return [(name, COMMODITIES[name]) for name in names]


def main() -> None:
    watch = choose_watchlist()

    while True:
        clear_screen()
        quotes = fetch_quotes_bulk(watch)
        render_table(quotes)

        console.print(
            "\n[bold]Commands:[/bold]  [cyan]r[/cyan]=refresh  [cyan]c[/cyan]=change chart period  [cyan]q[/cyan]=quit"
        )
        console.print("[dim]Or type a row number (e.g., 2) to open an interactive chart.[/dim]\n")

        cmd = input("> ").strip().lower()

        if cmd in ("q", "quit", "exit"):
            break

        if cmd in ("r", ""):
            continue

        if cmd == "c":
            console.print("\nChoose chart period: 1) 1mo  2) 3mo  3) 6mo  4) 1y  5) 2y")
            p = input("period> ").strip()
            period_map = {"1": "1mo", "2": "3mo", "3": "6mo", "4": "1y", "5": "2y"}
            chosen = period_map.get(p, "6mo")

            console.print("\nType a row number to chart with that period.")
            row = input("row> ").strip()
            if row.isdigit():
                idx = int(row) - 1
                if 0 <= idx < len(watch):
                    label, ticker = watch[idx]
                    clear_screen()
                    render_chart(label, ticker, period=chosen)
                    input("\nPress Enter to return to dashboard...")
            continue

        if cmd.isdigit():
            idx = int(cmd) - 1
            if 0 <= idx < len(watch):
                label, ticker = watch[idx]
                clear_screen()
                render_chart(label, ticker, period="6mo")
                input("\nPress Enter to return to dashboard...")
            continue

        console.print("[yellow]Unknown command.[/yellow]")
        time.sleep(0.8)


if __name__ == "__main__":
    main()