# schwabttk/visualize.py
import os
import webbrowser
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from schwabttk.price_history import load_stored, load_futures_symbols

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data", "futures")
CHARTS_DIR = os.path.join(BASE_DIR, "data", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)


# ── Themes ────────────────────────────────────────────────────────────────────
THEMES = {
    "blacknwhite": {
        "template"       : "plotly_white",
        "bg_color"       : "#F5F5F5",       # almost white, not pure
        "paper_color"    : "#F5F5F5",
        "up_color"       : "rgba(0,0,0,0)", # hollow/no fill
        "up_line"        : "#1A1A1A",       # almost black border
        "down_color"     : "#1A1A1A",       # almost black fill
        "down_line"      : "#1A1A1A",
        "vol_up"         : "#A0A0A0",       # grey volume
        "vol_down"       : "#404040",
        "line_colors"    : ["#1A1A1A", "#555555", "#888888",
                            "#333333", "#666666", "#999999"],
        "font_color"     : "#1A1A1A",
        "grid_color"     : "#DCDCDC",
    },
    "greenredoverdark": {
        "template"       : "plotly_dark",
        "bg_color"       : "#131722",       # TradingView dark
        "paper_color"    : "#131722",
        "up_color"       : "#26a69a",       # teal green fill
        "up_line"        : "#26a69a",
        "down_color"     : "#ef5350",       # red fill
        "down_line"      : "#ef5350",
        "vol_up"         : "#26a69a",
        "vol_down"       : "#ef5350",
        "line_colors"    : ["#26a69a", "#ef5350", "#2196F3",
                            "#FF9800", "#9C27B0", "#00BCD4"],
        "font_color"     : "#D1D4DC",
        "grid_color"     : "#2A2E39",
    },
    "greenredoverwhite": {
        "template"       : "plotly_white",
        "bg_color"       : "#FFFFFF",
        "paper_color"    : "#FFFFFF",
        "up_color"       : "#26a69a",
        "up_line"        : "#26a69a",
        "down_color"     : "#ef5350",
        "down_line"      : "#ef5350",
        "vol_up"         : "#26a69a",
        "vol_down"       : "#ef5350",
        "line_colors"    : ["#26a69a", "#ef5350", "#1565C0",
                            "#E65100", "#6A1B9A", "#00838F"],
        "font_color"     : "#131722",
        "grid_color"     : "#E0E0E0",
    },
}


def get_theme(name: str) -> dict:
    """Return theme dict by name (case-insensitive). Defaults to blacknwhite."""
    return THEMES.get(name.lower().replace(" ", "").replace("_", ""),
                      THEMES["blacknwhite"])


# ── Save & open ───────────────────────────────────────────────────────────────
def show(fig: go.Figure, name: str = "chart", df: pd.DataFrame = None):
    """Save chart as HTML with reliable adaptive y-axis and open in browser."""
    import json
    path = os.path.join(CHARTS_DIR, f"{name}.html")
    fig.write_html(path, include_plotlyjs="cdn")

    # Embed raw OHLC data as JSON if provided
    if df is not None and not df.empty:
        records = df[["datetime", "high", "low"]].copy()
        records["datetime"] = records["datetime"].astype(str)
        data_json = json.dumps(records.to_dict(orient="list"))
    else:
        data_json = "null"

    js = f"""
    <script>
    var OHLC_DATA = {data_json};

    window.addEventListener("load", function() {{

        function waitForPlotly(cb) {{
            var gd = document.querySelector(".plotly-graph-div");
            if (gd && gd.on && gd.layout) {{ cb(gd); }}
            else {{ setTimeout(function() {{ waitForPlotly(cb); }}, 150); }}
        }}

        waitForPlotly(function(gd) {{

            if (!OHLC_DATA) return;

            var dates = OHLC_DATA.datetime;  // array of "YYYY-MM-DD" strings
            var highs = OHLC_DATA.high;
            var lows  = OHLC_DATA.low;
            var isRescaling = false;

            function rescaleY(xStart, xEnd) {{
                if (isRescaling) return;

                var iStart, iEnd;

                // Categorical axis returns numeric indices
                if (typeof xStart === "number" && typeof xEnd === "number") {{
                    iStart = Math.max(0, Math.floor(xStart));
                    iEnd   = Math.min(dates.length - 1, Math.ceil(xEnd));

                // String date comparison fallback
                }} else if (typeof xStart === "string") {{
                    var s = xStart.substring(0, 10);
                    var e = xEnd.substring(0, 10);
                    iStart = 0;
                    iEnd   = dates.length - 1;
                    for (var k = 0; k < dates.length; k++) {{
                        if (dates[k].substring(0, 10) >= s) {{ iStart = k; break; }}
                    }}
                    for (var k = dates.length - 1; k >= 0; k--) {{
                        if (dates[k].substring(0, 10) <= e) {{ iEnd = k; break; }}
                    }}
                }} else {{
                    return;
                }}

                var visHigh = highs.slice(iStart, iEnd + 1);
                var visLow  = lows.slice(iStart,  iEnd + 1);

                if (visHigh.length === 0) return;

                var yMin = Math.min.apply(null, visLow);
                var yMax = Math.max.apply(null, visHigh);
                var pad  = (yMax - yMin) * 0.05;

                console.log("Rescale y:", yMin.toFixed(2), "to", yMax.toFixed(2),
                            "| idx", iStart, "to", iEnd,
                            "| candles", visHigh.length);

                isRescaling = true;
                Plotly.relayout(gd, {{
                    "yaxis.range[0]" : yMin - pad,
                    "yaxis.range[1]" : yMax + pad,
                    "yaxis.autorange": false
                }}).then(function() {{
                    isRescaling = false;
                }});
            }}

            function getXRange() {{
                var lay = gd.layout;
                if (lay.xaxis && lay.xaxis.range && lay.xaxis.range.length === 2) {{
                    return [lay.xaxis.range[0], lay.xaxis.range[1]];
                }}
                return null;
            }}

            gd.on("plotly_relayout", function(ev) {{
                if (isRescaling) return;

                // X drag zoom — indices
                if (ev["xaxis.range[0]"] !== undefined &&
                    ev["xaxis.range[1]"] !== undefined) {{
                    rescaleY(ev["xaxis.range[0]"], ev["xaxis.range[1]"]);
                    return;
                }}

                // Range buttons or pan — read from layout after delay
                setTimeout(function() {{
                    if (isRescaling) return;
                    var r = getXRange();
                    if (r) rescaleY(r[0], r[1]);
                }}, 80);
            }});

            // Double-click full reset
            gd.on("plotly_doubleclick", function() {{
                isRescaling = true;
                Plotly.relayout(gd, {{
                    "xaxis.autorange": true,
                    "yaxis.autorange": true
                }}).then(function() {{ isRescaling = false; }});
                return false;
            }});

            console.log("Adaptive y-axis ready. Loaded", dates.length, "candles.");
        }});
    }});
    </script>
    """

    with open(path, "a", encoding="utf-8") as f:
        f.write(js)

    webbrowser.open(f"file:///{path}")
    print(f"[INFO] Chart saved: {path}")


# ── Single symbol OHLC chart ──────────────────────────────────────────────────
def plot_ohlc(ticker: str,
              start: str = None,
              end: str = None,
              show_volume: bool = True,
              title: str = None,
              theme: str = "blacknwhite") -> go.Figure:
    """
    Plot interactive OHLC candlestick chart with volume.

    Parameters
    ----------
    ticker      : e.g. 'M2K', 'ZC', '6B'
    start       : 'YYYY-MM-DD' filter start (optional)
    end         : 'YYYY-MM-DD' filter end (optional)
    show_volume : include volume subplot
    title       : chart title (default: ticker name)
    theme       : 'blacknwhite' | 'greenredoverdark' | 'greenredoverwhite'
    """
    t   = get_theme(theme)
    df  = load_stored(ticker)

    if df.empty:
        print(f"[WARN] No data found for {ticker}")
        return go.Figure()

    if start:
        df = df[df["datetime"] >= pd.Timestamp(start)]
    if end:
        df = df[df["datetime"] <= pd.Timestamp(end)]
    df = df.reset_index(drop=True)

    # Format dates as strings — categorical axis eliminates weekend gaps
    df["dt_str"] = df["datetime"].dt.strftime("%Y-%m-%d")

    chart_title = title or f"{ticker} — Daily OHLC"
    rows        = 2 if show_volume else 1
    heights     = [0.75, 0.25] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights
    )

    # ── Candlestick ────────────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df["dt_str"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name=ticker,
            increasing=dict(
                fillcolor=t["up_color"],
                line=dict(color=t["up_line"], width=1)
            ),
            decreasing=dict(
                fillcolor=t["down_color"],
                line=dict(color=t["down_line"], width=1)
            )
        ),
        row=1, col=1
    )

    # ── Volume bars ────────────────────────────────────────────────────────
    if show_volume:
        colors = [
            t["vol_up"] if c >= o else t["vol_down"]
            for o, c in zip(df["open"], df["close"])
        ]
        fig.add_trace(
            go.Bar(
                x=df["dt_str"],
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )

    # ── Layout ─────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=chart_title,
            font=dict(size=18, color=t["font_color"])
        ),
        template=t["template"],
        paper_bgcolor=t["paper_color"],
        plot_bgcolor=t["bg_color"],
        height=700,
        showlegend=False,
        margin=dict(l=60, r=50, t=60, b=40),
        font=dict(color=t["font_color"]),
        xaxis_rangeslider_visible=False,
        xaxis = dict(
            type="category",  # ← key line
            gridcolor=t["grid_color"],
            tickangle=45,
            nticks=12,  # show ~12 labels max
        ),
        yaxis=dict(
            title="Price",
            gridcolor=t["grid_color"],
            showgrid=True,
            fixedrange=False
        ),
        yaxis2=dict(
            title="Volume",
            gridcolor=t["grid_color"],
            fixedrange=False
        ) if show_volume else {}
    )

    fig.update_xaxes(
        type="category",
        gridcolor=t["grid_color"],
        tickangle=45,
        nticks=12,
        row=1, col=1
    )

    return fig, df


# ── Multi-symbol panel ────────────────────────────────────────────────────────
def plot_multi(tickers: list = None,
               start: str = None,
               end: str = None,
               normalize: bool = True,
               theme: str = "blacknwhite") -> go.Figure:
    """
    Plot multiple futures close prices on the same chart.

    Parameters
    ----------
    tickers   : list of tickers — defaults to all available
    start     : 'YYYY-MM-DD' filter start
    end       : 'YYYY-MM-DD' filter end
    normalize : True = base-100 indexed, False = raw close
    theme     : 'blacknwhite' | 'greenredoverdark' | 'greenredoverwhite'
    """
    t       = get_theme(theme)
    tickers = tickers or list(load_futures_symbols().keys())
    title   = "Futures — Normalized Close (Base=100)" if normalize \
              else "Futures — Close Prices"
    colors  = t["line_colors"]
    fig     = go.Figure()

    for i, ticker in enumerate(tickers):
        df = load_stored(ticker)
        if df.empty:
            print(f"  [WARN] No data for {ticker} — skipping")
            continue
        if start:
            df = df[df["datetime"] >= pd.Timestamp(start)]
        if end:
            df = df[df["datetime"] <= pd.Timestamp(end)]
        if df.empty:
            continue

        y = (df["close"] / df["close"].iloc[0]) * 100 if normalize \
            else df["close"]

        fig.add_trace(go.Scatter(
            x=df["datetime"],
            y=y,
            mode="lines",
            name=ticker,
            line=dict(
                width=1.5,
                color=colors[i % len(colors)]
            )
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=t["font_color"])
        ),
        template=t["template"],
        paper_bgcolor=t["paper_color"],
        plot_bgcolor=t["bg_color"],
        height=600,
        margin=dict(l=60, r=50, t=60, b=40),
        font=dict(color=t["font_color"]),
        xaxis=dict(
            title="Date",
            gridcolor=t["grid_color"]
        ),
        yaxis=dict(
            title="Index (Base=100)" if normalize else "Price",
            gridcolor=t["grid_color"]
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=t["font_color"])
        )
    )

    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────
def plot_corr(tickers: list = None,
              start: str = None,
              end: str = None,
              theme: str = "blacknwhite") -> go.Figure:
    """
    Plot return correlation heatmap across futures.

    Parameters
    ----------
    tickers : list of tickers — defaults to all available
    start   : 'YYYY-MM-DD' filter start
    end     : 'YYYY-MM-DD' filter end
    theme   : 'blacknwhite' | 'greenredoverdark' | 'greenredoverwhite'
    """
    t       = get_theme(theme)
    tickers = tickers or list(load_futures_symbols().keys())
    closes  = {}

    for ticker in tickers:
        df = load_stored(ticker)
        if df.empty:
            continue
        if start:
            df = df[df["datetime"] >= pd.Timestamp(start)]
        if end:
            df = df[df["datetime"] <= pd.Timestamp(end)]
        if not df.empty:
            closes[ticker] = df.set_index("datetime")["close"]

    if not closes:
        print("[WARN] No data loaded")
        return go.Figure()

    prices  = pd.DataFrame(closes).dropna(how="all")
    returns = prices.pct_change().dropna(how="all")
    corr    = returns.corr()

    # colorscale: grey-based for blacknwhite, RdBu for others
    colorscale = "Greys" if theme.lower().replace("_","") == "blacknwhite" \
                 else "RdBu"

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale=colorscale,
        zmid=0,
        zmin=-1,
        zmax=1,
        text=corr.round(2).values,
        texttemplate="%{text}",
        textfont=dict(size=11, color=t["font_color"]),
        colorbar=dict(
            title="Corr",
            tickfont=dict(color=t["font_color"])
        )
    ))

    fig.update_layout(
        title=dict(
            text="Daily Return Correlations",
            font=dict(size=18, color=t["font_color"])
        ),
        template=t["template"],
        paper_bgcolor=t["paper_color"],
        plot_bgcolor=t["bg_color"],
        height=600,
        margin=dict(l=80, r=50, t=60, b=80),
        font=dict(color=t["font_color"])
    )

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Single OHLC — black & white (default)
    fig, df = plot_ohlc("M2K", start="2023-01-01", theme="blacknwhite")
    show(fig, "M2K_bnw", df=df)

    # Single OHLC — green/red over dark
    fig, df = plot_ohlc("M2K", start="2023-01-01", theme="greenredoverdark")
    show(fig, "M2K_dark", df=df)

    # Multi — green/red over white
    fig = plot_multi(
        tickers=["M2K", "MGC", "MCL", "ZN", "6B", "MBT"],
        start="2023-01-01",
        theme="greenredoverwhite"
    )
    show(fig, "multi_white")

    # Correlation heatmap
    fig = plot_corr(start="2023-01-01", theme="greenredoverdark")
    show(fig, "corr_dark")