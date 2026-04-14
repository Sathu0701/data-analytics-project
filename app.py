"""
AlphaPulse - Investment Risk & Volatility Monitor
Premium Plotly Dash Dashboard
Brand: AlphaPulse | Client: Boutique Investment Firm
"""
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from data_fetcher import fetch_portfolio_data, fetch_volume_data, PORTFOLIO_TICKERS
from analytics import (
    compute_log_returns,
    compute_daily_pct_returns,
    compute_var,
    compute_var_table,
    run_monte_carlo,
    run_monte_carlo_paths,
    monte_carlo_summary,
    compute_correlation_matrix,
    compute_rolling_volatility,
    compute_portfolio_variance,
)

# ──────────────────────────────────────────────────────────────
# COLOUR PALETTE  (AlphaPulse brand)
# ──────────────────────────────────────────────────────────────
BG_DEEP      = "#040d1a"
BG_CARD      = "#0c1729"
BG_CARD2     = "#0f1e35"
ACCENT_GOLD  = "#f0b429"
ACCENT_TEAL  = "#00d4b4"
ACCENT_RED   = "#ff4d6d"
ACCENT_BLUE  = "#3d9cf0"
TEXT_PRIMARY = "#e8f1ff"
TEXT_MUTED   = "#6b85a8"
BORDER_COL   = "#1c3054"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY),
        xaxis=dict(gridcolor=BORDER_COL, linecolor=BORDER_COL, zerolinecolor=BORDER_COL),
        yaxis=dict(gridcolor=BORDER_COL, linecolor=BORDER_COL, zerolinecolor=BORDER_COL),
        coloraxis_colorbar=dict(bgcolor=BG_CARD, tickfont=dict(color=TEXT_PRIMARY)),
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRIMARY)),
        hoverlabel=dict(bgcolor=BG_CARD2, font_color=TEXT_PRIMARY, bordercolor=ACCENT_GOLD),
    )
)

CHART_COLORS = [
    "#3d9cf0", "#f0b429", "#00d4b4", "#ff4d6d",
    "#b06dff", "#ff8c42", "#39d353", "#ff6eb4",
    "#53d8fb", "#ffd700",
]

# ──────────────────────────────────────────────────────────────
# LOAD DATA ON STARTUP
# ──────────────────────────────────────────────────────────────
print("[*] Loading portfolio data ...")
try:
    PRICES_DF  = fetch_portfolio_data()
    VOLUMES_DF = fetch_volume_data()
    LOG_RETS   = compute_log_returns(PRICES_DF[PORTFOLIO_TICKERS])
    PCT_RETS   = compute_daily_pct_returns(PRICES_DF[PORTFOLIO_TICKERS])
    CORR_MAT   = compute_correlation_matrix(LOG_RETS)
    ROLL_VOL   = compute_rolling_volatility(LOG_RETS)
    VAR_TABLE  = compute_var_table(LOG_RETS)
    DATA_OK    = True
    print("[OK] Data loaded successfully.")
except Exception as exc:
    print(f"[ERR] Data load failed: {exc}")
    DATA_OK = False

AVAILABLE_TICKERS = PORTFOLIO_TICKERS if DATA_OK else []

# ──────────────────────────────────────────────────────────────
# DASH APP
# ──────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SLATE,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
)
app.title = "AlphaPulse | Investment Risk & Volatility Monitor"

# ──────────────────────────────────────────────────────────────
# HELPER: styled card
# ──────────────────────────────────────────────────────────────
def glass_card(*children, extra_style=None):
    style = {
        "background": f"linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD2} 100%)",
        "border": f"1px solid {BORDER_COL}",
        "borderRadius": "12px",
        "padding": "18px 20px",
        "marginBottom": "16px",
        "boxShadow": "0 4px 24px rgba(0,0,0,0.4)",
        **(extra_style or {}),
    }
    return html.Div(children, style=style)

def metric_badge(label, value, color=ACCENT_GOLD):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": TEXT_MUTED, "textTransform": "uppercase",
                               "letterSpacing": "1px", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "color": color}),
    ], style={"textAlign": "center", "padding": "10px 18px"})

# ──────────────────────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────────────────────
app.layout = html.Div(style={"backgroundColor": BG_DEEP, "minHeight": "100vh",
                              "fontFamily": "Inter, sans-serif", "color": TEXT_PRIMARY}, children=[

    # ── HEADER ──────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div("α", style={"fontSize": "34px", "fontWeight": "900", "color": ACCENT_GOLD,
                                  "background": f"linear-gradient(135deg, {ACCENT_GOLD}, #ff8c42)",
                                  "-webkit-background-clip": "text", "-webkit-text-fill-color": "transparent",
                                  "marginRight": "10px"}),
            html.Div([
                html.Div("AlphaPulse", style={"fontSize": "22px", "fontWeight": "800",
                                               "letterSpacing": "2px", "color": TEXT_PRIMARY}),
                html.Div("Investment Risk & Volatility Monitor",
                         style={"fontSize": "11px", "color": TEXT_MUTED, "letterSpacing": "1.5px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),

        # Header KPI strip
        html.Div(id="header-kpis", style={"display": "flex", "gap": "8px", "alignItems": "center"}),

        html.Div([
            html.Div("LIVE", style={"fontSize": "10px", "color": "#00d4b4", "fontWeight": "700",
                                     "letterSpacing": "2px"}),
            html.Div(style={"width": "8px", "height": "8px", "borderRadius": "50%",
                             "backgroundColor": "#00d4b4", "animation": "pulse 1.5s infinite",
                             "marginLeft": "6px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "padding": "16px 32px",
        "background": f"linear-gradient(90deg, {BG_CARD} 0%, {BG_CARD2} 100%)",
        "borderBottom": f"1px solid {BORDER_COL}",
        "position": "sticky", "top": "0", "zIndex": "900",
        "boxShadow": "0 2px 20px rgba(0,0,0,0.5)",
    }),

    # ── CONTROLS ────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("Select Stocks", style={"fontSize": "12px", "color": TEXT_MUTED,
                                                "letterSpacing": "1px", "marginBottom": "6px"}),
            dcc.Dropdown(
                id="ticker-dropdown",
                options=[{"label": t, "value": t} for t in AVAILABLE_TICKERS],
                value=AVAILABLE_TICKERS[:4],
                multi=True,
                placeholder="Choose tickers …",
                style={"backgroundColor": BG_CARD2, "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COL}"},
            ),
        ], style={"flex": "2", "minWidth": "300px"}),

        html.Div([
            html.Label("MC Simulations", style={"fontSize": "12px", "color": TEXT_MUTED,
                                                  "letterSpacing": "1px", "marginBottom": "6px"}),
            dcc.Slider(
                id="mc-slider",
                min=10_000, max=50_000, step=10_000, value=10_000,
                marks={10000: "10K", 20000: "20K", 30000: "30K", 40000: "40K", 50000: "50K"},
            ),
        ], style={"flex": "1.5", "minWidth": "260px"}),

        html.Div([
            html.Label("MC Base Ticker", style={"fontSize": "12px", "color": TEXT_MUTED,
                                                  "letterSpacing": "1px", "marginBottom": "6px"}),
            dcc.Dropdown(
                id="mc-ticker",
                options=[{"label": t, "value": t} for t in AVAILABLE_TICKERS],
                value=AVAILABLE_TICKERS[0] if AVAILABLE_TICKERS else None,
                style={"backgroundColor": BG_CARD2, "color": TEXT_PRIMARY, "border": f"1px solid {BORDER_COL}"},
            ),
        ], style={"flex": "1", "minWidth": "150px"}),

        dbc.Button(
            "▶  Run Analysis", id="run-btn", color="warning",
            style={"fontWeight": "700", "letterSpacing": "1px", "padding": "10px 24px",
                   "borderRadius": "8px", "alignSelf": "flex-end", "backgroundColor": ACCENT_GOLD,
                   "border": "none", "color": BG_DEEP},
        ),
    ], style={
        "display": "flex", "flexWrap": "wrap", "gap": "24px", "alignItems": "flex-end",
        "padding": "20px 32px", "backgroundColor": BG_CARD, "borderBottom": f"1px solid {BORDER_COL}",
    }),

    # ── MAIN CONTENT ────────────────────────────────────────
    html.Div([

        # Row 1: Core Metrics
        html.Div([
            html.Div("📈  Core Market Metrics",
                     style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_MUTED,
                             "letterSpacing": "2px", "textTransform": "uppercase",
                             "marginBottom": "12px", "paddingLeft": "4px"}),
            html.Div([
                # Price chart
                glass_card(
                    html.Div("Stock Price", style={"fontSize": "12px", "color": TEXT_MUTED,
                                                    "letterSpacing": "1px", "marginBottom": "8px"}),
                    dcc.Graph(id="price-chart", config={"displayModeBar": False},
                              style={"height": "280px"}),
                    extra_style={"flex": "1.5"},
                ),
                # Volume chart
                glass_card(
                    html.Div("Trading Volume", style={"fontSize": "12px", "color": TEXT_MUTED,
                                                       "letterSpacing": "1px", "marginBottom": "8px"}),
                    dcc.Graph(id="volume-chart", config={"displayModeBar": False},
                              style={"height": "280px"}),
                    extra_style={"flex": "1"},
                ),
                # Returns chart
                glass_card(
                    html.Div("Daily % Returns", style={"fontSize": "12px", "color": TEXT_MUTED,
                                                        "letterSpacing": "1px", "marginBottom": "8px"}),
                    dcc.Graph(id="returns-chart", config={"displayModeBar": False},
                              style={"height": "280px"}),
                    extra_style={"flex": "1"},
                ),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        ]),

        # Row 2: Deep Analytics
        html.Div([
            html.Div("🔬  Deep Production Analytics",
                     style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_MUTED,
                             "letterSpacing": "2px", "textTransform": "uppercase",
                             "marginBottom": "12px", "paddingLeft": "4px"}),
            html.Div([
                # Monte Carlo
                glass_card(
                    html.Div([
                        html.Div("Monte Carlo Simulation", style={"fontSize": "12px", "color": TEXT_MUTED,
                                                                    "letterSpacing": "1px"}),
                        html.Div(id="mc-stats", style={"display": "flex", "gap": "16px",
                                                        "marginTop": "4px"}),
                    ], style={"marginBottom": "8px"}),
                    dcc.Graph(id="mc-chart", config={"displayModeBar": False},
                              style={"height": "300px"}),
                    extra_style={"flex": "1.2"},
                ),
                # Correlation Heatmap
                glass_card(
                    html.Div("Correlation Heatmap", style={"fontSize": "12px", "color": TEXT_MUTED,
                                                            "letterSpacing": "1px", "marginBottom": "8px"}),
                    dcc.Graph(id="corr-chart", config={"displayModeBar": False},
                              style={"height": "300px"}),
                    extra_style={"flex": "1"},
                ),
                # Rolling Volatility
                glass_card(
                    html.Div("30-Day Rolling Volatility (Annualised)",
                             style={"fontSize": "12px", "color": TEXT_MUTED,
                                    "letterSpacing": "1px", "marginBottom": "8px"}),
                    dcc.Graph(id="vol-chart", config={"displayModeBar": False},
                              style={"height": "300px"}),
                    extra_style={"flex": "1"},
                ),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        ]),

        # Row 3: VaR Panel
        html.Div(id="var-panel"),

    ], style={"padding": "24px 32px"}),

    # Footer
    html.Div([
        html.Span("AlphaPulse", style={"color": ACCENT_GOLD, "fontWeight": "700"}),
        html.Span("  |  Investment Risk & Volatility Monitor  |  Data: Yahoo Finance  |  "
                  "Analytics: NumPy · SciPy  |  Built for Zaalima Development pvt.ltd",
                  style={"color": TEXT_MUTED}),
    ], style={"textAlign": "center", "padding": "16px", "fontSize": "11px",
               "borderTop": f"1px solid {BORDER_COL}"}),

])


# ──────────────────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────────────────

def apply_theme(fig):
    """Apply the AlphaPulse dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(family="Inter, sans-serif", color=TEXT_PRIMARY, size=11),
        xaxis=dict(gridcolor=BORDER_COL, linecolor=BORDER_COL, zerolinecolor="#1c3054",
                   tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor=BORDER_COL, linecolor=BORDER_COL, zerolinecolor="#1c3054",
                   tickfont=dict(color=TEXT_MUTED)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_PRIMARY, size=10)),
        hoverlabel=dict(bgcolor=BG_CARD2, font_color=TEXT_PRIMARY, bordercolor=ACCENT_GOLD),
        margin=dict(l=50, r=20, t=36, b=40),
    )
    return fig


@app.callback(
    Output("price-chart", "figure"),
    Output("volume-chart", "figure"),
    Output("returns-chart", "figure"),
    Output("header-kpis", "children"),
    Input("run-btn", "n_clicks"),
    State("ticker-dropdown", "value"),
    prevent_initial_call=False,
)
def update_core_charts(n_clicks, tickers):
    tickers = tickers or AVAILABLE_TICKERS[:4]
    if not DATA_OK or not tickers:
        empty = go.Figure()
        return empty, empty, empty, []

    # ── Price chart ─────────────────────────────────────────
    fig_price = go.Figure()
    for i, t in enumerate(tickers):
        if t not in PRICES_DF.columns:
            continue
        series = PRICES_DF[t].dropna()
        # Normalise to 100 for easier comparison
        normed = series / series.iloc[0] * 100
        fig_price.add_trace(go.Scatter(
            x=normed.index, y=normed.values,
            name=t, line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"<b>{t}</b><br>Date: %{{x|%b %d %Y}}<br>Normalised: %{{y:.1f}}<extra></extra>",
        ))
    fig_price.update_layout(title="Price (Normalised, base=100)", title_font_size=13)
    apply_theme(fig_price)

    # ── Volume chart (last 60 days, one ticker) ─────────────
    fig_vol = go.Figure()
    lead = tickers[0] if tickers[0] in VOLUMES_DF.columns else None
    if lead:
        vol_data = VOLUMES_DF[lead].dropna().tail(60)
        fig_vol.add_trace(go.Bar(
            x=vol_data.index, y=vol_data.values,
            name=f"{lead} Volume",
            marker_color=ACCENT_TEAL,
            opacity=0.85,
            hovertemplate=f"<b>{lead}</b><br>%{{x|%b %d}}<br>Volume: %{{y:,.0f}}<extra></extra>",
        ))
    fig_vol.update_layout(title=f"Trading Volume — {tickers[0]} (60d)", title_font_size=13,
                           bargap=0.2)
    apply_theme(fig_vol)

    # ── Daily % Returns ────────────────────────────────────
    fig_ret = go.Figure()
    for i, t in enumerate(tickers):
        if t not in PCT_RETS.columns:
            continue
        ret = PCT_RETS[t].dropna().tail(120) * 100
        fig_ret.add_trace(go.Scatter(
            x=ret.index, y=ret.values,
            name=t, mode="lines",
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1.2),
            hovertemplate=f"<b>{t}</b><br>%{{x|%b %d}}: %{{y:.2f}}%<extra></extra>",
        ))
    fig_ret.add_hline(y=0, line_dash="dot", line_color=TEXT_MUTED, line_width=1)
    fig_ret.update_layout(title="Daily % Returns (120d)", title_font_size=13)
    apply_theme(fig_ret)

    # ── Header KPIs ─────────────────────────────────────────
    kpis = []
    for t in tickers[:4]:
        if t not in LOG_RETS.columns:
            continue
        var95 = compute_var(LOG_RETS[t].dropna()) * 100
        latest_price = PRICES_DF[t].dropna().iloc[-1] if t in PRICES_DF.columns else 0
        kpis.append(metric_badge(f"{t} VaR 95%", f"{var95:.2f}%", ACCENT_RED))

    return fig_price, fig_vol, fig_ret, kpis


@app.callback(
    Output("mc-chart", "figure"),
    Output("mc-stats", "children"),
    Input("run-btn", "n_clicks"),
    State("mc-ticker", "value"),
    State("mc-slider", "value"),
    prevent_initial_call=False,
)
def update_monte_carlo(n_clicks, ticker, n_sims):
    if not DATA_OK or not ticker or ticker not in LOG_RETS.columns:
        return go.Figure(), []

    n_sims = n_sims or 10_000
    np.random.seed(42)
    returns = LOG_RETS[ticker].dropna()
    terminal_vals = run_monte_carlo(returns, n_simulations=n_sims, n_days=252)
    summary = monte_carlo_summary(terminal_vals)

    # Histogram with bell-curve overlay
    fig = go.Figure()

    # Main histogram
    fig.add_trace(go.Histogram(
        x=terminal_vals,
        nbinsx=120,
        name="Portfolio Value",
        marker_color=ACCENT_BLUE,
        opacity=0.75,
        histnorm="probability density",
        hovertemplate="Value: %{x:.1f}<br>Density: %{y:.4f}<extra></extra>",
    ))

    # KDE overlay (normal approximation)
    from scipy.stats import norm
    x_range = np.linspace(terminal_vals.min(), terminal_vals.max(), 400)
    kde_y = norm.pdf(x_range, summary["mean"], summary["std"])
    fig.add_trace(go.Scatter(
        x=x_range, y=kde_y,
        mode="lines", name="Normal Fit",
        line=dict(color=ACCENT_GOLD, width=2.5),
        hoverinfo="skip",
    ))

    # VaR 95% line
    fig.add_vline(
        x=summary["var_95"], line_dash="dash", line_color=ACCENT_RED, line_width=1.8,
        annotation_text="VaR 95%", annotation_font_color=ACCENT_RED,
        annotation_position="top left",
    )
    # Median line
    fig.add_vline(
        x=summary["median"], line_dash="dot", line_color=ACCENT_TEAL, line_width=1.5,
        annotation_text="Median", annotation_font_color=ACCENT_TEAL,
        annotation_position="top right",
    )

    skew_label = "Right Skew ▶" if summary["skewness"] > 0 else "◀ Left Skew"
    skew_color = ACCENT_GOLD if summary["skewness"] > 0 else ACCENT_RED

    fig.update_layout(
        title=f"Monte Carlo Distribution — {ticker} ({n_sims:,} runs, 1-year horizon)",
        title_font_size=13,
        showlegend=True,
        bargap=0.02,
    )
    apply_theme(fig)
    fig.update_xaxes(title_text="Portfolio Value ($)")
    fig.update_yaxes(title_text="Probability Density")

    # Stat chips
    stats_children = [
        html.Span(f"Mean: {summary['mean']:.1f}",
                  style={"fontSize": "11px", "color": ACCENT_TEAL, "fontWeight": "600",
                          "background": "rgba(0,212,180,0.1)", "padding": "3px 8px", "borderRadius": "4px"}),
        html.Span(f"σ: {summary['std']:.2f}",
                  style={"fontSize": "11px", "color": TEXT_MUTED, "background": "rgba(255,255,255,0.05)",
                          "padding": "3px 8px", "borderRadius": "4px"}),
        html.Span(skew_label,
                  style={"fontSize": "11px", "color": skew_color, "fontWeight": "600",
                          "background": f"rgba(240,180,41,0.1)", "padding": "3px 8px", "borderRadius": "4px"}),
        html.Span(f"Kurt: {summary['kurtosis']:.2f}",
                  style={"fontSize": "11px", "color": TEXT_MUTED, "background": "rgba(255,255,255,0.05)",
                          "padding": "3px 8px", "borderRadius": "4px"}),
    ]
    return fig, stats_children


@app.callback(
    Output("corr-chart", "figure"),
    Input("run-btn", "n_clicks"),
    State("ticker-dropdown", "value"),
    prevent_initial_call=False,
)
def update_correlation(n_clicks, tickers):
    tickers = tickers or AVAILABLE_TICKERS[:4]
    if not DATA_OK or not tickers or len(tickers) < 2:
        return go.Figure()

    valid = [t for t in tickers if t in LOG_RETS.columns]
    if len(valid) < 2:
        return go.Figure()

    corr = compute_correlation_matrix(LOG_RETS[valid])
    z = corr.values
    text_annot = [[f"{v:.2f}" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=valid, y=valid,
        text=text_annot,
        texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale=[
            [0.0, "#ff4d6d"],
            [0.3, "#7a1f3a"],
            [0.5, BG_CARD2],
            [0.7, "#1a4a6e"],
            [1.0, "#3d9cf0"],
        ],
        zmin=-1, zmax=1,
        hovertemplate="%{y} × %{x}<br>Corr: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Corr", tickfont=dict(color=TEXT_PRIMARY, size=10),
                      bgcolor=BG_CARD, outlinecolor=BORDER_COL),
    ))
    fig.update_layout(title="Pairwise Correlation Matrix", title_font_size=13)
    apply_theme(fig)
    fig.update_xaxes(tickangle=-30)
    return fig


@app.callback(
    Output("vol-chart", "figure"),
    Input("run-btn", "n_clicks"),
    State("ticker-dropdown", "value"),
    prevent_initial_call=False,
)
def update_volatility(n_clicks, tickers):
    tickers = tickers or AVAILABLE_TICKERS[:4]
    if not DATA_OK or not tickers:
        return go.Figure()

    fig = go.Figure()
    for i, t in enumerate(tickers):
        if t not in ROLL_VOL.columns:
            continue
        vol_series = ROLL_VOL[t].dropna() * 100   # as percentage
        fig.add_trace(go.Scatter(
            x=vol_series.index, y=vol_series.values,
            name=t,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            fill="tozeroy" if i == 0 else None,
            fillcolor=f"rgba({int(CHART_COLORS[i % len(CHART_COLORS)][1:3],16)},"
                      f"{int(CHART_COLORS[i % len(CHART_COLORS)][3:5],16)},"
                      f"{int(CHART_COLORS[i % len(CHART_COLORS)][5:7],16)},0.08)" if i == 0 else None,
            hovertemplate=f"<b>{t}</b><br>%{{x|%b %d}}<br>Ann. Vol: %{{y:.1f}}%<extra></extra>",
        ))
    fig.update_layout(title="30-Day Rolling Annualised Volatility (%)", title_font_size=13)
    apply_theme(fig)
    fig.update_yaxes(title_text="Volatility (%)")
    return fig


@app.callback(
    Output("var-panel", "children"),
    Input("run-btn", "n_clicks"),
    State("ticker-dropdown", "value"),
    prevent_initial_call=False,
)
def update_var_panel(n_clicks, tickers):
    tickers = tickers or AVAILABLE_TICKERS[:4]
    if not DATA_OK:
        return []

    valid = [t for t in tickers if t in LOG_RETS.columns]
    if not valid:
        return []

    var_df = compute_var_table(LOG_RETS[valid])

    cards = []
    for _, row in var_df.iterrows():
        # Colour code by risk
        col95 = ACCENT_RED if row["VaR_95%"] > 2.5 else ACCENT_GOLD
        col99 = ACCENT_RED if row["VaR_99%"] > 4.0 else ACCENT_GOLD

        cards.append(html.Div([
            html.Div(row["Ticker"], style={"fontSize": "15px", "fontWeight": "700",
                                            "color": ACCENT_BLUE, "marginBottom": "10px"}),
            html.Div([
                html.Div([
                    html.Div("VaR 95%", style={"fontSize": "10px", "color": TEXT_MUTED,
                                                "letterSpacing": "1px", "marginBottom": "4px"}),
                    html.Div(f"{row['VaR_95%']:.3f}%", style={"fontSize": "20px", "fontWeight": "700",
                                                               "color": col95}),
                ]),
                html.Div(style={"width": "1px", "height": "40px", "backgroundColor": BORDER_COL}),
                html.Div([
                    html.Div("VaR 99%", style={"fontSize": "10px", "color": TEXT_MUTED,
                                                "letterSpacing": "1px", "marginBottom": "4px"}),
                    html.Div(f"{row['VaR_99%']:.3f}%", style={"fontSize": "20px", "fontWeight": "700",
                                                               "color": col99}),
                ]),
            ], style={"display": "flex", "gap": "20px", "alignItems": "center"}),
        ], style={
            "background": f"linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD2} 100%)",
            "border": f"1px solid {BORDER_COL}",
            "borderTop": f"3px solid {ACCENT_RED}",
            "borderRadius": "10px",
            "padding": "16px 20px",
            "minWidth": "160px",
            "boxShadow": "0 4px 16px rgba(0,0,0,0.3)",
        }))

    return html.Div([
        html.Div("⚠  Value at Risk (VaR) — Historical Simulation",
                 style={"fontSize": "13px", "fontWeight": "600", "color": TEXT_MUTED,
                         "letterSpacing": "2px", "textTransform": "uppercase",
                         "marginBottom": "12px", "paddingLeft": "4px"}),
        html.Div(cards, style={"display": "flex", "gap": "14px", "flexWrap": "wrap"}),
    ])


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AlphaPulse Dashboard starting...")
    print("  Open browser at:  http://127.0.0.1:8050")
    print("=" * 55 + "\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
