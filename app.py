"""Interactive naturalisation dashboard with clickable time series."""

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from dash import Dash, html, dcc, Output, Input, no_update, ctx
import dash_bootstrap_components as dbc

DEPTS = {"075": "Paris (075)", "078": "Yvelines (078)"}
DEPT_COLORS = {"075": "#4fc3f7", "078": "#ffb74d"}
DEPT_FILL = {"075": "rgba(79,195,247,0.12)", "078": "rgba(255,183,77,0.12)"}
BG = "#0a0a14"
CARD_BG = "#111827"
PANEL = "#0f1729"
ACCENT = "#1e293b"
TEXT = "#e2e8f0"
TEXT_DIM = "#94a3b8"

# Load data
folder = Path(__file__).parent
df_all = pd.read_csv(folder / "all_entries.csv", dtype={"dept": str})
df_all = df_all[df_all["dept"].notna() & (df_all["dept"] != "")]
df_all = df_all[df_all["dept"].isin(["075", "078"])]
df_nat = df_all[df_all["nat_type"] == "NAT"].copy()
df_nat["pub_date_dt"] = pd.to_datetime(df_nat["pub_date"])
df_nat["month"] = df_nat["pub_date_dt"].dt.to_period("M").astype(str)

ALL_MONTHS = pd.period_range("2025-01", "2026-03", freq="M").astype(str).tolist()

# Stats
stats = {}
for sy in [2024, 2025, None]:
    stats[sy] = {}
    for dept in DEPTS:
        s = df_nat[df_nat["dept"] == dept]
        if sy:
            s = s[s["serie_year"] == sy]
        v = s["wait_months"]
        if len(v) >= 3:
            stats[sy][dept] = {"n": len(v), "mean": v.mean(), "median": v.median(),
                               "min": v.min(), "max": v.max()}

LAYOUT_COMMON = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=PANEL,
    font=dict(family="Inter, -apple-system, sans-serif", color=TEXT, size=12),
    margin=dict(t=45, b=45, l=50, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=ACCENT, borderwidth=1,
                font=dict(size=11)),
)


def make_gaussian(df, serie_year=None, title=""):
    fig = go.Figure()
    mx = np.linspace(0, 60, 500)
    for dept, name in DEPTS.items():
        s = df[df["dept"] == dept]
        if serie_year:
            s = s[s["serie_year"] == serie_year]
        vals = s["wait_months"].values
        if len(vals) < 3:
            continue
        mu, sigma = norm.fit(vals)
        pdf_vals = norm.pdf(mx, loc=mu, scale=sigma)
        fig.add_trace(go.Scatter(
            x=mx, y=pdf_vals, mode="lines",
            name=f"{name}<br>n={len(vals)}  μ={mu:.1f}  σ={sigma:.1f}",
            line=dict(color=DEPT_COLORS[dept], width=2.5),
            fill="tozeroy", fillcolor=DEPT_FILL[dept],
        ))
        fig.add_vline(x=mu, line=dict(color=DEPT_COLORS[dept], width=1.5, dash="dash"),
                      annotation_text=f"{mu:.0f}mo", annotation_font_color=DEPT_COLORS[dept],
                      annotation_font_size=11)
    fig.update_layout(**LAYOUT_COMMON, title=dict(text=title, font=dict(size=14)),
                      xaxis_title="Months from Jan 1 of série year",
                      yaxis_title="Density", xaxis=dict(range=[0, 50]),
                      height=320)
    return fig


def make_timeseries(df, serie_year=None, title=""):
    fig = go.Figure()
    sub = df[df["pub_date_dt"] >= "2025-01-01"].copy()
    if serie_year:
        sub = sub[sub["serie_year"] == serie_year]

    for dept, name in DEPTS.items():
        dept_sub = sub[sub["dept"] == dept]
        monthly = dept_sub.groupby("month").size()
        if len(monthly) == 0:
            continue
        monthly = monthly.reindex(ALL_MONTHS, fill_value=0)
        fig.add_trace(go.Bar(
            x=ALL_MONTHS, y=monthly.values, name=name,
            marker_color=DEPT_COLORS[dept], opacity=0.7,
            marker_line=dict(width=0),
            text=[str(v) if v > 0 else "" for v in monthly.values],
            textposition="outside", textfont=dict(size=8, color=TEXT_DIM),
        ))
    fig.update_layout(**LAYOUT_COMMON, title=dict(text=title, font=dict(size=14)),
                      xaxis_title="", yaxis_title="Dossiers",
                      barmode="group", height=320,
                      xaxis=dict(tickangle=-45, type="category"),
                      bargap=0.15, bargroupgap=0.05)
    return fig


COHORT_COLORS = {"2023": "#ce93d8", "2024": "#4fc3f7", "2025": "#66bb6a"}
ALL_MONTHS_FULL = pd.period_range("2024-01", "2026-03", freq="M").astype(str).tolist()

# Unfiltered NAT for stacked bars
df_nat_unfiltered = df_all[df_all["nat_type"] == "NAT"].copy()
df_nat_unfiltered["pub_date_dt"] = pd.to_datetime(df_nat_unfiltered["pub_date"])
df_nat_unfiltered["month"] = df_nat_unfiltered["pub_date_dt"].dt.to_period("M").astype(str)


def make_stacked(dept_code, dept_name):
    fig = go.Figure()
    sub = df_nat_unfiltered[(df_nat_unfiltered["dept"] == dept_code) &
                            (df_nat_unfiltered["pub_date_dt"] >= "2024-01-01")]

    for year in [2023, 2024, 2025]:
        ys = sub[sub["serie_year"] == year]
        monthly = ys.groupby("month").size().reindex(ALL_MONTHS_FULL, fill_value=0)
        fig.add_trace(go.Bar(
            x=ALL_MONTHS_FULL, y=monthly.values, name=f"{year}X",
            marker_color=COHORT_COLORS[str(year)], opacity=0.8,
            text=[str(v) if v > 5 else "" for v in monthly.values],
            textposition="inside", textfont=dict(size=8),
        ))

    fig.update_layout(**LAYOUT_COMMON,
                      title=dict(text=f"{dept_name} — cohort breakdown (all data)", font=dict(size=14)),
                      xaxis_title="", yaxis_title="Dossiers",
                      barmode="stack", height=350,
                      xaxis=dict(tickangle=-45, type="category"),
                      bargap=0.15)
    return fig


def stat_card(label, value, sub=""):
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="mb-1", style={"color": TEXT_DIM, "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "1px"}),
            html.H4(value, className="mb-0", style={"color": TEXT, "fontWeight": "600"}),
            html.Small(sub, style={"color": TEXT_DIM}) if sub else None,
        ], className="p-3"),
        style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px"},
    )


def section_title(text):
    return html.H5(text, className="mb-3 mt-4",
                   style={"color": TEXT, "fontWeight": "500", "borderBottom": f"1px solid {ACCENT}",
                           "paddingBottom": "8px"})


def note(text):
    return html.P(text, style={"color": TEXT_DIM, "fontSize": "12px", "fontStyle": "italic",
                               "margin": "4px 8px 16px", "lineHeight": "1.5"})


# Build app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div([
    # Header
    html.Div([
        html.H2("Naturalisation Analytics", style={"color": TEXT, "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Wait time analysis — Paris (075) vs Yvelines (078) — NAT only",
               style={"color": TEXT_DIM, "fontSize": "14px", "marginBottom": "0"}),
    ], style={"padding": "24px 32px 16px", "borderBottom": f"1px solid {ACCENT}"}),

    dbc.Container([
        # KPI cards
        dbc.Row([
            dbc.Col(stat_card("2024X Paris", f"{stats[2024].get('075', {}).get('mean', 0):.1f} mo",
                              f"n={stats[2024].get('075', {}).get('n', 0):,}  median={stats[2024].get('075', {}).get('median', 0):.1f}"), md=3),
            dbc.Col(stat_card("2024X Yvelines", f"{stats[2024].get('078', {}).get('mean', 0):.1f} mo",
                              f"n={stats[2024].get('078', {}).get('n', 0):,}  median={stats[2024].get('078', {}).get('median', 0):.1f}"), md=3),
            dbc.Col(stat_card("2025X Paris", f"{stats[2025].get('075', {}).get('mean', 0):.1f} mo",
                              f"n={stats[2025].get('075', {}).get('n', 0):,}  median={stats[2025].get('075', {}).get('median', 0):.1f}"), md=3),
            dbc.Col(stat_card("2025X Yvelines", f"{stats[2025].get('078', {}).get('mean', 0):.1f} mo",
                              f"n={stats[2025].get('078', {}).get('n', 0):,}  median={stats[2025].get('078', {}).get('median', 0):.1f}"), md=3),
        ], className="mt-4 g-3"),

        # Gaussian distributions
        section_title("Wait Time Distributions"),
        dbc.Row([
            dbc.Col([
                dbc.Card(dcc.Graph(figure=make_gaussian(df_nat, 2024, "2024X Dossiers"), config={"displayModeBar": False}),
                         style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px", "overflow": "hidden"}),
                note(f"Entries < 6mo excluded (likely re-filed old dossiers). "
                     f"Paris (n={stats[2024].get('075', {}).get('n', 0):,}, median {stats[2024].get('075', {}).get('median', 0):.0f}mo): reliable — 95% of Jan-Mar 2026 output is 2024X. "
                     f"Yvelines (n={stats[2024].get('078', {}).get('n', 0):,}, median {stats[2024].get('078', {}).get('median', 0):.0f}mo): UNRELIABLE — only fast-track pipeline dossiers. "
                     f"Yvelines still processing 2023X in 2026; slow-pipeline 2024X not yet published."),
            ], md=6),
            dbc.Col([
                dbc.Card(dcc.Graph(figure=make_gaussian(df_nat, 2025, "2025X Dossiers (incomplete)"), config={"displayModeBar": False}),
                         style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px", "overflow": "hidden"}),
                note(f"2025X is barely started — only {stats[2025].get('075', {}).get('n', 0)} Paris + {stats[2025].get('078', {}).get('n', 0)} Yvelines entries so far. "
                     f"Distribution is right-censored (only fastest cases visible). Not usable for prediction."),
            ], md=6),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card(dcc.Graph(figure=make_gaussian(df_nat, title="All Dossiers Combined"), config={"displayModeBar": False}),
                         style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px", "overflow": "hidden"}),
                note("Combined 2023X + 2024X + 2025X. Mixes cohorts with different processing speeds. "
                     "For Paris: use the 2024X Gaussian. For Yvelines: no reliable Gaussian yet."),
            ], md=12),
        ], className="g-3 mt-1"),

        # Time series
        section_title("Monthly Processing Volume — click a bar to view entries"),
        dbc.Row([
            dbc.Col([
                dbc.Card(dcc.Graph(id="ts-all", figure=make_timeseries(df_nat, title="All NAT Dossiers"), config={"displayModeBar": False}),
                         style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px", "overflow": "hidden"}),
                note("Monthly NAT volume (wait >= 6mo). Sep-Dec 2024 drought is real and nationwide — not a data gap. "
                     "Paris publishes 3-5x more dossiers per month than Yvelines."),
            ], md=12),
        ], className="g-3"),
        dbc.Row([
            dbc.Col([
                dbc.Card(dcc.Graph(id="ts-2024", figure=make_timeseries(df_nat, 2024, "2024X Dossiers"), config={"displayModeBar": False}),
                         style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px", "overflow": "hidden"}),
                note("Paris 2024X ramped up from Jul 2025 and dominates by Jan 2026 (95% of output). "
                     "Yvelines 2024X peaked briefly in late 2024 then dropped — most still in slow pipeline."),
            ], md=6),
            dbc.Col([
                dbc.Card(dcc.Graph(id="ts-2025", figure=make_timeseries(df_nat, 2025, "2025X Dossiers"), config={"displayModeBar": False}),
                         style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px", "overflow": "hidden"}),
                note("Very few 2025X published. These are mostly low serie numbers (first dossiers filed Jan 2025). "
                     "Bulk 2025X processing not expected until late 2026 at earliest."),
            ], md=6),
        ], className="g-3 mt-1"),

        # Stacked bars
        section_title("Cohort Breakdown by Month (all data, no filter)"),
        dbc.Row([
            dbc.Col([
                dbc.Card(dcc.Graph(figure=make_stacked("075", "Paris (075)"), config={"displayModeBar": False}),
                         style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px", "overflow": "hidden"}),
                note("Clear transition: 2023X -> 2024X around Jul 2025. 2025X barely visible. "
                     "Two numbering tracks: short numbers (<10k, fast) and 6-digit (200k+, slow ~22mo)."),
            ], md=6),
            dbc.Col([
                dbc.Card(dcc.Graph(figure=make_stacked("078", "Yvelines (078)"), config={"displayModeBar": False}),
                         style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px", "overflow": "hidden"}),
                note("2023X still dominant even in Mar 2026. 2024X only appears in bursts. "
                     "Yvelines 2024X is 96% short numbers (fast track) — 6-digit slow pipeline hasn't started publishing."),
            ], md=6),
        ], className="g-3"),

        html.Div(style={"height": "40px"}),
    ], fluid=True),

    # Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="modal-title"), close_button=True,
                        style={"backgroundColor": CARD_BG, "borderBottom": f"1px solid {ACCENT}"}),
        dbc.ModalBody(id="modal-body",
                      style={"maxHeight": "70vh", "overflowY": "auto", "backgroundColor": CARD_BG}),
    ], id="modal", size="xl", is_open=False, centered=True),

], style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": "Inter, -apple-system, sans-serif"})


@app.callback(
    Output("modal", "is_open"),
    Output("modal-title", "children"),
    Output("modal-body", "children"),
    Input("ts-all", "clickData"),
    Input("ts-2024", "clickData"),
    Input("ts-2025", "clickData"),
    prevent_initial_call=True,
)
def on_bar_click(click_all, click_2024, click_2025):
    triggered = ctx.triggered_id
    click = {"ts-all": click_all, "ts-2024": click_2024, "ts-2025": click_2025}.get(triggered)
    if not click:
        return no_update, no_update, no_update

    point = click["points"][0]
    month = point["x"]
    curve = point["curveNumber"]
    dept = "075" if curve == 0 else "078"
    serie_year = {"ts-2024": 2024, "ts-2025": 2025}.get(triggered)

    sub = df_nat[(df_nat["month"] == month) & (df_nat["dept"] == dept)]
    if serie_year:
        sub = sub[sub["serie_year"] == serie_year]
    sub = sub.sort_values("serie_num")

    title = f"{DEPTS[dept]} — {month}"
    if serie_year:
        title += f" ({serie_year}X)"
    title += f" — {len(sub)} entries"

    if len(sub) == 0:
        return True, title, html.P("No entries found.", style={"color": TEXT_DIM})

    table = dbc.Table.from_dataframe(
        sub[["pub_date", "serie_full", "wait_months", "text"]].rename(columns={
            "pub_date": "Date", "serie_full": "Série",
            "wait_months": "Wait (mo)", "text": "Entry",
        }),
        striped=True, bordered=True, hover=True, color="dark", size="sm",
        style={"fontSize": "12px"},
    )
    return True, title, table


server = app.server  # for gunicorn

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    print(f"Dashboard: http://localhost:{port}")
    app.run(debug=False, host="0.0.0.0", port=port)
