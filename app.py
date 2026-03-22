"""Interactive naturalisation dashboard with department selector."""

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from dash import Dash, html, dcc, Output, Input, no_update, ctx
import dash_bootstrap_components as dbc

# Department names
DEPT_NAMES = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes",
    "09": "Ariège", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze", "21": "Côte-d'Or",
    "22": "Côtes-d'Armor", "23": "Creuse", "24": "Dordogne", "25": "Doubs",
    "26": "Drôme", "27": "Eure", "28": "Eure-et-Loir", "29": "Finistère",
    "2A": "Corse-du-Sud", "2B": "Haute-Corse", "30": "Gard", "31": "Haute-Garonne",
    "32": "Gers", "33": "Gironde", "34": "Hérault", "35": "Ille-et-Vilaine",
    "36": "Indre", "37": "Indre-et-Loire", "38": "Isère", "39": "Jura",
    "40": "Landes", "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire",
    "44": "Loire-Atlantique", "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne",
    "48": "Lozère", "49": "Maine-et-Loire", "50": "Manche", "51": "Marne",
    "52": "Haute-Marne", "53": "Mayenne", "54": "Meurthe-et-Moselle", "55": "Meuse",
    "56": "Morbihan", "57": "Moselle", "58": "Nièvre", "59": "Nord",
    "60": "Oise", "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dôme",
    "64": "Pyrénées-Atlantiques", "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales",
    "67": "Bas-Rhin", "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône",
    "71": "Saône-et-Loire", "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie",
    "75": "Paris", "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines",
    "79": "Deux-Sèvres", "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne",
    "83": "Var", "84": "Vaucluse", "85": "Vendée", "86": "Vienne",
    "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne",
    "90": "Territoire de Belfort", "91": "Essonne", "92": "Hauts-de-Seine",
    "93": "Seine-Saint-Denis", "94": "Val-de-Marne", "95": "Val-d'Oise",
    "971": "Guadeloupe", "972": "Martinique", "973": "Guyane",
    "974": "La Réunion", "976": "Mayotte", "99": "Étranger",
    "02A": "Corse-du-Sud", "02B": "Haute-Corse", "020": "Corse",
}


def dept_label(code):
    short = code.lstrip("0") or "0"
    name = DEPT_NAMES.get(code, "") or DEPT_NAMES.get(short, "")
    return f"{short} {name}" if name else short


# Colors
PALETTE = [
    "#4fc3f7", "#ffb74d", "#66bb6a", "#ef5350", "#ce93d8",
    "#4db6ac", "#ff8a65", "#9575cd", "#aed581", "#e57373",
    "#64b5f6", "#ffd54f", "#81c784", "#f06292", "#ba68c8",
]
COHORT_COLORS = {"2023": "#ce93d8", "2024": "#4fc3f7", "2025": "#66bb6a"}
BG = "#0a0a14"
CARD_BG = "#111827"
PANEL = "#0f1729"
ACCENT = "#1e293b"
TEXT = "#e2e8f0"
TEXT_DIM = "#94a3b8"
MIN_WAIT = 6

# Load data
folder = Path(__file__).parent
df_all = pd.read_csv(folder / "all_entries.csv", dtype={"dept": str})
df_all = df_all[df_all["dept"].notna() & (df_all["dept"] != "")]
df_nat_full = df_all[df_all["nat_type"] == "NAT"].copy()
df_nat_full["pub_date_dt"] = pd.to_datetime(df_nat_full["pub_date"])
df_nat_full["month"] = df_nat_full["pub_date_dt"].dt.to_period("M").astype(str)

ALL_MONTHS = pd.period_range("2025-01", "2026-03", freq="M").astype(str).tolist()
ALL_MONTHS_FULL = pd.period_range("2024-01", "2026-03", freq="M").astype(str).tolist()

# Build department options sorted by count
dept_counts = df_nat_full["dept"].value_counts()
DEPT_OPTIONS = [{"label": f"{dept_label(d)} ({n:,} NAT)", "value": d}
                for d, n in dept_counts.items() if n >= 10]
DEFAULT_DEPTS = ["075", "078"]

LAYOUT_COMMON = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=PANEL,
    font=dict(family="Inter, -apple-system, sans-serif", color=TEXT, size=12),
    margin=dict(t=45, b=45, l=50, r=20),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor=ACCENT, borderwidth=1,
                font=dict(size=10)),
)


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def get_color(idx):
    return PALETTE[idx % len(PALETTE)]


def make_gaussian(depts, serie_year=None, title=""):
    fig = go.Figure()
    mx = np.linspace(0, 60, 500)
    df = df_nat_full[df_nat_full["wait_months"] >= MIN_WAIT]
    for i, dept in enumerate(depts):
        s = df[df["dept"] == dept]
        if serie_year:
            s = s[s["serie_year"] == serie_year]
        vals = s["wait_months"].values
        if len(vals) < 3:
            continue
        col = get_color(i)
        mu, sigma = norm.fit(vals)
        pdf_vals = norm.pdf(mx, loc=mu, scale=sigma)
        fig.add_trace(go.Scatter(
            x=mx, y=pdf_vals, mode="lines",
            name=f"dép. {dept}<br>n={len(vals)}  μ={mu:.1f}  σ={sigma:.1f}",
            line=dict(color=col, width=2.5),
            fill="tozeroy", fillcolor=hex_to_rgba(col, 0.12),
        ))
        fig.add_vline(x=mu, line=dict(color=col, width=1.5, dash="dash"),
                      annotation_text=f"{mu:.0f}mo", annotation_font_color=col,
                      annotation_font_size=11)
    fig.update_layout(**LAYOUT_COMMON, title=dict(text=title, font=dict(size=14)),
                      xaxis_title="Months from Jan 1 of série year",
                      yaxis_title="Density", xaxis=dict(range=[0, 50]),
                      height=350)
    return fig


def make_timeseries(depts, serie_year=None, title=""):
    fig = go.Figure()
    df = df_nat_full[df_nat_full["wait_months"] >= MIN_WAIT]
    sub = df[df["pub_date_dt"] >= "2025-01-01"].copy()
    if serie_year:
        sub = sub[sub["serie_year"] == serie_year]
    for i, dept in enumerate(depts):
        dept_sub = sub[sub["dept"] == dept]
        monthly = dept_sub.groupby("month").size()
        if len(monthly) == 0:
            continue
        monthly = monthly.reindex(ALL_MONTHS, fill_value=0)
        col = get_color(i)
        fig.add_trace(go.Bar(
            x=ALL_MONTHS, y=monthly.values, name=dept_label(dept),
            marker_color=col, opacity=0.7, marker_line=dict(width=0),
            text=[str(v) if v > 0 else "" for v in monthly.values],
            textposition="outside", textfont=dict(size=8, color=TEXT_DIM),
        ))
    fig.update_layout(**LAYOUT_COMMON, title=dict(text=title, font=dict(size=14)),
                      xaxis_title="", yaxis_title="Dossiers",
                      barmode="group", height=350,
                      xaxis=dict(tickangle=-45, type="category"),
                      bargap=0.15, bargroupgap=0.05)
    return fig


def make_stacked(dept, title=""):
    fig = go.Figure()
    sub = df_nat_full[(df_nat_full["dept"] == dept) &
                      (df_nat_full["pub_date_dt"] >= "2024-01-01")]
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
                      title=dict(text=title, font=dict(size=14)),
                      xaxis_title="", yaxis_title="Dossiers",
                      barmode="stack", height=350,
                      xaxis=dict(tickangle=-45, type="category"),
                      bargap=0.15)
    return fig


def section_title(text):
    return html.H5(text, className="mb-3 mt-4",
                   style={"color": TEXT, "fontWeight": "500", "borderBottom": f"1px solid {ACCENT}",
                           "paddingBottom": "8px"})


def note(text):
    return html.P(text, style={"color": TEXT_DIM, "fontSize": "12px", "fontStyle": "italic",
                               "margin": "4px 8px 16px", "lineHeight": "1.5"})


CARD_STYLE = {"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}",
              "borderRadius": "8px", "overflow": "hidden"}

# Build app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Dark dropdown styling for Dash 4.x
app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>Naturalisation Analytics</title>
{%favicon%}
{%css%}
<style>
/* Dash 4.x dropdown */
.dash-dropdown .Select-control {
    background-color: #111827 !important;
    border-color: #1e293b !important;
    color: #e2e8f0 !important;
}
.dash-dropdown .Select-menu-outer {
    background-color: #111827 !important;
    border-color: #1e293b !important;
}
.dash-dropdown .Select-option {
    background-color: #111827 !important;
    color: #e2e8f0 !important;
}
.dash-dropdown .Select-option.is-focused,
.dash-dropdown .Select-option:hover {
    background-color: #1e293b !important;
}
.dash-dropdown .Select-value {
    background-color: #1e293b !important;
    border-color: #334155 !important;
    color: #e2e8f0 !important;
}
.dash-dropdown .Select-value-label {
    color: #e2e8f0 !important;
}
.dash-dropdown .Select-placeholder,
.dash-dropdown .Select-input > input {
    color: #94a3b8 !important;
}
.dash-dropdown .Select-clear-zone,
.dash-dropdown .Select-arrow-zone {
    color: #94a3b8 !important;
}
.dash-dropdown .Select-value-icon {
    border-right-color: #334155 !important;
}
.dash-dropdown .Select-value-icon:hover {
    background-color: #334155 !important;
    color: #ef5350 !important;
}
/* Also handle newer class names */
.dash-dropdown input {
    color: #e2e8f0 !important;
}
.dash-dropdown [class*="menu"] {
    background-color: #111827 !important;
}
.dash-dropdown [class*="option"] {
    background-color: #111827 !important;
    color: #e2e8f0 !important;
}
.dash-dropdown [class*="option"]:hover,
.dash-dropdown [class*="option"][class*="focused"] {
    background-color: #1e293b !important;
}
.dash-dropdown [class*="control"] {
    background-color: #111827 !important;
    border-color: #1e293b !important;
}
.dash-dropdown [class*="singleValue"],
.dash-dropdown [class*="multiValue"] {
    color: #e2e8f0 !important;
}
.dash-dropdown [class*="multiValue"] {
    background-color: #1e293b !important;
}
.dash-dropdown [class*="multiValueLabel"] {
    color: #e2e8f0 !important;
}
.dash-dropdown [class*="multiValueRemove"]:hover {
    background-color: #334155 !important;
    color: #ef5350 !important;
}
.dash-dropdown [class*="placeholder"] {
    color: #94a3b8 !important;
}
.dash-dropdown [class*="indicatorSeparator"] {
    background-color: #1e293b !important;
}
.dash-dropdown [class*="dropdownIndicator"],
.dash-dropdown [class*="clearIndicator"] {
    color: #94a3b8 !important;
}
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>'''

app.layout = html.Div([
    # Header
    html.Div([
        html.H2("Naturalisation Analytics", style={"color": TEXT, "fontWeight": "700", "marginBottom": "4px"}),
        html.P("Wait time analysis by department — NAT only",
               style={"color": TEXT_DIM, "fontSize": "14px", "marginBottom": "0"}),
    ], style={"padding": "24px 32px 16px", "borderBottom": f"1px solid {ACCENT}"}),

    dbc.Container([
        # Department selector
        dbc.Row([
            dbc.Col([
                html.Label("Select departments:", style={"color": TEXT, "fontSize": "13px", "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="dept-select",
                    options=DEPT_OPTIONS,
                    value=DEFAULT_DEPTS,
                    multi=True,
                    placeholder="Select departments...",
                    style={"backgroundColor": CARD_BG},
                    className="dash-bootstrap",
                ),
            ], md=8),
        ], className="mt-4 mb-3"),

        # KPI cards
        html.Div(id="kpi-cards"),

        # 1. All dossiers combined
        section_title("All Dossiers Combined (wait >= 6mo)"),
        dbc.Row([
            dbc.Col([dcc.Graph(id="gauss-all", config={"displayModeBar": False})], md=12),
        ], className="g-3"),
        note("Combined 2023X + 2024X + 2025X. Mixes cohorts with different processing speeds."),

        # 2. Cohort breakdown
        section_title("Cohort Breakdown by Month (all data, no filter)"),
        html.Div(id="stacked-bars"),
        note("Shows 2023X/2024X/2025X mix per month. Use this to assess if a department "
             "has finished 2023X processing and whether 2024X data is complete enough for the Gaussian."),

        # 3. Monthly processing volume
        section_title("Monthly Processing Volume — click a bar to view entries"),
        dbc.Row([
            dbc.Col([dcc.Graph(id="ts-all", config={"displayModeBar": False})], md=12),
        ], className="g-3"),
        note("Monthly NAT volume (wait >= 6mo). Sep-Dec 2024 drought is real and nationwide."),
        dbc.Row([
            dbc.Col([dcc.Graph(id="ts-2024", config={"displayModeBar": False})], md=6),
            dbc.Col([dcc.Graph(id="ts-2025", config={"displayModeBar": False})], md=6),
        ], className="g-3 mt-1"),
        note("2024X: check if department has transitioned from 2023X (see stacked bars). "
             "2025X: very few published — bulk processing not expected until late 2026."),

        # 4. Wait time distributions
        section_title("Wait Time Distributions (wait >= 6mo)"),
        dbc.Row([
            dbc.Col([dcc.Graph(id="gauss-2024", config={"displayModeBar": False})], md=6),
            dbc.Col([dcc.Graph(id="gauss-2025", config={"displayModeBar": False})], md=6),
        ], className="g-3"),
        note("Entries < 6mo excluded (likely re-filed old dossiers). "
             "2025X is incomplete — only fastest cases visible, not usable for prediction. "
             "Check stacked bars above to assess if a department's 2024X distribution is complete."),

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


# Callbacks
@app.callback(
    Output("kpi-cards", "children"),
    Output("gauss-2024", "figure"),
    Output("gauss-2025", "figure"),
    Output("gauss-all", "figure"),
    Output("ts-all", "figure"),
    Output("ts-2024", "figure"),
    Output("ts-2025", "figure"),
    Output("stacked-bars", "children"),
    Input("dept-select", "value"),
)
def update_charts(selected_depts):
    if not selected_depts:
        empty = go.Figure()
        empty.update_layout(**LAYOUT_COMMON, height=350)
        return [], empty, empty, empty, empty, empty, empty, []

    depts = selected_depts

    # KPI cards — row per cohort year
    df_filtered = df_nat_full[df_nat_full["wait_months"] >= MIN_WAIT]
    md_size = max(2, 12 // max(len(depts), 1))
    kpi_rows = []
    for sy in [2024, 2025]:
        row_cards = []
        for dept in depts:
            s = df_filtered[(df_filtered["dept"] == dept) & (df_filtered["serie_year"] == sy)]
            if len(s) >= 3:
                v = s["wait_months"]
                row_cards.append(dbc.Col(
                    dbc.Card(dbc.CardBody([
                        html.P(f"{sy}X {dept_label(dept)}", className="mb-1",
                               style={"color": TEXT_DIM, "fontSize": "11px", "textTransform": "uppercase", "letterSpacing": "1px"}),
                        html.H4(f"{v.mean():.1f} mo", className="mb-0", style={"color": TEXT, "fontWeight": "600"}),
                        html.Small(f"n={len(s):,}  median={v.median():.1f}", style={"color": TEXT_DIM}),
                    ], className="p-3"),
                    style={"backgroundColor": CARD_BG, "border": f"1px solid {ACCENT}", "borderRadius": "8px"}),
                    md=md_size,
                ))
        if row_cards:
            kpi_rows.append(dbc.Row(row_cards, className="mt-2 g-3"))
    kpi_row = html.Div(kpi_rows, className="mb-3") if kpi_rows else html.Div()

    # Gaussians
    g2024 = make_gaussian(depts, 2024, "2024X Dossiers")
    g2025 = make_gaussian(depts, 2025, "2025X Dossiers (incomplete)")
    g_all = make_gaussian(depts, title="All Dossiers Combined")

    # Time series
    ts_all = make_timeseries(depts, title="All NAT Dossiers")
    ts_2024 = make_timeseries(depts, 2024, "2024X Dossiers")
    ts_2025 = make_timeseries(depts, 2025, "2025X Dossiers")

    # Stacked bars — one per department
    stacked = []
    cols_per_row = min(len(depts), 3)
    md = 12 // cols_per_row
    row_children = []
    for i, dept in enumerate(depts):
        row_children.append(dbc.Col([
            dbc.Card(dcc.Graph(figure=make_stacked(dept, f"dép. {dept} — cohort breakdown"),
                               config={"displayModeBar": False}), style=CARD_STYLE),
        ], md=md))
        if (i + 1) % cols_per_row == 0 or i == len(depts) - 1:
            stacked.append(dbc.Row(row_children, className="g-3 mt-1"))
            row_children = []

    return kpi_row, g2024, g2025, g_all, ts_all, ts_2024, ts_2025, stacked


@app.callback(
    Output("modal", "is_open"),
    Output("modal-title", "children"),
    Output("modal-body", "children"),
    Input("ts-all", "clickData"),
    Input("ts-2024", "clickData"),
    Input("ts-2025", "clickData"),
    Input("dept-select", "value"),
    prevent_initial_call=True,
)
def on_bar_click(click_all, click_2024, click_2025, selected_depts):
    triggered = ctx.triggered_id
    if triggered == "dept-select":
        return no_update, no_update, no_update

    click = {"ts-all": click_all, "ts-2024": click_2024, "ts-2025": click_2025}.get(triggered)
    if not click or not selected_depts:
        return no_update, no_update, no_update

    point = click["points"][0]
    month = point["x"]
    curve = point["curveNumber"]
    dept = selected_depts[curve] if curve < len(selected_depts) else selected_depts[0]
    serie_year = {"ts-2024": 2024, "ts-2025": 2025}.get(triggered)

    sub = df_nat_full[(df_nat_full["month"] == month) & (df_nat_full["dept"] == dept)]
    if serie_year:
        sub = sub[sub["serie_year"] == serie_year]
    sub = sub.sort_values("serie_num")

    title = f"dép. {dept} — {month}"
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
