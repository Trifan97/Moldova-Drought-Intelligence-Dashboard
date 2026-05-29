"""
constants.py — shared configuration for Moldova Drought Intelligence Dashboard
"""

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY        = "#0B1E33"
NAVY2       = "#0F2640"
NAVY3       = "#162f47"
TEAL        = "#1AA99A"
TEAL_DARK   = "#14877a"
AMBER       = "#E8923A"
WHITE       = "#FFFFFF"
TEXT        = "#E8EAED"
MUTED       = "#8A9BAE"
BORDER      = "#1E3550"

# 7-class drought palette  (dark → light → dark  through neutral green)
CLASS_COLORS = {
    "Extremely Dry"  : "#8B0000",
    "Severely Dry"   : "#D4693A",
    "Moderately Dry" : "#F4A460",
    "Normal"         : "#4a9b6f",
    "Moderately Wet" : "#5BA3C9",
    "Severely Wet"   : "#2166AC",
    "Extremely Wet"  : "#08306B",
}

CLASS_ORDER = [
    "Extremely Dry", "Severely Dry", "Moderately Dry",
    "Normal",
    "Moderately Wet", "Severely Wet", "Extremely Wet",
]

# PyDeck colour tuples  [R, G, B, A]
CLASS_PYDECK = {
    "Extremely Dry"  : [139,   0,   0, 220],
    "Severely Dry"   : [212, 105,  58, 200],
    "Moderately Dry" : [244, 164,  96, 180],
    "Normal"         : [ 74, 155, 111, 160],
    "Moderately Wet" : [ 91, 163, 201, 180],
    "Severely Wet"   : [ 33, 102, 172, 200],
    "Extremely Wet"  : [  8,  48, 107, 220],
}

# Station coordinates — precise survey coordinates from MD_Stations_all.shp
# (previously some values were rounded approximations; Baltata and Bravicea
#  had errors of ~22 km and ~17 km respectively)
STATION_COORDS = {
    "Baltata":      (47.055364, 29.036234),
    "Balti":        (47.766735, 27.887407),
    "Bravicea":     (47.372135, 28.438405),
    "Briceni":      (48.352057, 27.102111),
    "Cahul":        (45.899149, 28.213516),
    "Camenca":      (48.049768, 28.686427),
    "Ceadir-Lunga": (46.071370, 28.824953),
    "Chisinau":     (46.971568, 28.848373),
    "Comrat":       (46.302793, 28.629546),
    "Cornesti":     (47.367089, 27.994043),
    "Dubasari":     (47.280522, 29.129234),
    "Falesti":      (47.583338, 27.704996),
    "Leova":        (46.488363, 28.283490),
    "Rabnita":      (47.772418, 29.016197),
    "Soroca":       (48.198435, 28.312000),
    "Tiraspol":     (46.865628, 29.584229),
}

# Periods
PERIODS = {
    "Full period (1961–2020)": "full",
    "Period 1 (1961–1990)":   "p1",
    "Period 2 (1991–2020)":   "p2",
}
PERIOD_LABELS = {v: k for k, v in PERIODS.items()}
PERIOD_YEARS  = {"full": (1961, 2020), "p1": (1961, 1990), "p2": (1991, 2020)}

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

# Moldova map centre / zoom
MAP_CENTER_LAT = 47.15
MAP_CENTER_LON = 28.50
MAP_ZOOM       = 6.8

# ── Plotly dark theme override ────────────────────────────────────────────────
import plotly.io as pio
import plotly.graph_objects as go

PLOTLY_TEMPLATE = go.layout.Template()
PLOTLY_TEMPLATE.layout = go.Layout(
    paper_bgcolor = NAVY2,
    plot_bgcolor  = NAVY,
    font          = dict(color=TEXT, family="DM Sans, sans-serif", size=12),
    title_font    = dict(color=TEXT, size=14),
    xaxis = dict(
        gridcolor=BORDER, linecolor=BORDER,
        tickcolor=MUTED,  zerolinecolor=BORDER,
    ),
    yaxis = dict(
        gridcolor=BORDER, linecolor=BORDER,
        tickcolor=MUTED,  zerolinecolor=BORDER,
    ),
    legend = dict(
        bgcolor=NAVY2, bordercolor=BORDER, borderwidth=1,
        font=dict(color=TEXT, size=11),
    ),
    colorway=[TEAL, AMBER, "#D4693A", "#5BA3C9", "#8B0000", "#08306B", "#4a9b6f"],
    margin=dict(l=50, r=20, t=40, b=40),
    hoverlabel=dict(
        bgcolor=NAVY2, bordercolor=TEAL,
        font=dict(color=TEXT, size=12),
    ),
)
pio.templates["drought_dark"] = PLOTLY_TEMPLATE
pio.templates.default = "drought_dark"
