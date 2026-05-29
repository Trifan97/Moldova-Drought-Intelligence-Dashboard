"""
utils/styles.py — Shared CSS for the Moldova Drought Intelligence Dashboard.

Usage
-----
from utils.styles import apply_dark_theme
apply_dark_theme()                    # base dark theme (all pages)
apply_dark_theme("station_explorer") # base + page-specific extras
"""

import streamlit as st

# ── Base CSS — applied on every page ─────────────────────────────────────────
_BASE_CSS = """
/* ── Reset Streamlit chrome ───────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Global background & font ─────────────────────────────────────────────── */
body, .stApp { background-color: #0B1E33; color: #E8EAED; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0F2640 !important;
    border-right: 1px solid #1E3550;
}
[data-testid="stSidebar"] * { color: #E8EAED !important; }

/* ── Headings ─────────────────────────────────────────────────────────────── */
h1 { color: #E8EAED !important; font-size: 1.6rem !important; }
h2 { color: #E8EAED !important; font-size: 1.2rem !important;
     border-bottom: 1px solid #1E3550; padding-bottom: 6px; }
h3 { color: #1AA99A !important; font-size: 1rem !important; }

/* ── Metric cards ─────────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #0F2640;
    border: 1px solid #1E3550;
    border-radius: 8px;
    padding: 10px 14px;
    border-left: 3px solid #1AA99A;
}
[data-testid="metric-container"] label {
    color: #8A9BAE !important;
    font-size: 0.72rem !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #E8EAED !important;
    font-size: 1.6rem !important;
    font-weight: 600;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

/* ── Dividers ─────────────────────────────────────────────────────────────── */
hr { border-color: #1E3550 !important; }

/* ── Form widgets ─────────────────────────────────────────────────────────── */
.stSelectbox > div > div { background: #0F2640 !important; border-color: #1E3550 !important; }
.stRadio > label { color: #E8EAED !important; }

/* ── Shared card ──────────────────────────────────────────────────────────── */
.card {
    background: #0F2640;
    border: 1px solid #1E3550;
    border-radius: 8px;
    padding: 14px 16px;
    border-left: 3px solid #1AA99A;
    margin-bottom: 10px;
}

/* ── Insight / info boxes ─────────────────────────────────────────────────── */
.insight-box {
    background: rgba(26,169,154,0.08);
    border: 1px solid rgba(26,169,154,0.3);
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.85rem;
    color: #E8EAED;
    margin-bottom: 14px;
    line-height: 1.7;
}
.insight-box b { color: #1AA99A; }

/* ── Tag pills ────────────────────────────────────────────────────────────── */
.tag {
    display: inline-block;
    background: rgba(26,169,154,0.12);
    border: 1px solid rgba(26,169,154,0.3);
    color: #1AA99A;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin: 2px;
}
"""

# ── Page-specific CSS extensions ──────────────────────────────────────────────
_PAGE_CSS: dict[str, str] = {

    "home": """
        /* Navigation cards on the landing page */
        .nav-card {
            background: #0F2640;
            border: 1px solid #1E3550;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 10px;
            transition: border-color 0.2s;
            border-left: 3px solid #1AA99A;
        }
        .nav-card:hover { border-color: #1AA99A; }
        .nav-card h3 { margin: 0 0 6px 0 !important; }
        .nav-card p  { color: #8A9BAE; font-size: 0.85rem; margin: 0; }
    """,

    "station_explorer": """
        /* Stat cards in the station panel */
        .stat-card {
            background: #0F2640;
            border: 1px solid #1E3550;
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 3px solid #1AA99A;
            margin-bottom: 10px;
        }
        .stat-label {
            color: #8A9BAE;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 2px;
        }
        .stat-value { color: #E8EAED; font-size: 1.35rem; font-weight: 600; }
        .stat-sub   { color: #8A9BAE; font-size: 0.78rem; }
        .delta-pos  { color: #1AA99A; }
        .delta-neg  { color: #E8923A; }
        .info-box {
            background: rgba(26,169,154,0.08);
            border: 1px solid rgba(26,169,154,0.25);
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 0.82rem;
            color: #E8EAED;
            margin-bottom: 12px;
        }
    """,

    "drought_risk": """
        /* Risk-tier colour helpers */
        .risk-high   { color: #8B0000 !important; font-weight: 600; }
        .risk-med    { color: #D4693A !important; font-weight: 600; }
        .risk-low    { color: #4a9b6f !important; font-weight: 600; }
        /* Station risk cards in the sidebar ranking */
        .risk-card {
            background: #0F2640;
            border: 1px solid #1E3550;
            border-radius: 8px;
            padding: 12px 16px;
            border-left: 3px solid;
            margin-bottom: 8px;
        }
    """,

    "ml_explorer": """
        /* Probability bar track */
        .prob-bar-wrap {
            background: #162f47;
            border-radius: 4px;
            height: 20px;
            width: 100%;
            margin-bottom: 6px;
            overflow: hidden;
        }
        .prob-bar { height: 100%; border-radius: 4px; transition: width 0.3s; }
        /* Model summary card */
        .model-card {
            background: #0F2640;
            border: 1px solid #1E3550;
            border-radius: 8px;
            padding: 14px 16px;
            border-left: 3px solid #1AA99A;
            margin-bottom: 10px;
        }
    """,
}


def apply_dark_theme(page: str = "") -> None:
    """
    Inject the shared dark-theme CSS, plus optional page-specific extras.

    Parameters
    ----------
    page : str
        One of: "home", "station_explorer", "climate_signal",
                "drought_risk", "ml_explorer"
        Leave empty (or omit) for base styles only.
    """
    css = _BASE_CSS
    if page and page in _PAGE_CSS:
        css += _PAGE_CSS[page]
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
