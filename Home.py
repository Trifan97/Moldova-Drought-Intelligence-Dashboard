"""
Home.py — Moldova Drought Intelligence Dashboard
Landing page: key stats, navigation cards, data overview.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Moldova Drought Intelligence Dashboard",
    page_icon="🌦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import utils (path trick for Streamlit multipage) ─────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.data_loader import load_station_summary, load_annual_timeseries
from utils.constants   import (
    TEAL, AMBER, NAVY, NAVY2, BORDER, TEXT, MUTED,
    CLASS_COLORS, CLASS_ORDER,
)
from utils.styles import apply_dark_theme

# ── Custom CSS — Palantir-like dark UI ────────────────────────────────────────
apply_dark_theme("home")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-size:1.8rem;'>🌦</div>
        <div style='font-size:1rem; font-weight:600; color:#E8EAED;'>Moldova Drought</div>
        <div style='font-size:0.75rem; color:#8A9BAE;'>Intelligence Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Navigate**")
    st.page_link("pages/1_Station_Explorer.py",     label="🗺️  Station Explorer")
    st.page_link("pages/2_Climate_Change_Signal.py", label="📈  Climate Change Signal")
    st.page_link("pages/3_Drought_Risk_Map.py",      label="⚠️  Drought Risk Map")
    st.page_link("pages/4_ML_Model_Explorer.py",     label="🤖  ML Model Explorer")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#8A9BAE; line-height:1.7;'>
    <b style='color:#E8EAED'>Data</b><br>
    16 meteorological stations<br>
    Republic of Moldova<br>
    1961–2020 · 11,520 obs (synthetic)<br><br>
    <b style='color:#E8EAED'>ML pipeline</b><br>
    HistGradientBoosting<br>
    SMOTE 30% · Calibrated<br>
    Macro F1 = 0.85+<br><br>
    <b style='color:#E8EAED'>Research</b><br>
    PhD · CZU Prague<br>
    Int. J. of Climatology (2025)
    </div>
    """, unsafe_allow_html=True)


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex; align-items:center; gap:12px; margin-bottom:4px;'>
  <span style='font-size:2rem;'>🌦</span>
  <div>
    <h1 style='margin:0;'>Moldova Drought Intelligence Dashboard</h1>
    <p style='color:#8A9BAE; margin:0; font-size:0.9rem;'>
      60 years · 16 meteorological stations · WMO 7-class SPEI severity scheme
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── KPI row ───────────────────────────────────────────────────────────────────
ss  = load_station_summary()
ann = load_annual_timeseries()

full = ss[ss["period"] == "full"]
p1   = ss[ss["period"] == "p1"]
p2   = ss[ss["period"] == "p2"]

mean_precip_full  = full["precip_annual_mean"].mean()
mean_precip_p1    = p1["precip_annual_mean"].mean()
mean_precip_p2    = p2["precip_annual_mean"].mean()
mean_temp_full    = full["t_med_mean"].mean()
mean_temp_p1      = p1["t_med_mean"].mean()
mean_temp_p2      = p2["t_med_mean"].mean()
mean_pct_dry_full = full["pct_any_dry"].mean()
mean_pct_dry_p1   = p1["pct_any_dry"].mean()
mean_pct_dry_p2   = p2["pct_any_dry"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Stations",            "16",         "Moldova, 1961–2020")
c2.metric("Mean annual precip",  f"{mean_precip_full:.0f} mm",
          f"P2 {mean_precip_p2:.0f} vs P1 {mean_precip_p1:.0f} mm")
c3.metric("Mean temperature",    f"{mean_temp_full:.1f} °C",
          f"+{mean_temp_p2 - mean_temp_p1:.2f} °C (P2 vs P1)")
c4.metric("Dry months (any)",    f"{mean_pct_dry_full:.1f}%",
          f"P2 {mean_pct_dry_p2:.1f}% vs P1 {mean_pct_dry_p1:.1f}%")
c5.metric("Publications",        "6+",         "Int. J. Climatology 2025")

st.markdown("---")

# ── Page navigation cards ─────────────────────────────────────────────────────
st.markdown("## 🧭 Explore the Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class='nav-card'>
        <h3>🗺️ Station Explorer</h3>
        <p>Interactive dark map of Moldova with all 16 meteorological stations.
        Click any station to view its climatogram, annual precipitation sums,
        and drought class frequency for 1961–1990 vs 1991–2020.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='nav-card'>
        <h3>⚠️ Drought Risk Map</h3>
        <p>Composite Drought Risk Score per administrative region, computed from
        frequency, duration and intensity of drought events. Adjustable severity
        weights and period selector.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='nav-card'>
        <h3>📈 Climate Change Signal</h3>
        <p>Two-period comparison (1961–1990 vs 1991–2020): temperature trends,
        precipitation variability, SPEI-3 running mean, and shift in drought
        class frequencies across all stations.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='nav-card'>
        <h3>🤖 ML Model Explorer</h3>
        <p>Explore the HistGradientBoosting classifier (Macro F1 = 0.85+).
        Per-class F1 heatmap, confusion matrix, feature importances.
        Interactive input → calibrated 7-class probability output.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Dataset overview mini-chart ───────────────────────────────────────────────
st.markdown("## 📊 Dataset overview")
col_a, col_b = st.columns([3, 2])

with col_a:
    # All-station mean annual precipitation with period bands
    ann_all = ann.groupby("year")["precip_annual_sum"].mean().reset_index()
    colors  = ["rgba(26,169,154,0.65)" if y <= 1990
               else "rgba(232,146,58,0.65)" for y in ann_all["year"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ann_all["year"], y=ann_all["precip_annual_sum"],
        marker_color=colors, name="Mean annual precip",
        hovertemplate="<b>%{x}</b><br>%{y:.0f} mm<extra></extra>",
    ))
    p1_m = ann_all[ann_all.year <= 1990]["precip_annual_sum"].mean()
    p2_m = ann_all[ann_all.year >= 1991]["precip_annual_sum"].mean()
    fig.add_hline(y=p1_m, line_color=TEAL, line_dash="dot", line_width=1,
                  annotation_text=f"P1: {p1_m:.0f} mm",
                  annotation_font_color=TEAL)
    fig.add_hline(y=p2_m, line_color=AMBER, line_dash="dot", line_width=1,
                  annotation_text=f"P2: {p2_m:.0f} mm",
                  annotation_font_color=AMBER,
                  annotation_position="bottom right")
    fig.add_vline(x=1990.5, line_color=MUTED, line_dash="dot", line_width=1)
    fig.update_layout(
        paper_bgcolor=NAVY2, plot_bgcolor=NAVY,
        font=dict(color=TEXT, size=11),
        title=dict(text="<b>All-station mean annual precipitation (mm/year)</b>",
                   font=dict(size=12, color=TEXT)),
        xaxis=dict(gridcolor=BORDER, dtick=10, title="Year"),
        yaxis=dict(gridcolor=BORDER, title="mm/year"),
        margin=dict(l=50, r=20, t=40, b=40),
        height=280, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    # Drought class donut — all stations full period
    class_cols = {
        "Extremely Dry":  "pct_extremely_dry",
        "Severely Dry":   "pct_severely_dry",
        "Moderately Dry": "pct_moderately_dry",
        "Normal":         "pct_normal",
        "Moderately Wet": "pct_moderately_wet",
        "Severely Wet":   "pct_severely_wet",
        "Extremely Wet":  "pct_extremely_wet",
    }
    vals   = [full[v].mean() for v in class_cols.values()]
    colors = [CLASS_COLORS[k] for k in class_cols.keys()]
    labels = list(class_cols.keys())

    fig2 = go.Figure(go.Pie(
        labels=labels, values=vals,
        hole=0.5,
        marker=dict(colors=colors, line=dict(color=NAVY2, width=1.5)),
        textinfo="percent",
        textfont=dict(size=10, color=TEXT),
        sort=False, direction="clockwise",
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig2.update_layout(
        paper_bgcolor=NAVY2,
        font=dict(color=TEXT, size=11),
        title=dict(text="<b>7-class distribution (all stations, 1961–2020)</b>",
                   font=dict(size=12, color=TEXT)),
        legend=dict(
            x=1.02, y=0.5, font=dict(size=9),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=120, t=40, b=10),
        height=280,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#8A9BAE; font-size:0.75rem; padding:10px 0;'>
Tudor Trifan · PhD Candidate, Czech University of Life Sciences Prague ·
<a href='https://tudor-trifan.github.io' style='color:#1AA99A;'>Portfolio</a> ·
<a href='https://github.com/Trifan97' style='color:#1AA99A;'>GitHub</a> ·
<a href='https://www.linkedin.com/in/tudor-trifan/' style='color:#1AA99A;'>LinkedIn</a>
</div>
""", unsafe_allow_html=True)
