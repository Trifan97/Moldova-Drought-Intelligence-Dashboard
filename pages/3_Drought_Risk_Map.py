"""
pages/3_Drought_Risk_Map.py
Composite Drought Risk Score (DRS) per station.
Choropleth-style scatter map + ranked risk table.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pydeck as pdk

from utils.data_loader  import (
    load_station_summary, load_annual_timeseries,
    load_moldova_boundary, load_master_dataset,
)
from utils.constants    import (
    TEAL, AMBER, NAVY, NAVY2, BORDER, TEXT, MUTED,
    CLASS_COLORS, STATION_COORDS,
    MAP_CENTER_LAT, MAP_CENTER_LON, MAP_ZOOM,
)
from utils.styles import apply_dark_theme

st.set_page_config(
    page_title="Drought Risk Map — Moldova Drought",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme("drought_risk")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def compute_drs(w_freq, w_dur, w_int, period_key):
    """
    Compute Drought Risk Score per station.
    DRS = w_freq·Frequency + w_dur·Duration + w_int·Intensity
    All components normalised 0–1 before weighting.
    """
    df = load_master_dataset()
    ss = load_station_summary()

    y1, y2 = {"full": (1961,2020), "p1": (1961,1990), "p2": (1991,2020)}[period_key]
    sub = df[df.year.between(y1, y2)].copy()

    rows = []
    for station in sorted(STATION_COORDS.keys()):
        st_df = sub[sub.station == station].sort_values(["year","month"])
        if len(st_df) == 0:
            continue

        # Frequency: % months with any drought (SPEI-3 < -1.0)
        is_dry = st_df.spei_3 < -1.0
        freq = is_dry.mean() * 100

        # Duration: mean length of consecutive dry spells
        dry_seq = []
        count = 0
        for v in is_dry:
            if v:
                count += 1
            else:
                if count > 0:
                    dry_seq.append(count)
                count = 0
        if count > 0:
            dry_seq.append(count)
        mean_dur = np.mean(dry_seq) if dry_seq else 0

        # Intensity: mean absolute SPEI during dry periods
        dry_vals = st_df[is_dry].spei_3.abs()
        mean_int = dry_vals.mean() if len(dry_vals) > 0 else 0

        # Annual precip stats for context
        ann_sums = st_df.groupby("year")["precip"].sum()

        # Worst year (min annual sum)
        worst_year = int(ann_sums.idxmin()) if len(ann_sums) > 0 else None

        rows.append({
            "station": station,
            "lat": STATION_COORDS[station][0],
            "lon": STATION_COORDS[station][1],
            "freq": round(freq, 2),
            "mean_duration": round(mean_dur, 2),
            "mean_intensity": round(mean_int, 3),
            "worst_year": worst_year,
            "precip_mean": round(ann_sums.mean(), 1),
            "precip_std": round(ann_sums.std(), 1),
        })

    drs_df = pd.DataFrame(rows)

    # Normalise 0–1
    for col in ["freq", "mean_duration", "mean_intensity"]:
        lo, hi = drs_df[col].min(), drs_df[col].max()
        drs_df[f"{col}_norm"] = (drs_df[col] - lo) / (hi - lo + 1e-9)

    drs_df["drs"] = (
        w_freq * drs_df["freq_norm"] +
        w_dur  * drs_df["mean_duration_norm"] +
        w_int  * drs_df["mean_intensity_norm"]
    )
    drs_df["drs"] = drs_df["drs"].round(4)

    # Risk tier based on quartiles
    q = drs_df["drs"].quantile([0.25, 0.50, 0.75]).values
    def tier(v):
        if v >= q[2]:   return "High"
        elif v >= q[1]: return "Elevated"
        elif v >= q[0]: return "Moderate"
        else:           return "Low"
    drs_df["risk_tier"] = drs_df["drs"].apply(tier)

    TIER_COLORS = {
        "High":     [139,   0,   0, 220],
        "Elevated": [212, 105,  58, 200],
        "Moderate": [244, 164,  96, 170],
        "Low":      [ 74, 155, 111, 150],
    }
    TIER_HEX = {
        "High": "#8B0000", "Elevated": "#D4693A",
        "Moderate": "#F4A460", "Low": "#4a9b6f",
    }
    drs_df["color"]    = drs_df["risk_tier"].map(TIER_COLORS)
    drs_df["color_hex"]= drs_df["risk_tier"].map(TIER_HEX)
    return drs_df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚠️ Drought Risk Map")
    st.markdown("---")
    st.markdown("**Analysis period**")
    period_map = {
        "Full period (1961–2020)": "full",
        "Period 1 (1961–1990)":   "p1",
        "Period 2 (1991–2020)":   "p2",
    }
    period_label = st.selectbox("Period", list(period_map.keys()), index=0)
    period_key   = period_map[period_label]

    st.markdown("---")
    st.markdown("**DRS component weights**")
    st.caption("Must sum to 1.0")

    w_freq = st.slider("Frequency weight",  0.0, 1.0, 0.40, 0.05)
    w_dur  = st.slider("Duration weight",   0.0, 1.0, 0.35, 0.05)
    w_int  = st.slider("Intensity weight",  0.0, 1.0,
                       round(1.0 - w_freq - w_dur, 2), 0.05,
                       disabled=True,
                       help="Auto = 1 - Frequency - Duration")
    w_int  = max(0.0, round(1.0 - w_freq - w_dur, 2))

    total = w_freq + w_dur + w_int
    if abs(total - 1.0) > 0.01:
        st.warning(f"Weights sum to {total:.2f} — adjust sliders")
    else:
        st.success(f"✓ Weights sum to {total:.2f}")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#8A9BAE; line-height:1.8;'>
    <b style='color:#E8EAED;'>DRS formula</b><br>
    DRS = w₁·Freq + w₂·Dur + w₃·Int<br>
    All components normalised 0–1<br><br>
    <b style='color:#E8EAED;'>Risk tiers</b><br>
    <span style='color:#8B0000;'>■</span> High    — top 25%<br>
    <span style='color:#D4693A;'>■</span> Elevated — 50–75%<br>
    <span style='color:#F4A460;'>■</span> Moderate — 25–50%<br>
    <span style='color:#4a9b6f;'>■</span> Low      — bottom 25%
    </div>
    """, unsafe_allow_html=True)

# ── Compute DRS ───────────────────────────────────────────────────────────────
drs_df = compute_drs(w_freq, w_dur, w_int, period_key)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>
  <span style='font-size:1.6rem;'>⚠️</span>
  <div>
    <h2 style='margin:0;font-size:1.3rem!important;'>Drought Risk Map</h2>
    <p style='color:#8A9BAE;margin:0;font-size:0.85rem;'>
      Composite DRS per station · {period_label} ·
      Weights: Freq={w_freq:.0%} · Dur={w_dur:.0%} · Int={w_int:.0%}
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("High risk stations",
          f"{(drs_df.risk_tier=='High').sum()}",
          "DRS top quartile")
k2.metric("Highest DRS station",
          drs_df.loc[drs_df.drs.idxmax(), "station"],
          f"DRS = {drs_df.drs.max():.4f}")
k3.metric("Lowest DRS station",
          drs_df.loc[drs_df.drs.idxmin(), "station"],
          f"DRS = {drs_df.drs.min():.4f}")
k4.metric("Mean dry frequency",
          f"{drs_df.freq.mean():.1f}%",
          f"Range: {drs_df.freq.min():.1f}–{drs_df.freq.max():.1f}%")

st.markdown("---")

# ── Map + table layout ────────────────────────────────────────────────────────
map_col, tbl_col = st.columns([1.5, 1], gap="medium")

with map_col:
    boundary_geojson = load_moldova_boundary()

    # Scale radius by DRS
    drs_df["radius"] = (drs_df["drs"] / drs_df["drs"].max() * 25_000 + 8_000).astype(int)

    station_layer = pdk.Layer(
        "ScatterplotLayer",
        data=drs_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_line_color=[255, 255, 255, 180],
        get_radius="radius",
        radius_min_pixels=8,
        radius_max_pixels=28,
        stroked=True,
        get_line_width=2,
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 80],
    )
    boundary_layer = pdk.Layer(
        "GeoJsonLayer",
        data=boundary_geojson,
        get_fill_color=[22, 47, 71, 50],
        get_line_color=[26, 169, 154, 180],
        line_width_min_pixels=2,
        pickable=False,
    )
    label_layer = pdk.Layer(
        "TextLayer",
        data=drs_df,
        get_position=["lon", "lat"],
        get_text="station",
        get_size=12,
        get_color=[232, 234, 237, 200],
        get_pixel_offset=[0, -26],
        pickable=False,
    )

    deck = pdk.Deck(
        layers=[boundary_layer, station_layer, label_layer],
        initial_view_state=pdk.ViewState(
            latitude=MAP_CENTER_LAT, longitude=MAP_CENTER_LON,
            zoom=MAP_ZOOM, pitch=0,
        ),
        map_style="dark",
        tooltip={
            "html": (
                "<b>{station}</b><br>"
                "DRS: <b>{drs}</b> · Risk: <b>{risk_tier}</b><br>"
                "Dry freq: {freq}%<br>"
                "Mean duration: {mean_duration} months<br>"
                "Mean intensity: {mean_intensity}<br>"
                "Worst year: {worst_year}<br>"
                "Mean annual precip: {precip_mean} mm"
            ),
            "style": {
                "backgroundColor": "#0F2640",
                "border": "1px solid #1AA99A",
                "borderRadius": "6px",
                "color": "#E8EAED",
                "fontSize": "12px",
                "padding": "8px 12px",
            },
        },
    )

    st.pydeck_chart(deck, use_container_width=True, height=500)

    # Legend
    st.markdown("""
    <div style='display:flex;gap:16px;margin-top:6px;font-size:0.78rem;color:#8A9BAE;'>
      <span><span style='color:#8B0000;font-size:1rem;'>●</span> High</span>
      <span><span style='color:#D4693A;font-size:1rem;'>●</span> Elevated</span>
      <span><span style='color:#F4A460;font-size:1rem;'>●</span> Moderate</span>
      <span><span style='color:#4a9b6f;font-size:1rem;'>●</span> Low</span>
      <span style='margin-left:8px;color:#8A9BAE;'>Dot size ∝ DRS magnitude</span>
    </div>
    """, unsafe_allow_html=True)

with tbl_col:
    st.markdown("### 📋 Station Risk Rankings")
    ranked = drs_df.sort_values("drs", ascending=False)[
        ["station","drs","risk_tier","freq","mean_duration","mean_intensity","worst_year"]
    ].reset_index(drop=True)
    ranked.index = ranked.index + 1

    # Styled display
    for _, row in ranked.iterrows():
        tier = row.risk_tier
        tier_colors = {
            "High": "#8B0000", "Elevated": "#D4693A",
            "Moderate": "#F4A460", "Low": "#4a9b6f",
        }
        c = tier_colors.get(tier, TEAL)
        st.markdown(f"""
        <div class='risk-card' style='border-left-color:{c};'>
          <div style='display:flex;justify-content:space-between;align-items:center;'>
            <span style='font-weight:600;color:#E8EAED;'>{int(ranked.index[ranked.station==row.station][0])}. {row.station}</span>
            <span style='color:{c};font-size:0.8rem;font-weight:600;'>{tier}</span>
          </div>
          <div style='font-size:0.75rem;color:#8A9BAE;margin-top:4px;line-height:1.8;'>
            DRS: <b style='color:#E8EAED;'>{row.drs:.4f}</b> &nbsp;|&nbsp;
            Dry: <b style='color:#E8EAED;'>{row.freq:.1f}%</b> &nbsp;|&nbsp;
            Dur: <b style='color:#E8EAED;'>{row.mean_duration:.1f} mo</b><br>
            Intensity: {row.mean_intensity:.3f} &nbsp;|&nbsp;
            Worst year: {int(row.worst_year) if row.worst_year else '—'}
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── DRS component breakdown chart ────────────────────────────────────────────
st.markdown("## 📊 DRS Component Breakdown")

fig_comp = go.Figure()
drs_sorted = drs_df.sort_values("drs", ascending=True)

for col, label, color in [
    ("freq_norm",           f"Frequency (w={w_freq:.0%})",  "#D4693A"),
    ("mean_duration_norm",  f"Duration (w={w_dur:.0%})",    "#F4A460"),
    ("mean_intensity_norm", f"Intensity (w={w_int:.0%})",   "#8B0000"),
]:
    fig_comp.add_trace(go.Bar(
        name=label,
        y=drs_sorted.station,
        x=drs_sorted[col] * (
            w_freq if "freq" in col
            else w_dur if "dur" in col
            else w_int
        ),
        orientation="h",
        marker_color=color,
        hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.3f}}<extra></extra>",
    ))

fig_comp.update_layout(
    paper_bgcolor=NAVY2, plot_bgcolor=NAVY,
    font=dict(color=TEXT, size=11),
    barmode="stack",
    title=dict(text="<b>DRS component breakdown per station (stacked, lowest → highest risk)</b>",
               font=dict(size=13, color=TEXT)),
    xaxis=dict(title="Weighted DRS contribution", gridcolor=BORDER),
    yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),
    legend=dict(x=0.6, y=0.02, bgcolor="rgba(15,38,64,0.8)",
                bordercolor=BORDER, borderwidth=1),
    margin=dict(l=120, r=20, t=40, b=40),
    height=420,
    hoverlabel=dict(bgcolor=NAVY2, bordercolor=TEAL, font=dict(color=TEXT)),
)
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("""
<div class='insight-box'>
📌 <b>How to interpret the DRS:</b>
Frequency = % months with SPEI-3 below −1.0 (any drought).
Duration = mean length of consecutive drought months.
Intensity = mean |SPEI-3| during drought periods (deeper = more intense).
All three components are normalised 0–1 before weighting.
Adjust the sidebar sliders to explore how different weighting schemes
affect the risk ranking — useful for comparing agricultural vs. water-resource risk perspectives.
</div>
""", unsafe_allow_html=True)
