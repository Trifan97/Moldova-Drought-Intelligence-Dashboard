"""
pages/2_Climate_Change_Signal.py
Two-period climate comparison: 1961–1990 vs 1991–2020
Temperature trends, precipitation variability, SPEI, drought class shifts.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader  import (
    load_station_summary, load_annual_timeseries, load_climatology,
)
from utils.constants    import (
    TEAL, AMBER, NAVY, NAVY2, BORDER, TEXT, MUTED,
    CLASS_COLORS, CLASS_ORDER, STATION_COORDS, MONTH_LABELS,
)
from utils.styles import apply_dark_theme

st.set_page_config(
    page_title="Climate Change Signal — Moldova Drought",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme()

# ── Helper ────────────────────────────────────────────────────────────────────
def layout(**kw):
    base = dict(
        paper_bgcolor=NAVY2, plot_bgcolor=NAVY,
        font=dict(color=TEXT, size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        hoverlabel=dict(bgcolor=NAVY2, bordercolor=TEAL, font=dict(color=TEXT)),
    )
    base.update(kw)
    return base

# ── Load data ─────────────────────────────────────────────────────────────────
ss  = load_station_summary()
ann = load_annual_timeseries()
clim = load_climatology()

p1_ss = ss[ss.period == "p1"]
p2_ss = ss[ss.period == "p2"]
full_ss = ss[ss.period == "full"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 Climate Change Signal")
    st.markdown("---")

    all_stations = sorted(STATION_COORDS.keys())
    station_sel = st.multiselect(
        "Stations to highlight",
        options=all_stations,
        default=["Chisinau", "Cahul", "Briceni"],
        max_selections=6,
        help="Highlight specific stations in scatter plots",
    )

    st.markdown("---")
    show_regression = st.toggle("Show trend lines", value=True)
    show_all_stations = st.toggle("Show all stations", value=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#8A9BAE; line-height:1.7;'>
    <b style='color:#E8EAED;'>Reference periods</b><br>
    P1: 1961–1990 (WMO standard)<br>
    P2: 1991–2020 (WMO standard)<br><br>
    Shift = P2 minus P1
    </div>
    """, unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>
  <span style='font-size:1.6rem;'>📈</span>
  <div>
    <h2 style='margin:0;font-size:1.3rem!important;'>Climate Change Signal</h2>
    <p style='color:#8A9BAE;margin:0;font-size:0.85rem;'>
      Two WMO reference periods · 1961–1990 vs 1991–2020 · All 16 stations
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI row ───────────────────────────────────────────────────────────────────
dt_mean = (p2_ss.t_med_mean - p1_ss.set_index("station").loc[p2_ss.station].t_med_mean.values).mean()
dp_mean = (p2_ss.precip_annual_mean - p1_ss.set_index("station").loc[p2_ss.station].precip_annual_mean.values).mean()
dd_mean = (p2_ss.pct_any_dry - p1_ss.set_index("station").loc[p2_ss.station].pct_any_dry.values).mean()
p1_temp_mean = p1_ss.t_med_mean.mean()
p2_temp_mean = p2_ss.t_med_mean.mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Temp shift (P2–P1)", f"{dt_mean:+.2f} °C",
          f"P1: {p1_temp_mean:.1f} °C → P2: {p2_temp_mean:.1f} °C")
k2.metric("Precip shift (P2–P1)", f"{dp_mean:+.1f} mm/yr",
          f"P1: {p1_ss.precip_annual_mean.mean():.0f} → P2: {p2_ss.precip_annual_mean.mean():.0f} mm")
k3.metric("Dry months shift", f"{dd_mean:+.2f}%",
          f"P1: {p1_ss.pct_any_dry.mean():.1f}% → P2: {p2_ss.pct_any_dry.mean():.1f}%")
k4.metric("Stations warming", f"{(p2_ss.t_med_mean.values > p1_ss.t_med_mean.values).sum()}/16",
          "All show positive temperature trend")

# Key insight
st.markdown(f"""
<div class='insight-box'>
⚡ <b>Key finding:</b> All 16 Moldovan stations warmed between the two reference periods,
with a mean increase of <b>{dt_mean:+.2f} °C</b>. The warming signal is strongest in summer months
(June–August), amplifying evapotranspiration demand and contributing to more frequent
Moderately Dry conditions in the south. Mean annual precipitation shows modest change
(<b>{dp_mean:+.1f} mm/yr</b>) but with <b>higher inter-annual variability</b> in P2 — the dry years
are drier and the wet years are wetter.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Section 1: Temperature change map ─────────────────────────────────────────
st.markdown("## 🌡️ Temperature Change: P1 → P2")

col1, col2 = st.columns([3, 2])

with col1:
    # Scatter: lat vs temp delta, sized by magnitude
    merged = p1_ss[["station","t_med_mean","precip_annual_mean","pct_any_dry"]].merge(
        p2_ss[["station","t_med_mean","precip_annual_mean","pct_any_dry"]],
        on="station", suffixes=("_p1","_p2")
    )
    merged["dt"] = merged.t_med_mean_p2 - merged.t_med_mean_p1
    merged["dp"] = merged.precip_annual_mean_p2 - merged.precip_annual_mean_p1
    merged["lat"] = merged.station.map({s: c[0] for s,c in STATION_COORDS.items()})
    merged["lon"] = merged.station.map({s: c[1] for s,c in STATION_COORDS.items()})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged.lon, y=merged.lat,
        mode="markers+text",
        marker=dict(
            size=merged.dt.abs() * 14 + 8,
            color=merged.dt,
            colorscale=[[0,"#3A78B5"],[0.5,"#F4A460"],[1,"#8B0000"]],
            colorbar=dict(title=dict(text="ΔT (°C)", font=dict(color=TEXT))),
            cmin=0, cmax=merged.dt.max()*1.1,
            line=dict(color="white", width=1),
        ),
        text=merged.station,
        textposition="top center",
        textfont=dict(size=10, color=TEXT),
        customdata=merged[["t_med_mean_p1","t_med_mean_p2","dt"]].values,
        hovertemplate=(
            "<b>%{text}</b><br>"
            "P1: %{customdata[0]:.2f} °C<br>"
            "P2: %{customdata[1]:.2f} °C<br>"
            "Δ: <b>%{customdata[2]:+.2f} °C</b>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        **layout(
            title=dict(text="<b>Temperature change by station location (P2 – P1)</b>",
                       font=dict(size=13, color=TEXT)),
            xaxis=dict(title="Longitude", gridcolor=BORDER),
            yaxis=dict(title="Latitude", gridcolor=BORDER),
            height=380,
            showlegend=False,
        )
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Bar chart: temp P1 vs P2 per station, sorted by latitude (N→S)
    merged_s = merged.sort_values("lat", ascending=False)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        name="P1 (1961–1990)", y=merged_s.station,
        x=merged_s.t_med_mean_p1,
        orientation="h",
        marker_color=TEAL, marker_opacity=0.6,
        hovertemplate="%{y}: %{x:.2f} °C<extra>P1</extra>",
    ))
    fig2.add_trace(go.Bar(
        name="P2 (1991–2020)", y=merged_s.station,
        x=merged_s.t_med_mean_p2,
        orientation="h",
        marker_color=AMBER, marker_opacity=0.9,
        hovertemplate="%{y}: %{x:.2f} °C<extra>P2</extra>",
    ))
    fig2.update_layout(
        **layout(
            title=dict(text="<b>Mean temperature by station (N → S)</b>",
                       font=dict(size=13, color=TEXT)),
            barmode="overlay",
            xaxis=dict(title="°C", gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),
            legend=dict(x=0.5, y=0.01, bgcolor="rgba(15,38,64,0.8)",
                        bordercolor=BORDER, borderwidth=1),
            height=380,
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Section 2: Precipitation change ──────────────────────────────────────────
st.markdown("## 🌧️ Precipitation: Annual Sums & Variability")

col3, col4 = st.columns(2)

with col3:
    # All-station mean annual precip timeseries
    ann_mean = ann.groupby("year")["precip_annual_sum"].mean().reset_index()
    ann_std  = ann.groupby("year")["precip_annual_sum"].std().reset_index()

    fig3 = go.Figure()
    # Uncertainty band
    fig3.add_trace(go.Scatter(
        x=list(ann_mean.year) + list(ann_mean.year[::-1]),
        y=list(ann_mean.precip_annual_sum + ann_std.precip_annual_sum) +
          list((ann_mean.precip_annual_sum - ann_std.precip_annual_sum)[::-1]),
        fill="toself",
        fillcolor="rgba(26,169,154,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1 std (cross-station)",
        showlegend=True,
        hoverinfo="skip",
    ))
    # Period means
    p1_ann_mean = ann_mean[ann_mean.year <= 1990].precip_annual_sum.mean()
    p2_ann_mean = ann_mean[ann_mean.year >= 1991].precip_annual_sum.mean()
    # Bar
    colors = ["rgba(26,169,154,0.7)" if y <= 1990 else "rgba(232,146,58,0.7)"
              for y in ann_mean.year]
    fig3.add_trace(go.Bar(
        x=ann_mean.year, y=ann_mean.precip_annual_sum,
        marker_color=colors, name="Annual mean",
        hovertemplate="<b>%{x}</b>: %{y:.0f} mm<extra></extra>",
    ))
    fig3.add_hline(y=p1_ann_mean, line_color=TEAL, line_dash="dot",
                   annotation_text=f"P1 mean: {p1_ann_mean:.0f} mm",
                   annotation_font_color=TEAL)
    fig3.add_hline(y=p2_ann_mean, line_color=AMBER, line_dash="dot",
                   annotation_text=f"P2 mean: {p2_ann_mean:.0f} mm",
                   annotation_font_color=AMBER,
                   annotation_position="bottom right")
    fig3.add_vline(x=1990.5, line_color=MUTED, line_dash="dot", line_width=1.2)
    if show_regression:
        for mask, color, label in [
            (ann_mean.year <= 1990, TEAL, "P1 trend"),
            (ann_mean.year >= 1991, AMBER, "P2 trend"),
        ]:
            sub = ann_mean[mask]
            c = np.polyfit(sub.year, sub.precip_annual_sum, 1)
            fig3.add_trace(go.Scatter(
                x=sub.year, y=np.polyval(c, sub.year),
                mode="lines", line=dict(color=color, width=1.5, dash="longdash"),
                name=f"{label} ({c[0]*10:+.1f} mm/dec)", showlegend=True,
            ))
    fig3.update_layout(
        **layout(
            title=dict(text="<b>All-station mean annual precipitation (mm/year)</b>",
                       font=dict(size=13, color=TEXT)),
            xaxis=dict(title="Year", gridcolor=BORDER, dtick=10),
            yaxis=dict(title="mm/year", gridcolor=BORDER),
            height=320,
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    # Precipitation variability: P1 vs P2 std per station
    p1_std = ann[ann.year.between(1961, 1990)].groupby("station")["precip_annual_sum"].std()
    p2_std = ann[ann.year.between(1991, 2020)].groupby("station")["precip_annual_sum"].std()
    std_df = pd.DataFrame({"P1 std": p1_std, "P2 std": p2_std}).reset_index()
    std_df["delta_std"] = std_df["P2 std"] - std_df["P1 std"]
    std_df = std_df.sort_values("delta_std", ascending=True)

    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        y=std_df.station, x=std_df.delta_std,
        orientation="h",
        marker_color=[AMBER if v > 0 else TEAL for v in std_df.delta_std],
        hovertemplate="%{y}: %{x:+.1f} mm<extra>ΔStd</extra>",
    ))
    fig4.add_vline(x=0, line_color=MUTED, line_width=1)
    fig4.update_layout(
        **layout(
            title=dict(
                text="<b>Change in annual precipitation variability (σ P2 – σ P1)</b>",
                font=dict(size=13, color=TEXT)),
            xaxis=dict(title="Δ Std (mm)", gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),
            showlegend=False,
            height=320,
        )
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# ── Section 3: Monthly precipitation shift ────────────────────────────────────
st.markdown("## 📅 Monthly Precipitation Shift (P2 – P1)")

# All-station mean monthly precip by period
clim_p1 = clim[clim.period == "p1"].groupby("month")["precip_monthly_mean"].mean()
clim_p2 = clim[clim.period == "p2"].groupby("month")["precip_monthly_mean"].mean()
clim_dt_p1 = clim[clim.period == "p1"].groupby("month")["t_med_mean"].mean()
clim_dt_p2 = clim[clim.period == "p2"].groupby("month")["t_med_mean"].mean()

delta_p = clim_p2 - clim_p1
delta_t = clim_dt_p2 - clim_dt_p1

col5, col6 = st.columns(2)

with col5:
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(
        name="P1 (1961–1990)",
        x=MONTH_LABELS, y=clim_p1.values,
        marker_color="rgba(26,169,154,0.55)", marker_line_width=0,
    ))
    fig5.add_trace(go.Bar(
        name="P2 (1991–2020)",
        x=MONTH_LABELS, y=clim_p2.values,
        marker_color="rgba(232,146,58,0.9)", marker_line_width=0,
    ))
    fig5.update_layout(
        **layout(
            title=dict(text="<b>Mean monthly precipitation: P1 vs P2 (all stations)</b>",
                       font=dict(size=13, color=TEXT)),
            barmode="group",
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(title="mm", gridcolor=BORDER, rangemode="tozero"),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(15,38,64,0.8)",
                        bordercolor=BORDER, borderwidth=1),
            height=300,
        )
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    # Delta bars coloured by sign
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(
        x=MONTH_LABELS,
        y=delta_p.values,
        marker_color=[TEAL if v >= 0 else AMBER for v in delta_p.values],
        name="ΔPrecip",
        hovertemplate="<b>%{x}</b><br>Δ: %{y:+.1f} mm<extra></extra>",
    ))
    fig6.add_hline(y=0, line_color=MUTED, line_width=1)
    fig6.add_trace(go.Scatter(
        x=MONTH_LABELS, y=delta_t.values,
        mode="lines+markers",
        line=dict(color="#D4693A", width=2),
        marker=dict(size=6),
        name="ΔTemp (°C)",
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>ΔT: %{y:+.2f} °C<extra></extra>",
    ))
    fig6.update_layout(
        **layout(
            title=dict(text="<b>Monthly change P2 – P1 (precip bars, temp line)</b>",
                       font=dict(size=13, color=TEXT)),
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(title="ΔPrecip (mm)", gridcolor=BORDER),
            yaxis2=dict(
                title=dict(text="ΔTemp (°C)", font=dict(color="#D4693A")),
                tickfont=dict(color="#D4693A"),
                overlaying="y", side="right",
                gridcolor="rgba(0,0,0,0)",
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(15,38,64,0.8)",
                        bordercolor=BORDER, borderwidth=1),
            height=300,
        )
    )
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("---")

# ── Section 4: Drought class frequency shift ──────────────────────────────────
st.markdown("## 🌵 Drought Class Frequency Shift (P2 – P1)")

col_labels = {
    "pct_extremely_dry": "Extremely Dry",
    "pct_severely_dry":  "Severely Dry",
    "pct_moderately_dry":"Moderately Dry",
    "pct_normal":        "Normal",
    "pct_moderately_wet":"Moderately Wet",
    "pct_severely_wet":  "Severely Wet",
    "pct_extremely_wet": "Extremely Wet",
}

# Merge P1 and P2 per station
shift_df = p1_ss[["station"] + list(col_labels.keys())].merge(
    p2_ss[["station"] + list(col_labels.keys())],
    on="station", suffixes=("_p1","_p2")
)
shift_df["lat"] = shift_df.station.map({s: c[0] for s,c in STATION_COORDS.items()})
shift_df = shift_df.sort_values("lat", ascending=False)

# Delta per class
for col, label in col_labels.items():
    shift_df[f"delta_{col}"] = shift_df[f"{col}_p2"] - shift_df[f"{col}_p1"]

# Stacked diverging bar chart
fig7 = go.Figure()
for col, label in col_labels.items():
    delta_vals = shift_df[f"delta_{col}"].values
    fig7.add_trace(go.Bar(
        name=label,
        y=shift_df.station,
        x=delta_vals,
        orientation="h",
        marker_color=CLASS_COLORS[label],
        hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:+.2f}}%<extra></extra>",
    ))

fig7.add_vline(x=0, line_color=MUTED, line_width=1.5)
fig7.update_layout(
    **layout(
        title=dict(
            text="<b>Change in drought class frequency per station (P2 – P1, % months)</b>",
            font=dict(size=13, color=TEXT)),
        barmode="relative",
        xaxis=dict(title="Change in % months (+ = more in P2)", gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),
        legend=dict(
            x=1.01, y=0.5,
            bgcolor="rgba(15,38,64,0.8)",
            bordercolor=BORDER, borderwidth=1,
            font=dict(size=10),
        ),
        height=520,
        margin=dict(l=120, r=160, t=40, b=40),
    )
)
st.plotly_chart(fig7, use_container_width=True)

# Insight summary
top_warming = merged.nlargest(3, "dt")[["station","dt"]].values
st.markdown(f"""
<div class='insight-box'>
📌 <b>Reading the chart:</b> bars extending right (positive) mean that class became
more frequent in P2 (1991–2020) vs P1 (1961–1990). Bars extending left mean it became
less frequent.<br><br>
🌡 Top warming stations: {top_warming[0][0]} (+{top_warming[0][1]:.2f}°C),
{top_warming[1][0]} (+{top_warming[1][1]:.2f}°C),
{top_warming[2][0]} (+{top_warming[2][1]:.2f}°C).<br>
Southern stations (Cahul, Comrat, Ceadir-Lunga) consistently show larger increases
in Moderately Dry frequency, consistent with the known north–south aridity gradient.
</div>
""", unsafe_allow_html=True)
