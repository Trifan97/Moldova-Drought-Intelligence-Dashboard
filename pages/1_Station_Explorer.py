"""
pages/1_Station_Explorer.py
Interactive PyDeck dark map + station climate panel with period comparison.
"""

import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader  import (
    load_station_summary, load_moldova_boundary,
    build_station_map_df, get_climatogram_data,
    get_drought_class_freq, get_annual_series,
    get_station_period_row,
)
from utils.map_builder  import build_deck
from utils.chart_builder import (
    climatogram, climatogram_comparison,
    annual_precip_chart, annual_temp_chart,
    drought_donut, period_comparison_bars,
    spei_timeseries,
)
from utils.constants import (
    PERIODS, PERIOD_LABELS, STATION_COORDS,
    TEAL, AMBER, NAVY2, BORDER, TEXT, MUTED, CLASS_COLORS,
)
from utils.styles import apply_dark_theme

st.set_page_config(
    page_title="Station Explorer — Moldova Drought",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme("station_explorer")

# ── Session state: persist selected station across reruns ─────────────────────
if "selected_station" not in st.session_state:
    st.session_state["selected_station"] = "Chisinau"

# ── Sidebar controls ──────────────────────────────────────────────────────────
_stations_sorted = sorted(STATION_COORDS.keys())

with st.sidebar:
    st.markdown("### 🗺️ Station Explorer")
    st.markdown("---")

    # Index driven by session state so map clicks keep the selectbox in sync
    selected_station = st.selectbox(
        "Select station",
        options=_stations_sorted,
        index=_stations_sorted.index(st.session_state["selected_station"]),
        help="Click on map OR select here",
    )
    # If the user changed the dropdown, persist it immediately
    if selected_station != st.session_state["selected_station"]:
        st.session_state["selected_station"] = selected_station

    st.markdown("**Analysis period**")
    period_label = st.radio(
        "Period",
        options=list(PERIODS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    period_key = PERIODS[period_label]

    st.markdown("---")
    st.markdown("**Map options**")
    show_labels  = st.toggle("Station labels",  value=True)
    show_comparison = st.toggle("Period comparison", value=False,
                                help="Show P1 vs P2 side-by-side charts")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#8A9BAE; line-height:1.7;'>
    <b style='color:#E8EAED;'>Colour legend</b><br>
    <span style='color:#1AA99A;'>■</span> Low dry frequency<br>
    <span style='color:#D4693A;'>■</span> High dry frequency<br><br>
    Click a station dot to inspect.<br>
    Hover for quick stats.
    </div>
    """, unsafe_allow_html=True)


# ── Load data ────────────────────────────────────────────────────────────────
boundary_geojson = load_moldova_boundary()
station_df       = build_station_map_df(period=period_key)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex; align-items:center; gap:10px; margin-bottom:6px;'>
  <span style='font-size:1.6rem;'>🗺️</span>
  <div>
    <h2 style='margin:0; font-size:1.3rem !important;'>Station Explorer</h2>
    <p style='color:#8A9BAE; margin:0; font-size:0.85rem;'>
      16 meteorological stations · Republic of Moldova · Click a station to inspect
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout: map left, panel right ────────────────────────────────────────────
map_col, panel_col = st.columns([1.4, 1], gap="medium")

with map_col:
    # Build and render deck
    deck = build_deck(
        station_df       = station_df,
        boundary_geojson = boundary_geojson,
        selected_station = selected_station,
        show_labels      = show_labels,
    )

    # pydeck_chart with click events
    event = st.pydeck_chart(
        deck,
        on_select="rerun",
        selection_mode="single-object",
        use_container_width=True,
        height=520,
    )

    # Handle map click → update session state and force a clean rerun so
    # the selectbox, map highlight, and panel all reflect the new station.
    if event and event.selection and event.selection.get("objects"):
        objs = event.selection["objects"]
        for layer_key, items in objs.items():
            if items:
                clicked_station = items[0].get("station")
                if clicked_station and clicked_station in STATION_COORDS:
                    if clicked_station != st.session_state["selected_station"]:
                        st.session_state["selected_station"] = clicked_station
                        st.rerun()

    # Colour scale legend
    st.markdown("""
    <div style='display:flex; align-items:center; gap:8px; margin-top:6px;
                font-size:0.75rem; color:#8A9BAE;'>
      <span>Low drought frequency</span>
      <div style='flex:1; height:6px; border-radius:3px;
           background: linear-gradient(to right, #1AA99A, #8B0000);'></div>
      <span>High drought frequency</span>
    </div>
    """, unsafe_allow_html=True)


with panel_col:
    # Always read from session state — this is the single source of truth
    # regardless of whether the update came from the dropdown or a map click.
    selected_station = st.session_state["selected_station"]

    row_full = get_station_period_row(selected_station, "full")
    row_p1   = get_station_period_row(selected_station, "p1")
    row_p2   = get_station_period_row(selected_station, "p2")
    row_sel  = get_station_period_row(selected_station, period_key)

    lat, lon = STATION_COORDS[selected_station]

    # Station header
    st.markdown(f"""
    <div style='background:#0F2640; border:1px solid #1E3550; border-radius:8px;
                padding:14px 16px; border-left:3px solid #1AA99A; margin-bottom:12px;'>
      <div style='font-size:1.1rem; font-weight:600; color:#E8EAED;'>
        📍 {selected_station}
      </div>
      <div style='font-size:0.8rem; color:#8A9BAE; margin-top:3px;'>
        {lat:.2f}°N, {lon:.2f}°E &nbsp;·&nbsp;
        Period: <b style='color:#1AA99A;'>{period_label}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Key stats row
    if row_sel is not None:
        ka, kb, kc = st.columns(3)

        # Precip delta P1 → P2
        if row_p1 is not None and row_p2 is not None:
            precip_delta = row_p2["precip_annual_mean"] - row_p1["precip_annual_mean"]
            temp_delta   = row_p2["t_med_mean"] - row_p1["t_med_mean"]
            dry_delta    = row_p2["pct_any_dry"] - row_p1["pct_any_dry"]
            precip_delta_str = f"{'↑' if precip_delta >= 0 else '↓'} {abs(precip_delta):.0f} mm P2 vs P1"
            temp_delta_str   = f"{'↑' if temp_delta >= 0 else '↓'} {abs(temp_delta):.2f} °C P2 vs P1"
            dry_delta_str    = f"{'↑' if dry_delta >= 0 else '↓'} {abs(dry_delta):.1f}% P2 vs P1"
        else:
            precip_delta_str = temp_delta_str = dry_delta_str = ""

        ka.metric("Annual precip",
                  f"{row_sel['precip_annual_mean']:.0f} mm",
                  precip_delta_str if period_key == "full" else "")
        kb.metric("Mean temp",
                  f"{row_sel['t_med_mean']:.1f} °C",
                  temp_delta_str if period_key == "full" else "")
        kc.metric("Dry months",
                  f"{row_sel['pct_any_dry']:.1f}%",
                  dry_delta_str if period_key == "full" else "")

    st.markdown("---")

    # ── Chart tabs ────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌡 Climatogram", "📊 Annual series", "🍩 Drought classes", "📅 P1 vs P2"
    ])

    clim_full = get_climatogram_data(selected_station, period_key)
    ann_full  = get_annual_series(selected_station, "full")

    with tab1:
        if not clim_full.empty:
            fig = climatogram(clim_full, selected_station, period_label)
            st.plotly_chart(fig, use_container_width=True)
            # Annual sum note
            if row_sel is not None:
                st.markdown(f"""
                <div class='info-box'>
                📌 Mean annual precipitation sum: <b>{row_sel['precip_annual_mean']:.0f} mm/year</b>
                &nbsp;(std: ±{row_sel['precip_annual_std']:.0f} mm)
                &nbsp;·&nbsp;Range: {row_sel['precip_annual_min']:.0f}–{row_sel['precip_annual_max']:.0f} mm
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No climatology data for this selection.")

    with tab2:
        if not ann_full.empty:
            fig_p = annual_precip_chart(ann_full, selected_station)
            st.plotly_chart(fig_p, use_container_width=True)
            fig_t = annual_temp_chart(ann_full, selected_station)
            st.plotly_chart(fig_t, use_container_width=True)
            fig_s = spei_timeseries(ann_full, selected_station)
            st.plotly_chart(fig_s, use_container_width=True)
        else:
            st.info("No annual timeseries data.")

    with tab3:
        freq = get_drought_class_freq(selected_station, period_key)
        if freq:
            fig_d = drought_donut(freq, selected_station, period_label)
            st.plotly_chart(fig_d, use_container_width=True)
            # Table
            freq_df = pd.DataFrame(
                {"Class": list(freq.keys()), "% months": list(freq.values())}
            ).sort_values("% months", ascending=False)
            st.dataframe(
                freq_df.style.format({"% months": "{:.1f}%"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("No drought class data.")

    with tab4:
        if row_p1 is not None and row_p2 is not None:
            # Summary comparison metrics

            m1, m2, m3 = st.columns(3)
            m1.metric("P1 precip", f"{row_p1['precip_annual_mean']:.0f} mm",
                      f"P2: {row_p2['precip_annual_mean']:.0f} mm")
            m2.metric("P1 temp",   f"{row_p1['t_med_mean']:.1f} °C",
                      f"P2: +{row_p2['t_med_mean'] - row_p1['t_med_mean']:.2f} °C")
            m3.metric("P1 dry%",   f"{row_p1['pct_any_dry']:.1f}%",
                      f"P2: {row_p2['pct_any_dry']:.1f}%")

            # Comparison climatogram
            clim_p1_df = get_climatogram_data(selected_station, "p1")
            clim_p2_df = get_climatogram_data(selected_station, "p2")
            if not clim_p1_df.empty and not clim_p2_df.empty:
                fig_cmp = climatogram_comparison(clim_p1_df, clim_p2_df,
                                                  selected_station)
                st.plotly_chart(fig_cmp, use_container_width=True)

            # Class frequency comparison
            fig_bars = period_comparison_bars(row_p1, row_p2, selected_station)
            st.plotly_chart(fig_bars, use_container_width=True)
        else:
            st.info("Period data not available for this station.")
