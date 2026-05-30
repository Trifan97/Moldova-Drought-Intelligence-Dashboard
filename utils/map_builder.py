"""
map_builder.py — PyDeck layer factory for Moldova Drought Dashboard
"""
from __future__ import annotations

import json
import pydeck as pdk
import pandas as pd
from utils.constants import (
    MAP_CENTER_LAT, MAP_CENTER_LON, MAP_ZOOM,
    TEAL, NAVY2,
)


MOLDAVA_BBOX = {"min_lat": 45.4, "max_lat": 48.7,
                "min_lon": 26.5, "max_lon": 30.2}

TOOLTIP_STYLE = {
    "backgroundColor": "#0F2640",
    "border": "1px solid #1AA99A",
    "borderRadius": "6px",
    "color": "#E8EAED",
    "fontSize": "12px",
    "padding": "8px 12px",
    "maxWidth": "220px",
}


def build_station_layer(station_df: pd.DataFrame,
                        selected_station: str | None = None) -> pdk.Layer:
    """
    ScatterplotLayer for meteorological stations.
    Colour encodes % dry months (red = drier, teal = wetter).
    Selected station is highlighted with white ring.
    """
    # Larger radius for selected
    station_df = station_df.copy()
    station_df["radius"] = station_df["station"].apply(
        lambda s: 14_000 if s == selected_station else 9_000
    )
    station_df["line_color"] = station_df["station"].apply(
        lambda s: [255, 255, 255, 240] if s == selected_station
                  else [200, 200, 200, 120]
    )

    return pdk.Layer(
        "ScatterplotLayer",
        data=station_df,
        get_position=["lon", "lat"],
        get_fill_color="color",
        get_line_color="line_color",
        get_radius="radius",
        radius_min_pixels=6,
        radius_max_pixels=22,
        stroked=True,
        get_line_width=2,
        line_width_min_pixels=2,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 80],
    )


def build_boundary_layer(geojson: dict) -> pdk.Layer:
    """GeoJsonLayer for Moldova country outline."""
    return pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        get_fill_color=[22, 47, 71, 60],      # semi-transparent navy
        get_line_color=[26, 169, 154, 200],    # teal border
        line_width_min_pixels=2,
        pickable=False,
        stroked=True,
        filled=True,
    )


def build_station_label_layer(station_df: pd.DataFrame) -> pdk.Layer:
    """TextLayer for station name labels."""
    return pdk.Layer(
        "TextLayer",
        data=station_df,
        get_position=["lon", "lat"],
        get_text="station",
        get_size=12,
        get_color=[232, 234, 237, 200],
        get_pixel_offset=[0, -20],
        pickable=False,
    )


def build_deck(station_df: pd.DataFrame,
               boundary_geojson: dict,
               selected_station: str | None = None,
               show_labels: bool = True,
               pitch: float = 0,
               bearing: float = 0) -> pdk.Deck:
    """
    Assemble the full PyDeck deck for Page 1.
    Returns a pdk.Deck object ready for st.pydeck_chart().
    """
    layers = [
        build_boundary_layer(boundary_geojson),
        build_station_layer(station_df, selected_station),
    ]
    if show_labels:
        layers.append(build_station_label_layer(station_df))

    view_state = pdk.ViewState(
        latitude=MAP_CENTER_LAT,
        longitude=MAP_CENTER_LON,
        zoom=MAP_ZOOM,
        pitch=pitch,
        bearing=bearing,
        min_zoom=5,
        max_zoom=12,
    )

    tooltip = {
        "html": (
            "<b>{station}</b><br>"
            "📍 {lat:.2f}°N, {lon:.2f}°E<br>"
            "🌧 Precip: <b>{precip_annual} mm/yr</b><br>"
            "🌡 Temp: <b>{t_med} °C</b><br>"
            "☀ Dry months: <b>{pct_dry}%</b><br>"
            "Dominant: {dominant}"
        ),
        "style": TOOLTIP_STYLE,
    }

    return pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style="dark",          # Carto Dark Matter — no API key needed
        tooltip=tooltip,
    )
