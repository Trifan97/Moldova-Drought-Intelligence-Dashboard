"""
data_loader.py — cached data loading for Moldova Drought Dashboard
All heavy I/O is wrapped in @st.cache_data so it runs once per session.
"""

import json
from pathlib import Path

import pandas as pd
import numpy as np
import streamlit as st

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data
def load_station_summary() -> pd.DataFrame:
    """Station-level summary stats for all periods."""
    df = pd.read_csv(DATA_DIR / "station_summary.csv")
    return df


@st.cache_data
def load_climatology() -> pd.DataFrame:
    """Monthly climatology per station per period."""
    df = pd.read_csv(DATA_DIR / "climatology_monthly.csv")
    return df


@st.cache_data
def load_annual_timeseries() -> pd.DataFrame:
    """Annual precipitation sums and temperature means per station."""
    df = pd.read_csv(DATA_DIR / "annual_timeseries.csv")
    return df


@st.cache_data
def load_master_dataset() -> pd.DataFrame:
    """Full monthly dataset — used for raw queries."""
    df = pd.read_csv(DATA_DIR / "master_dataset_v2.csv")
    df = df.sort_values(["station", "year", "month"]).reset_index(drop=True)

    def classify7(s):
        if   s < -2.0: return "Extremely Dry"
        elif s < -1.5: return "Severely Dry"
        elif s < -1.0: return "Moderately Dry"
        elif s <=  1.0: return "Normal"
        elif s <=  1.5: return "Moderately Wet"
        elif s <=  2.0: return "Severely Wet"
        else:           return "Extremely Wet"

    df["drought_class_7"] = df["spei_3"].apply(classify7)
    return df


@st.cache_data
def load_moldova_boundary() -> dict:
    """Moldova country boundary GeoJSON."""
    with open(DATA_DIR / "moldova_boundary.geojson") as f:
        return json.load(f)


@st.cache_data
def load_stations_geojson() -> dict:
    """Station point GeoJSON."""
    with open(DATA_DIR / "stations.geojson") as f:
        return json.load(f)


# ── Derived helpers ────────────────────────────────────────────────────────────

def get_station_period_row(station: str, period: str) -> pd.Series:
    """Return the summary row for one station + period."""
    df = load_station_summary()
    row = df[(df["station"] == station) & (df["period"] == period)]
    if len(row) == 0:
        return None
    return row.iloc[0]


def get_climatogram_data(station: str, period: str) -> pd.DataFrame:
    """Monthly climatology for one station + period."""
    df = load_climatology()
    return df[(df["station"] == station) & (df["period"] == period)].copy()


def get_annual_series(station: str, period_key: str = "full") -> pd.DataFrame:
    """Annual timeseries filtered by period."""
    from utils.constants import PERIOD_YEARS
    df = load_annual_timeseries()
    y1, y2 = PERIOD_YEARS[period_key]
    sub = df[(df["station"] == station) & df["year"].between(y1, y2)].copy()
    return sub


def get_drought_class_freq(station: str, period: str) -> dict:
    """Return class frequency dict {class_name: pct} for pie chart."""
    from utils.constants import CLASS_ORDER
    row = get_station_period_row(station, period)
    if row is None:
        return {}
    mapping = {
        "Extremely Dry":  row["pct_extremely_dry"],
        "Severely Dry":   row["pct_severely_dry"],
        "Moderately Dry": row["pct_moderately_dry"],
        "Normal":         row["pct_normal"],
        "Moderately Wet": row["pct_moderately_wet"],
        "Severely Wet":   row["pct_severely_wet"],
        "Extremely Wet":  row["pct_extremely_wet"],
    }
    return {k: mapping[k] for k in CLASS_ORDER}


def build_station_map_df(period: str = "full") -> pd.DataFrame:
    """
    Return a flat DataFrame with one row per station, ready for PyDeck.
    Includes lat, lon, colour (based on pct_any_dry), and hover fields.
    """
    from utils.constants import STATION_COORDS, CLASS_PYDECK

    df = load_station_summary()
    sub = df[df["period"] == period].copy()

    # Colour stations by pct_any_dry percentile
    # Low dry → teal; high dry → red
    lo = sub["pct_any_dry"].min()
    hi = sub["pct_any_dry"].max()

    def dry_color(pct):
        t = (pct - lo) / (hi - lo + 1e-6)
        r = int(139 * t + 26  * (1 - t))
        g = int(  0 * t + 169 * (1 - t))
        b = int(  0 * t + 154 * (1 - t))
        return [r, g, b, 220]

    rows = []
    for _, row in sub.iterrows():
        s = row["station"]
        lat, lon = STATION_COORDS[s]
        rows.append({
            "station":           s,
            "lat":               lat,
            "lon":               lon,
            "precip_annual":     row["precip_annual_mean"],
            "t_med":             row["t_med_mean"],
            "pct_dry":           row["pct_any_dry"],
            "dominant":          row["dominant_nonnormal"],
            "color":             dry_color(row["pct_any_dry"]),
            "tooltip_text":      (
                f"{s}\n"
                f"Precip: {row['precip_annual_mean']} mm/yr\n"
                f"Temp:   {row['t_med_mean']} °C\n"
                f"Dry months: {row['pct_any_dry']}%"
            ),
        })
    return pd.DataFrame(rows)
