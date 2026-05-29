"""
chart_builder.py — Plotly chart factory for Moldova Drought Dashboard
All functions return plotly.graph_objects.Figure objects.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.constants import (
    CLASS_COLORS, CLASS_ORDER, MONTH_LABELS,
    TEAL, AMBER, NAVY, NAVY2, BORDER, TEXT, MUTED,
)


def _base_layout(**overrides) -> dict:
    """Shared layout kwargs for all charts."""
    base = dict(
        paper_bgcolor=NAVY2,
        plot_bgcolor=NAVY,
        font=dict(color=TEXT, size=12),
        margin=dict(l=50, r=20, t=40, b=45),
        hoverlabel=dict(bgcolor=NAVY2, bordercolor=TEAL,
                        font=dict(color=TEXT, size=12)),
    )
    base.update(overrides)
    return base


# ── Climatogram ───────────────────────────────────────────────────────────────

def climatogram(clim_df: pd.DataFrame, station: str,
                period_label: str) -> go.Figure:
    """
    Dual-axis climatogram: precipitation bars + temperature line.
    Standard Walter-Lieth style for climate science.
    """
    months = list(range(1, 13))
    clim = clim_df.sort_values("month")

    fig = go.Figure()

    # Precipitation bars
    fig.add_trace(go.Bar(
        x=MONTH_LABELS,
        y=clim["precip_monthly_mean"],
        name="Precip (mm)",
        marker_color=TEAL,
        marker_opacity=0.85,
        yaxis="y1",
        hovertemplate="<b>%{x}</b><br>Precip: %{y:.1f} mm<extra></extra>",
    ))

    # Temperature line
    fig.add_trace(go.Scatter(
        x=MONTH_LABELS,
        y=clim["t_med_mean"],
        name="Temp (°C)",
        mode="lines+markers",
        line=dict(color=AMBER, width=2.5),
        marker=dict(size=6, color=AMBER),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Temp: %{y:.1f} °C<extra></extra>",
    ))

    # Zero temperature reference line
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED,
                  line_width=1, yref="y2")

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"<b>{station}</b> — {period_label}",
                font=dict(size=13, color=TEXT),
            ),
            xaxis=dict(
                gridcolor=BORDER, linecolor=BORDER,
                tickfont=dict(size=11),
            ),
            yaxis=dict(
                title=dict(text="Precipitation (mm)", font=dict(color=TEAL)),
                tickfont=dict(color=TEAL),
                gridcolor=BORDER,
                rangemode="tozero",
            ),
            yaxis2=dict(
                title=dict(text="Temperature (°C)", font=dict(color=AMBER)),
                tickfont=dict(color=AMBER),
                overlaying="y",
                side="right",
                gridcolor="rgba(0,0,0,0)",
                zeroline=False,
            ),
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(15,38,64,0.8)",
                bordercolor=BORDER,
                borderwidth=1,
            ),
            barmode="overlay",
            height=320,
        )
    )
    return fig


# ── Dual-period climatogram comparison ───────────────────────────────────────

def climatogram_comparison(clim_p1: pd.DataFrame, clim_p2: pd.DataFrame,
                            station: str) -> go.Figure:
    """
    Overlaid precipitation bars + temperature lines for P1 vs P2.
    P1 = solid/lighter, P2 = darker/bold.
    """
    fig = go.Figure()
    configs = [
        (clim_p1, "1961–1990", TEAL,  "rgba(26,169,154,0.35)", "dot"),
        (clim_p2, "1991–2020", AMBER, "rgba(232,146,58,0.55)",  "solid"),
    ]
    for clim, label, tcolor, bar_color, dash in configs:
        clim = clim.sort_values("month")
        fig.add_trace(go.Bar(
            x=MONTH_LABELS, y=clim["precip_monthly_mean"],
            name=f"Precip {label}",
            marker_color=bar_color,
            yaxis="y1",
            hovertemplate=f"<b>%{{x}}</b> {label}<br>Precip: %{{y:.1f}} mm<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=MONTH_LABELS, y=clim["t_med_mean"],
            name=f"Temp {label}",
            mode="lines+markers",
            line=dict(color=tcolor, width=2.2, dash=dash),
            marker=dict(size=5, color=tcolor),
            yaxis="y2",
            hovertemplate=f"<b>%{{x}}</b> {label}<br>Temp: %{{y:.1f}} °C<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"<b>{station}</b> — Period comparison",
                font=dict(size=13, color=TEXT),
            ),
            barmode="group",
            xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
            yaxis=dict(
                title="Precipitation (mm)",
                gridcolor=BORDER, rangemode="tozero",
            ),
            yaxis2=dict(
                title="Temperature (°C)",
                overlaying="y", side="right",
                gridcolor="rgba(0,0,0,0)",
            ),
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(15,38,64,0.8)",
                bordercolor=BORDER, borderwidth=1,
                font=dict(size=10),
            ),
            height=340,
        )
    )
    return fig


# ── Annual precipitation timeseries ──────────────────────────────────────────

def annual_precip_chart(ann_df: pd.DataFrame, station: str,
                        show_trend: bool = True) -> go.Figure:
    """
    Annual precipitation sum bar chart with optional linear trend line.
    Period divider at 1990/1991 boundary.
    """
    fig = go.Figure()

    # Colour bars by period
    colors = [TEAL if y <= 1990 else AMBER for y in ann_df["year"]]

    fig.add_trace(go.Bar(
        x=ann_df["year"],
        y=ann_df["precip_annual_sum"],
        name="Annual sum",
        marker_color=colors,
        marker_opacity=0.8,
        hovertemplate="<b>%{x}</b><br>%{y:.0f} mm<extra></extra>",
    ))

    if show_trend and len(ann_df) >= 5:
        # Full-period linear trend
        x_vals = ann_df["year"].values
        y_vals = ann_df["precip_annual_sum"].values
        mask   = ~np.isnan(y_vals)
        coeffs = np.polyfit(x_vals[mask], y_vals[mask], 1)
        trend  = np.polyval(coeffs, x_vals)
        slope_dec = coeffs[0] * 10
        fig.add_trace(go.Scatter(
            x=ann_df["year"], y=trend,
            name=f"Trend ({slope_dec:+.1f} mm/decade)",
            mode="lines",
            line=dict(color="white", width=1.5, dash="dash"),
            hoverinfo="skip",
        ))

    # Period divider
    fig.add_vline(x=1990.5, line_color=MUTED,
                  line_dash="dot", line_width=1.2)
    fig.add_annotation(x=1975, y=0, yref="paper",
                       text="P1: 1961–1990", showarrow=False,
                       font=dict(color=TEAL, size=10), yanchor="bottom")
    fig.add_annotation(x=2005, y=0, yref="paper",
                       text="P2: 1991–2020", showarrow=False,
                       font=dict(color=AMBER, size=10), yanchor="bottom")

    p1_mean = ann_df[ann_df.year <= 1990]["precip_annual_sum"].mean()
    p2_mean = ann_df[ann_df.year >= 1991]["precip_annual_sum"].mean()
    if not np.isnan(p1_mean):
        fig.add_hline(y=p1_mean, line_color=TEAL,
                      line_dash="dot", line_width=1,
                      annotation_text=f"P1 mean: {p1_mean:.0f} mm",
                      annotation_font_color=TEAL,
                      annotation_position="top left")
    if not np.isnan(p2_mean):
        fig.add_hline(y=p2_mean, line_color=AMBER,
                      line_dash="dot", line_width=1,
                      annotation_text=f"P2 mean: {p2_mean:.0f} mm",
                      annotation_font_color=AMBER,
                      annotation_position="bottom right")

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"<b>{station}</b> — Annual precipitation (mm/year)",
                font=dict(size=13, color=TEXT),
            ),
            xaxis=dict(title="Year", gridcolor=BORDER, linecolor=BORDER,
                       dtick=10),
            yaxis=dict(title="Precipitation (mm/year)",
                       gridcolor=BORDER, rangemode="tozero"),
            showlegend=True,
            height=280,
        )
    )
    return fig


# ── Annual temperature timeseries ────────────────────────────────────────────

def annual_temp_chart(ann_df: pd.DataFrame, station: str) -> go.Figure:
    """Annual mean temperature with period means and trend."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=ann_df["year"],
        y=ann_df["t_med_annual_mean"],
        mode="lines+markers",
        name="Annual mean T",
        line=dict(color=AMBER, width=1.8),
        marker=dict(size=4, color=AMBER),
        hovertemplate="<b>%{x}</b><br>%{y:.2f} °C<extra></extra>",
    ))

    if len(ann_df) >= 5:
        x_vals = ann_df["year"].values
        y_vals = ann_df["t_med_annual_mean"].ffill().values
        coeffs = np.polyfit(x_vals, y_vals, 1)
        trend  = np.polyval(coeffs, x_vals)
        slope_dec = coeffs[0] * 10
        fig.add_trace(go.Scatter(
            x=ann_df["year"], y=trend,
            name=f"Trend ({slope_dec:+.2f} °C/decade)",
            mode="lines",
            line=dict(color="white", width=1.5, dash="dash"),
        ))

    p1_mean = ann_df[ann_df.year <= 1990]["t_med_annual_mean"].mean()
    p2_mean = ann_df[ann_df.year >= 1991]["t_med_annual_mean"].mean()
    fig.add_vline(x=1990.5, line_color=MUTED, line_dash="dot", line_width=1.2)
    if not np.isnan(p1_mean):
        fig.add_hline(y=p1_mean, line_color=TEAL, line_dash="dot", line_width=1,
                      annotation_text=f"P1: {p1_mean:.1f} °C",
                      annotation_font_color=TEAL)
    if not np.isnan(p2_mean):
        fig.add_hline(y=p2_mean, line_color=AMBER, line_dash="dot", line_width=1,
                      annotation_text=f"P2: {p2_mean:.1f} °C",
                      annotation_font_color=AMBER,
                      annotation_position="bottom right")

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"<b>{station}</b> — Annual mean temperature (°C)",
                font=dict(size=13, color=TEXT),
            ),
            xaxis=dict(title="Year", gridcolor=BORDER, linecolor=BORDER, dtick=10),
            yaxis=dict(title="Temperature (°C)", gridcolor=BORDER),
            height=250,
        )
    )
    return fig


# ── Drought class frequency donut ────────────────────────────────────────────

def drought_donut(freq_dict: dict, station: str,
                  period_label: str) -> go.Figure:
    """Donut chart of 7-class drought frequency."""
    labels = list(freq_dict.keys())
    values = list(freq_dict.values())
    colors = [CLASS_COLORS[l] for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.52,
        marker=dict(colors=colors, line=dict(color=NAVY2, width=1.5)),
        textinfo="percent",
        textfont=dict(size=11, color=TEXT),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        sort=False,
        direction="clockwise",
    ))

    fig.add_annotation(
        text=f"<b>{station}</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=12, color=TEXT),
    )

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"Drought class distribution — {period_label}",
                font=dict(size=12, color=TEXT),
            ),
            showlegend=True,
            legend=dict(
                x=1.02, y=0.5,
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10),
                orientation="v",
            ),
            height=320,
            margin=dict(l=10, r=130, t=40, b=10),
        )
    )
    return fig


# ── Period comparison bar chart ───────────────────────────────────────────────

def period_comparison_bars(row_p1: pd.Series, row_p2: pd.Series,
                            station: str) -> go.Figure:
    """
    Side-by-side bars comparing key metrics P1 vs P2 for a station.
    """
    CLASS_ORDER_SHORT = [
        "Ext. Dry", "Sev. Dry", "Mod. Dry",
        "Normal",
        "Mod. Wet", "Sev. Wet", "Ext. Wet",
    ]
    pct_cols_p1 = [
        row_p1["pct_extremely_dry"], row_p1["pct_severely_dry"],
        row_p1["pct_moderately_dry"], row_p1["pct_normal"],
        row_p1["pct_moderately_wet"], row_p1["pct_severely_wet"],
        row_p1["pct_extremely_wet"],
    ]
    pct_cols_p2 = [
        row_p2["pct_extremely_dry"], row_p2["pct_severely_dry"],
        row_p2["pct_moderately_dry"], row_p2["pct_normal"],
        row_p2["pct_moderately_wet"], row_p2["pct_severely_wet"],
        row_p2["pct_extremely_wet"],
    ]
    bar_colors = [CLASS_COLORS[c] for c in [
        "Extremely Dry","Severely Dry","Moderately Dry","Normal",
        "Moderately Wet","Severely Wet","Extremely Wet"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="1961–1990", x=CLASS_ORDER_SHORT, y=pct_cols_p1,
        marker_color=bar_colors, marker_opacity=0.55,
        hovertemplate="<b>%{x}</b> P1<br>%{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="1991–2020", x=CLASS_ORDER_SHORT, y=pct_cols_p2,
        marker_color=bar_colors, marker_opacity=0.95,
        hovertemplate="<b>%{x}</b> P2<br>%{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"<b>{station}</b> — Class frequency: P1 vs P2",
                font=dict(size=13, color=TEXT),
            ),
            barmode="group",
            xaxis=dict(tickfont=dict(size=10), gridcolor=BORDER),
            yaxis=dict(title="% of months", gridcolor=BORDER,
                       rangemode="tozero"),
            legend=dict(
                x=0.01, y=0.99,
                bgcolor="rgba(15,38,64,0.8)",
                bordercolor=BORDER, borderwidth=1,
            ),
            height=300,
        )
    )
    return fig


# ── SPEI-3 annual mean timeseries ────────────────────────────────────────────

def spei_timeseries(ann_df: pd.DataFrame, station: str) -> go.Figure:
    """SPEI-3 annual mean with wet/dry shading."""
    fig = go.Figure()

    # Positive area (wet)
    fig.add_trace(go.Scatter(
        x=ann_df["year"], y=ann_df["spei3_annual_mean"].clip(lower=0),
        fill="tozeroy", fillcolor="rgba(91,163,201,0.30)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Wet", hoverinfo="skip", showlegend=False,
    ))
    # Negative area (dry)
    fig.add_trace(go.Scatter(
        x=ann_df["year"], y=ann_df["spei3_annual_mean"].clip(upper=0),
        fill="tozeroy", fillcolor="rgba(212,105,58,0.30)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Dry", hoverinfo="skip", showlegend=False,
    ))
    # SPEI line
    fig.add_trace(go.Scatter(
        x=ann_df["year"], y=ann_df["spei3_annual_mean"],
        mode="lines",
        line=dict(color=TEXT, width=1.5),
        name="SPEI-3 mean",
        hovertemplate="<b>%{x}</b><br>SPEI-3: %{y:.3f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=MUTED, line_width=1)
    fig.add_hline(y=-1, line_color="#D4693A", line_dash="dot",
                  line_width=0.8, annotation_text="Dry threshold",
                  annotation_font_color="#D4693A", annotation_font_size=10)
    fig.add_hline(y=1, line_color="#5BA3C9", line_dash="dot",
                  line_width=0.8, annotation_text="Wet threshold",
                  annotation_font_color="#5BA3C9", annotation_font_size=10,
                  annotation_position="bottom right")
    fig.add_vline(x=1990.5, line_color=MUTED, line_dash="dot", line_width=1.2)

    fig.update_layout(
        **_base_layout(
            title=dict(
                text=f"<b>{station}</b> — SPEI-3 annual mean",
                font=dict(size=13, color=TEXT),
            ),
            xaxis=dict(title="Year", gridcolor=BORDER, dtick=10),
            yaxis=dict(title="SPEI-3", gridcolor=BORDER),
            height=240,
        )
    )
    return fig
