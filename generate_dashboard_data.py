"""
generate_dashboard_data.py
--------------------------
Builds every data file the Moldova Drought Intelligence Dashboard needs,
starting from the synthetic base climate dataset (no real station data).

Pipeline
--------
1. Ensure a synthetic base `master_dataset.csv` exists
   (created by generate_synthetic_data.py, seed=42).
2. Engineer the 10 extra v2 features (temporal lags/rolls + spatial coords +
   aridity + the 7-class target) → writes `data/master_dataset_v2.csv`.
3. Aggregate the enriched dataset into the three derived tables the dashboard
   reads:
       data/station_summary.csv      (16 stations × 3 periods)
       data/climatology_monthly.csv  (16 × 3 × 12 months)
       data/annual_timeseries.csv    (16 × 60 years)

Station coordinates are taken from utils/constants.STATION_COORDS — the single
source of truth — so the maps, the summary tables, and the ML feature
`station_lat/lon` all agree.

Usage
-----
    python3 generate_dashboard_data.py
"""

import os
import subprocess
import numpy as np
import pandas as pd

from utils.constants import STATION_COORDS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
BASE_CSV = os.path.join(HERE, "master_dataset.csv")

PERIODS = [("full", 1961, 2020), ("p1", 1961, 1990), ("p2", 1991, 2020)]

CLASS_NAMES = [
    "Extremely Dry", "Severely Dry", "Moderately Dry", "Normal",
    "Moderately Wet", "Severely Wet", "Extremely Wet",
]
PCT_COLS = {
    "Extremely Dry":  "pct_extremely_dry",
    "Severely Dry":   "pct_severely_dry",
    "Moderately Dry": "pct_moderately_dry",
    "Normal":         "pct_normal",
    "Moderately Wet": "pct_moderately_wet",
    "Severely Wet":   "pct_severely_wet",
    "Extremely Wet":  "pct_extremely_wet",
}
DRY_CLASSES = ["Extremely Dry", "Severely Dry", "Moderately Dry"]


# ── Classifiers ────────────────────────────────────────────────────────────────

def classify_spei_7(spei: float) -> str:
    """WMO 7-class drought severity scheme (McKee et al. 1993)."""
    if spei < -2.0:  return "Extremely Dry"
    if spei < -1.5:  return "Severely Dry"
    if spei < -1.0:  return "Moderately Dry"
    if spei <= 1.0:  return "Normal"
    if spei <= 1.5:  return "Moderately Wet"
    if spei <= 2.0:  return "Severely Wet"
    return "Extremely Wet"


def classify_spei_3(spei: float) -> str:
    if spei < -1.0:  return "Dry"
    if spei > 1.0:   return "Wet"
    return "Normal"


# ── Step 1: synthetic base data ─────────────────────────────────────────────────

def ensure_base_dataset() -> None:
    if os.path.exists(BASE_CSV):
        print(f"Base dataset found: {BASE_CSV}")
        return
    print("Base dataset missing — generating synthetic master_dataset.csv ...")
    subprocess.run(
        ["python3", os.path.join(HERE, "generate_synthetic_data.py"),
         "--seed", "42", "--output", BASE_CSV],
        check=True,
    )


# ── Step 2: v2 feature engineering ──────────────────────────────────────────────

def engineer_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["station", "year", "month"]).reset_index(drop=True)

    df["precip_lag6"]  = df.groupby("station")["precip"].shift(6)
    df["precip_lag12"] = df.groupby("station")["precip"].shift(12)
    df["t_med_lag6"]   = df.groupby("station")["t_med"].shift(6)
    df["precip_roll6"] = df.groupby("station")["precip"].transform(
        lambda x: x.rolling(6, min_periods=3).mean())
    df["precip_roll12"] = df.groupby("station")["precip"].transform(
        lambda x: x.rolling(12, min_periods=6).mean())
    df["t_med_anom_roll3"] = df.groupby("station")["t_med_anom"].transform(
        lambda x: x.rolling(3, min_periods=1).mean())

    df["station_lat"] = df["station"].map({k: v[0] for k, v in STATION_COORDS.items()})
    df["station_lon"] = df["station"].map({k: v[1] for k, v in STATION_COORDS.items()})
    df["aridity_idx"] = df["precip"] / (df["t_med"] + 10).clip(lower=0.5)

    df["drought_class_7"] = df["spei_3"].apply(classify_spei_7)
    return df


# ── Step 3: aggregations ────────────────────────────────────────────────────────

def build_annual_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (station, year), g in df.groupby(["station", "year"]):
        valid = g[g["spei_3"].notna()]
        n = len(valid)
        cls3 = valid["spei_3"].apply(classify_spei_3)
        rows.append({
            "station": station,
            "year": int(year),
            "precip_annual_sum": round(g["precip"].sum(), 1),
            "t_med_annual_mean": round(g["t_med"].mean(), 2),
            "t_max_annual_mean": round(g["t_max"].mean(), 2),
            "t_min_annual_mean": round(g["t_min"].mean(), 2),
            "spei3_annual_mean": round(valid["spei_3"].mean(), 2) if n else np.nan,
            "n_months": n,
            "pct_dry":    round(100 * (cls3 == "Dry").sum()    / n, 1) if n else 0.0,
            "pct_wet":    round(100 * (cls3 == "Wet").sum()    / n, 1) if n else 0.0,
            "pct_normal": round(100 * (cls3 == "Normal").sum() / n, 1) if n else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["station", "year"]).reset_index(drop=True)


def build_climatology(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, sdf in df.groupby("station"):
        for period, y0, y1 in PERIODS:
            sub = sdf[sdf["year"].between(y0, y1)]
            for month, mg in sub.groupby("month"):
                valid = mg[mg["spei_3"].notna()]
                rows.append({
                    "station": station,
                    "period": period,
                    "year_start": y0,
                    "year_end": y1,
                    "month": int(month),
                    "precip_monthly_mean": round(mg["precip"].mean(), 1),
                    "precip_monthly_std":  round(mg["precip"].std(ddof=1), 1),
                    "t_med_mean": round(mg["t_med"].mean(), 2),
                    "t_min_mean": round(mg["t_min"].mean(), 2),
                    "t_max_mean": round(mg["t_max"].mean(), 2),
                    "spei3_mean": round(valid["spei_3"].mean(), 3) if len(valid) else np.nan,
                })
    return pd.DataFrame(rows).sort_values(
        ["station", "period", "month"]).reset_index(drop=True)


def build_station_summary(df: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, sdf in df.groupby("station"):
        lat, lon = STATION_COORDS[station]
        for period, y0, y1 in PERIODS:
            sub = sdf[sdf["year"].between(y0, y1)]
            ann = annual[(annual.station == station) &
                         (annual.year.between(y0, y1))]
            valid = sub[sub["spei_3"].notna()]
            n = len(valid)
            cls7 = valid["drought_class_7"]

            # 7-class percentages over months with a defined SPEI-3
            pct = {name: round(100 * (cls7 == name).sum() / n, 1) if n else 0.0
                   for name in CLASS_NAMES}
            pct_any_dry = round(sum(pct[c] for c in DRY_CLASSES), 1)
            nonnormal = {k: v for k, v in pct.items() if k != "Normal"}
            dominant = max(nonnormal, key=nonnormal.get) if nonnormal else "Normal"

            # temperature trend (°C per decade) from annual mean temperature
            if len(ann) >= 2:
                slope = np.polyfit(ann["year"], ann["t_med_annual_mean"], 1)[0]
            else:
                slope = 0.0

            row = {
                "station": station,
                "lat": lat,
                "lon": lon,
                "period": period,
                "year_start": y0,
                "year_end": y1,
                "precip_annual_mean": round(ann["precip_annual_sum"].mean(), 1),
                "precip_annual_std":  round(ann["precip_annual_sum"].std(ddof=1), 1),
                "precip_annual_min":  round(ann["precip_annual_sum"].min(), 1),
                "precip_annual_max":  round(ann["precip_annual_sum"].max(), 1),
                "t_med_mean": round(sub["t_med"].mean(), 2),
                "t_min_mean": round(sub["t_min"].mean(), 2),
                "t_max_mean": round(sub["t_max"].mean(), 2),
                "t_slope_per_decade": round(slope * 10, 3),
                "n_months": n,
            }
            row.update({PCT_COLS[k]: pct[k] for k in CLASS_NAMES})
            row["pct_any_dry"] = pct_any_dry
            row["dominant_nonnormal"] = dominant
            rows.append(row)
    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_base_dataset()
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Reading base dataset and engineering v2 features ...")
    base = pd.read_csv(BASE_CSV)
    df = engineer_v2(base)

    v2_path = os.path.join(DATA_DIR, "master_dataset_v2.csv")
    df.to_csv(v2_path, index=False)
    print(f"  Saved: {v2_path}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

    print("Building annual_timeseries.csv ...")
    annual = build_annual_timeseries(df)
    annual.to_csv(os.path.join(DATA_DIR, "annual_timeseries.csv"), index=False)
    print(f"  Saved: annual_timeseries.csv  ({len(annual):,} rows)")

    print("Building climatology_monthly.csv ...")
    clim = build_climatology(df)
    clim.to_csv(os.path.join(DATA_DIR, "climatology_monthly.csv"), index=False)
    print(f"  Saved: climatology_monthly.csv  ({len(clim):,} rows)")

    print("Building station_summary.csv ...")
    summary = build_station_summary(df, annual)
    summary.to_csv(os.path.join(DATA_DIR, "station_summary.csv"), index=False)
    print(f"  Saved: station_summary.csv  ({len(summary):,} rows)")

    # Sanity check
    full = summary[summary.period == "full"]
    print("\nSanity check (full period, all-station means):")
    print(f"  precip_annual_mean : {full.precip_annual_mean.mean():.0f} mm/yr")
    print(f"  t_med_mean         : {full.t_med_mean.mean():.2f} °C")
    print(f"  pct_any_dry        : {full.pct_any_dry.mean():.1f}%")
    print(f"  pct_normal         : {full.pct_normal.mean():.1f}%")
    print(f"  n_months (full)    : {full.n_months.iloc[0]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
