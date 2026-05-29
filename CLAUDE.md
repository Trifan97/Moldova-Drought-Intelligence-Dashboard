# CLAUDE.md — project guide for AI assistants

## What this is
A Streamlit multipage dashboard exploring 60 years (1961–2020) of SPEI-based
drought dynamics for 16 Moldovan meteorological stations, plus a calibrated
7-class drought-severity classifier.

## Critical data rule
- **Only synthetic data is public.** Real station observations from the State
  Hydrometeorological Service of Moldova are proprietary and must NEVER be
  committed. Any file matching `*_REAL.*` is gitignored — keep it that way.
- All shipped CSVs in `data/` and artefacts in `models/` are derived from the
  synthetic generator (`generate_synthetic_data.py`, seed = 42).

## Data flow
```
generate_synthetic_data.py  → master_dataset.csv          (synthetic base, 11,520 rows)
generate_dashboard_data.py  → data/master_dataset_v2.csv  (24 features + 7-class target)
                            → data/annual_timeseries.csv
                            → data/climatology_monthly.csv
                            → data/station_summary.csv
train_model.py              → models/v2_*.pkl + metrics.json
```
`utils/constants.STATION_COORDS` is the single source of truth for station
coordinates (maps, summary tables, and the `station_lat/lon` ML features all
read from it).

## ML contract (do not break)
- `LabelEncoder.classes_` is set to `constants.CLASS_ORDER` **before** transform,
  so `predict_proba` columns line up 1-to-1 with the dashboard's class order.
- Feature order is fixed: `FEATURES_V2 = FEATURES_V1 (15) + NEW_FEATURES (9)`.
- The ML Explorer page reads `models/metrics.json`; regenerate it via
  `train_model.py` after any data change rather than hardcoding numbers.

## Run / regenerate
```bash
pip install -r requirements.txt
python3 generate_dashboard_data.py   # rebuild data
python3 train_model.py               # retrain + write metrics.json
python3 -m streamlit run Home.py     # launch
```

## Conventions
- Python 3.9, `python3` / `python3 -m streamlit`.
- Dark "Palantir-like" theme via `utils/styles.apply_dark_theme(...)`.
- Charts: Plotly (dark); maps: PyDeck `dark` style (no API key required).
- Boundary GeoJSON is loaded with plain `json` — no geopandas/rasterio needed
  at runtime; GIS source folders are excluded from the repo.
