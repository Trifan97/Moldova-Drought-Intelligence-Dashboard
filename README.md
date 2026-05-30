# Moldova Drought Intelligence Dashboard

[![Live demo](https://img.shields.io/badge/Live%20demo-Streamlit%20Cloud-FF4B4B?logo=streamlit&logoColor=white)](https://moldova-drought-intelligence-dashboard-7ing3ca97hx998d3skhngd.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.855-1AA99A)

**🚀 Live demo:** https://moldova-drought-intelligence-dashboard-7ing3ca97hx998d3skhngd.streamlit.app/

An interactive Streamlit dashboard for exploring 60 years of drought dynamics
across 16 meteorological stations in the Republic of Moldova (1961–2020),
built on the WMO 7-class SPEI severity scheme and a calibrated gradient-boosting
classifier.

> **Data note.** This public repository ships **synthetic** climate data only.
> The numbers, maps, and model are generated from a reproducible synthetic
> dataset (`generate_synthetic_data.py`, seed = 42) that reproduces the
> statistical structure of the real records without exposing any proprietary
> observations from the State Hydrometeorological Service of Moldova.

## Pages

| Page | What it shows |
|---|---|
| **🗺️ Station Explorer** | Dark interactive map of all 16 stations; per-station climatogram, annual precipitation, and 1961–1990 vs 1991–2020 drought-class frequency. |
| **📈 Climate Change Signal** | Two-period comparison: temperature trends, precipitation variability, SPEI-3 running mean, and shifts in drought-class frequency. |
| **⚠️ Drought Risk Map** | Composite Drought Risk Score per region from frequency, duration and intensity of events, with adjustable severity weights. |
| **🤖 ML Model Explorer** | Calibrated 7-class classifier diagnostics — per-class F1, confusion matrix, permutation feature importance, and a live input → probability predictor. |

## Machine-learning model

A 7-class SPEI drought-severity classifier trained on the synthetic dataset.

| Metric | Score |
|---|---|
| Macro F1 (held-out test) | **0.855** |
| Weighted F1 | 0.913 |
| Accuracy | 0.911 |
| ROC-AUC (one-vs-rest macro) | 0.992 |
| CV macro F1 (5-fold) | 0.955 ± 0.007 |

**Pipeline:** median imputation → SMOTE (minorities to 30 % of *Normal*) →
StandardScaler → HistGradientBoostingClassifier (balanced sample weights) →
isotonic probability calibration (`CalibratedClassifierCV`, cv = 3).
24 engineered features (temporal lags/rolls, anomalies, spatial coordinates,
aridity index). Top drivers: 3-month precipitation roll, lagged median
temperature, and station latitude.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the synthetic data + retrain the model
python3 generate_dashboard_data.py   # → data/*.csv
python3 train_model.py               # → models/*.pkl + metrics.json

# 3. Launch the dashboard
python3 -m streamlit run Home.py
```

The repo ships with the generated data (`data/*.csv`) and trained artefacts
(`models/*.pkl`), so step 3 works out of the box.

## Repository structure

```
.
├── Home.py                       # Landing page — KPIs, navigation, overview charts
├── pages/
│   ├── 1_Station_Explorer.py
│   ├── 2_Climate_Change_Signal.py
│   ├── 3_Drought_Risk_Map.py
│   └── 4_ML_Model_Explorer.py
├── utils/                        # constants, data loaders, chart/map builders, styles
├── data/                         # synthetic CSVs + Moldova boundary GeoJSON
├── models/                       # calibrated classifier + preprocessing artefacts
├── generate_synthetic_data.py    # synthetic base climate generator (seed=42)
├── generate_dashboard_data.py    # feature engineering + dashboard aggregations
├── train_model.py                # trains the 7-class classifier, ships pkls
├── requirements.txt
├── LICENSE
└── README.md
```

## Methodology

- **SPEI-3** (Standardized Precipitation-Evapotranspiration Index, 3-month
  accumulation) drives both the 7-class severity labels and the model target.
- **7-class WMO/McKee scheme:** Extremely / Severely / Moderately Dry · Normal ·
  Moderately / Severely / Extremely Wet.
- Synthetic generation models Gamma-distributed precipitation with AR(1)
  autocorrelation, a Thornthwaite-style PET term, and a north–south aridity
  gradient — southern stations (Cahul, Comrat, Ceadir-Lunga) are drier.

## Related repositories

- [Supervised ML — SPEI 7-class classification](https://github.com/Trifan97/Supervised-ML-SPEI-Prediction)
- [Unsupervised ML — SPEI drought clustering](https://github.com/Trifan97/SPEI-Unsupervised-ML-Clustering)

## Author

**Tudor Trifan** — PhD Candidate, Czech University of Life Sciences Prague
[Portfolio](https://tudor-trifan.github.io) ·
[GitHub](https://github.com/Trifan97) ·
[LinkedIn](https://www.linkedin.com/in/tudor-trifan/)

## License

Released under the [MIT License](LICENSE).
