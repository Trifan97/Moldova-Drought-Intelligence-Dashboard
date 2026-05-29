"""
pages/4_ML_Model_Explorer.py
Explore the 7-class HistGradientBoosting pipeline.
Interactive input → calibrated class probability output.
Feature importances, confusion matrix, per-class metrics.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.constants import (
    CLASS_COLORS, CLASS_ORDER, TEAL, AMBER,
    NAVY, NAVY2, BORDER, TEXT, MUTED, MONTH_LABELS,
    STATION_COORDS,
)
from utils.styles import apply_dark_theme

st.set_page_config(
    page_title="ML Model Explorer — Moldova Drought",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dark_theme("ml_explorer")


def layout(**kw):
    base = dict(
        paper_bgcolor=NAVY2, plot_bgcolor=NAVY,
        font=dict(color=TEXT, size=11),
        margin=dict(l=50, r=20, t=40, b=40),
        hoverlabel=dict(bgcolor=NAVY2, bordercolor=TEAL, font=dict(color=TEXT)),
    )
    base.update(kw)
    return base


# ── Try to load trained model ─────────────────────────────────────────────────
MODEL_AVAILABLE = False
model = imputer = scaler = le = feature_meta = None

@st.cache_resource
def load_model():
    import joblib
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    try:
        m  = joblib.load(os.path.join(model_dir, "v2_best_model_calibrated.pkl"))
        im = joblib.load(os.path.join(model_dir, "v2_imputer.pkl"))
        sc = joblib.load(os.path.join(model_dir, "v2_scaler.pkl"))
        l  = joblib.load(os.path.join(model_dir, "v2_label_encoder.pkl"))
        fm = joblib.load(os.path.join(model_dir, "v2_feature_meta.pkl"))
        return m, im, sc, l, fm, True
    except Exception:
        return None, None, None, None, None, False

model, imputer, scaler, le, feature_meta, MODEL_AVAILABLE = load_model()


# ── Model results — loaded from models/metrics.json (written by train_model.py) ─
# These are the real held-out test metrics from training on the synthetic dataset.
@st.cache_data
def load_metrics():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "models", "metrics.json")
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

# Fallback mirrors the committed synthetic-run metrics if metrics.json is absent.
_FALLBACK = {
    "model_name":   "HistGradientBoosting + SMOTE 30% + balanced sample_weight",
    "macro_f1":     0.855,
    "weighted_f1":  0.913,
    "roc_auc":      0.992,
    "accuracy":     0.911,
    "cv_f1_mean":   0.955,
    "cv_f1_std":    0.007,
    "per_class": {
        "Extremely Dry":  {"precision":0.85,"recall":0.79,"f1":0.82,"support": 29},
        "Severely Dry":   {"precision":0.88,"recall":0.72,"f1":0.79,"support":140},
        "Moderately Dry": {"precision":0.74,"recall":0.94,"f1":0.83,"support":268},
        "Normal":         {"precision":0.98,"recall":0.93,"f1":0.96,"support":1457},
        "Moderately Wet": {"precision":0.82,"recall":0.92,"f1":0.87,"support":288},
        "Severely Wet":   {"precision":0.88,"recall":0.76,"f1":0.82,"support": 89},
        "Extremely Wet":  {"precision":0.97,"recall":0.85,"f1":0.90,"support": 33},
    },
    "feature_importance": {
        "precip_roll3": 0.42, "t_med_lag1": 0.22, "t_med_lag6": 0.11,
        "month_sin": 0.07, "station_lat": 0.05, "t_med_anom_roll3": 0.04,
        "month_cos": 0.02, "t_max": 0.01, "precip_roll12": 0.01,
    },
    "new_features": ["precip_lag6","precip_lag12","t_med_lag6","precip_roll6",
                     "precip_roll12","t_med_anom_roll3","station_lat",
                     "station_lon","aridity_idx"],
    "n_rows": 11520, "smote_normal": 5829, "smote_target": 1748,
}

V2_RESULTS = load_metrics() or _FALLBACK

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 ML Model Explorer")
    st.markdown("---")
    st.markdown(f"""
    <div class='model-card' style='margin-bottom:12px;'>
      <div style='font-size:0.8rem; font-weight:600; color:#E8EAED; margin-bottom:6px;'>
        Active model
      </div>
      <div style='font-size:0.75rem; color:#8A9BAE; line-height:1.7;'>
        HistGradientBoosting<br>
        + SMOTE 30% + balanced sw<br>
        Macro F1 = <b style='color:#1AA99A;'>{V2_RESULTS['macro_f1']:.3f}</b><br>
        AUC = <b style='color:#1AA99A;'>{V2_RESULTS['roc_auc']:.3f}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

    active_tab = st.radio(
        "View",
        ["Model metrics", "Feature importance", "Live prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if MODEL_AVAILABLE:
        st.success("✓ Calibrated model loaded")
    else:
        st.info("ℹ Model not loaded — run pipeline first to enable live predictions.\nMetrics shown from last run.")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:center;gap:10px;margin-bottom:6px;'>
  <span style='font-size:1.6rem;'>🤖</span>
  <div>
    <h2 style='margin:0;font-size:1.3rem!important;'>ML Model Explorer</h2>
    <p style='color:#8A9BAE;margin:0;font-size:0.85rem;'>
      7-class drought severity classification · HistGradientBoosting · Macro F1 = {V2_RESULTS['macro_f1']:.3f}
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Macro F1 ★",    f"{V2_RESULTS['macro_f1']:.3f}",   "7 classes, equal weight")
k2.metric("Weighted F1",   f"{V2_RESULTS['weighted_f1']:.3f}", "support-weighted")
k3.metric("ROC-AUC",       f"{V2_RESULTS['roc_auc']:.3f}",    "one-vs-rest macro")
k4.metric("Accuracy",      f"{V2_RESULTS['accuracy']:.3f}",   "held-out test set")
k5.metric("CV Macro F1",   f"{V2_RESULTS['cv_f1_mean']:.3f}",
          f"±{V2_RESULTS['cv_f1_std']:.3f}")

st.markdown("---")

# ── Tab content ───────────────────────────────────────────────────────────────

if active_tab == "Model metrics":
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown("### Per-class F1-score")
        fig = go.Figure()
        classes = CLASS_ORDER
        f1_v2  = [V2_RESULTS["per_class"][c]["f1"] for c in classes]
        short  = ["ExDry","SevDry","ModDry","Normal","ModWet","SevWet","ExWet"]
        colors = [CLASS_COLORS[c] for c in classes]

        fig.add_trace(go.Bar(
            name="F1-score", x=short, y=f1_v2,
            marker_color=colors, marker_opacity=0.95,
            text=[f"{v:.2f}" for v in f1_v2], textposition="outside",
            hovertemplate="<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(
            **layout(
                xaxis=dict(gridcolor=BORDER),
                yaxis=dict(title="F1-score", gridcolor=BORDER,
                           range=[0, 1.05]),
                showlegend=False,
                height=320,
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Per-class metrics table")
        rows = []
        for cls in CLASS_ORDER:
            d = V2_RESULTS["per_class"][cls]
            rows.append({
                "Class":     cls,
                "Precision": f"{d['precision']:.2f}",
                "Recall":    f"{d['recall']:.2f}",
                "F1":        f"{d['f1']:.2f}",
                "Support":   d["support"],
            })
        df_tbl = pd.DataFrame(rows)
        st.dataframe(df_tbl, use_container_width=True, hide_index=True, height=290)

    # Model pipeline details
    st.markdown("---")
    st.markdown("### Pipeline architecture")
    st.markdown(f"""
    <div class='insight-box' style='line-height:2;'>
    <b>1. Feature engineering</b> — 24 features (15 original + 9 new temporal/spatial)<br>
    <b>2. Imputation</b> — SimpleImputer(strategy='median') — fit on train only<br>
    <b>3. SMOTE</b> — minority classes upsampled to 30% of Normal
        ({V2_RESULTS.get('smote_normal', 5829):,} → {V2_RESULTS.get('smote_target', 1748):,} each)<br>
    <b>4. Scaling</b> — StandardScaler fit on SMOTE-resampled training set<br>
    <b>5. Model</b> — HistGradientBoostingClassifier(max_iter=300, lr=0.05, max_depth=6)<br>
    <b>6. Class imbalance</b> — compute_sample_weight('balanced') at fit-time<br>
    <b>7. Calibration</b> — CalibratedClassifierCV(method='isotonic', cv=3)<br>
    <b>Primary metric</b> — Macro F1 (all 7 classes weighted equally)
    </div>
    """, unsafe_allow_html=True)

    # Per-class precision vs recall
    st.markdown("### Per-class precision vs recall")
    prec = [V2_RESULTS["per_class"][c]["precision"] for c in CLASS_ORDER]
    rec  = [V2_RESULTS["per_class"][c]["recall"] for c in CLASS_ORDER]
    short = ["ExDry","SevDry","ModDry","Normal","ModWet","SevWet","ExWet"]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(name="Precision", x=short, y=prec,
                          marker_color=TEAL, marker_opacity=0.85,
                          hovertemplate="<b>%{x}</b> precision<br>%{y:.3f}<extra></extra>"))
    fig3.add_trace(go.Bar(name="Recall", x=short, y=rec,
                          marker_color=AMBER, marker_opacity=0.85,
                          hovertemplate="<b>%{x}</b> recall<br>%{y:.3f}<extra></extra>"))
    fig3.update_layout(
        **layout(
            barmode="group",
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(title="Score", gridcolor=BORDER, range=[0, 1.05]),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(15,38,64,0.8)",
                        bordercolor=BORDER, borderwidth=1),
            height=300,
        )
    )
    st.plotly_chart(fig3, use_container_width=True)


elif active_tab == "Feature importance":
    st.markdown("### Feature importances — v2 HistGradientBoosting")

    NEW_FEATURES = [
        "precip_lag6","precip_lag12","t_med_lag6","precip_roll6",
        "precip_roll12","t_med_anom_roll3","station_lat","station_lon","aridity_idx",
    ]

    fi = pd.Series(V2_RESULTS["feature_importance"]).sort_values(ascending=True)
    colors_fi = ["#27500A" if f in NEW_FEATURES else "#0C447C" for f in fi.index]

    fig_fi = go.Figure(go.Bar(
        y=fi.index, x=fi.values,
        orientation="h",
        marker_color=colors_fi,
        hovertemplate="<b>%{y}</b>: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(
        **layout(
            title=dict(text="<b>Feature importances — v2 pipeline (green = new features)</b>",
                       font=dict(size=13, color=TEXT)),
            xaxis=dict(title="Permutation importance (share of total)", gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER, tickfont=dict(size=10)),
            showlegend=False,
            height=420,
            margin=dict(l=160, r=20, t=40, b=40),
        )
    )
    col_fi, col_fi2 = st.columns([2, 1])
    with col_fi:
        st.plotly_chart(fig_fi, use_container_width=True)
    with col_fi2:  # noqa: E501
        fi_all = V2_RESULTS["feature_importance"]
        new_total = sum(v for k, v in fi_all.items() if k in NEW_FEATURES)
        old_total = sum(v for k, v in fi_all.items() if k not in NEW_FEATURES)
        top3 = sorted(fi_all.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_html = "<br>".join(
            f"{i+1}. {k} ({v*100:.1f}%)" + (" ← new" if k in NEW_FEATURES else "")
            for i, (k, v) in enumerate(top3))
        st.markdown(f"""
        <div class='insight-box' style='margin-top:20px;'>
        <b>New features contribution</b><br>
        <span style='color:#27500A;font-weight:600;'>■ {new_total*100:.1f}%</span>
        of total importance<br><br>
        <b>Existing features</b><br>
        <span style='color:#0C447C;font-weight:600;'>■ {old_total*100:.1f}%</span>
        of total importance<br><br>
        <b>Top 3 features:</b><br>
        {top3_html}<br><br>
        Recent precipitation memory and lagged temperature dominate the signal —
        consistent with the physical drivers of the SPEI-3 water balance.
        </div>
        """, unsafe_allow_html=True)


elif active_tab == "Live prediction":
    st.markdown("### 🎯 Live drought class prediction")

    if not MODEL_AVAILABLE:
        st.info("""
        **Model not loaded.** To enable live predictions:
        1. Run `python3 generate_dashboard_data.py` to build the synthetic dataset
        2. Run `python3 train_model.py` to train the model and write the `.pkl` files to `models/`
        3. Restart the dashboard

        The prediction panel below shows a demo with example values.
        """)

    st.markdown("""
    <div class='insight-box'>
    Enter climate values for a station-month observation.
    The model returns a calibrated probability for each of the 7 drought severity classes.
    </div>
    """, unsafe_allow_html=True)

    # ── Input panel ──────────────────────────────────────────────────────────
    col_inp1, col_inp2, col_inp3 = st.columns(3)

    with col_inp1:
        st.markdown("**📍 Station & time**")
        pred_station = st.selectbox("Station", sorted(STATION_COORDS.keys()),
                                    index=sorted(STATION_COORDS.keys()).index("Chisinau"))
        pred_month   = st.slider("Month", 1, 12, 7,
                                  format="%d (%s)" % (7, MONTH_LABELS[6]))
        pred_year    = st.slider("Year (for context only)", 1961, 2020, 2007)

    with col_inp2:
        st.markdown("**🌡️ Temperature (°C)**")
        pred_t_med = st.slider("Mean temperature",   -15.0, 27.0,  22.0, 0.1)
        pred_t_min = st.slider("Min temperature",    -20.0, 21.0,  15.0, 0.1)
        pred_t_max = st.slider("Max temperature",    -11.0, 35.0,  30.0, 0.1)

    with col_inp3:
        st.markdown("**🌧️ Precipitation (mm)**")
        pred_precip = st.slider("Monthly total",    0.0, 353.0,  8.0, 1.0)
        pred_pr_lag1= st.slider("Prev month precip", 0.0, 353.0, 15.0, 1.0)
        pred_pr_lag3= st.slider("3-month lag precip",0.0, 353.0, 20.0, 1.0)

    # Derive cyclical month encoding and other features
    month_sin = np.sin(2 * np.pi * pred_month / 12)
    month_cos = np.cos(2 * np.pi * pred_month / 12)
    lat, lon   = STATION_COORDS[pred_station]

    # Load historical station data to derive anomalies
    @st.cache_data
    def get_station_monthly_means():
        from utils.data_loader import load_climatology
        clim = load_climatology()
        return clim[clim.period == "full"][["station","month",
                                            "precip_monthly_mean","t_med_mean"]].copy()
    means = get_station_monthly_means()
    st_means = means[(means.station==pred_station) & (means.month==pred_month)]
    if len(st_means) > 0:
        t_med_anom   = pred_t_med  - st_means.iloc[0].t_med_mean
        precip_anom  = pred_precip - st_means.iloc[0].precip_monthly_mean
        t_min_anom   = pred_t_min  - (st_means.iloc[0].t_med_mean - 3.5)
        t_max_anom   = pred_t_max  - (st_means.iloc[0].t_med_mean + 5.0)
    else:
        t_med_anom = precip_anom = t_min_anom = t_max_anom = 0.0

    # Build feature vector matching v2 pipeline exactly
    features_v2 = [
        "precip","t_med","t_min","t_max","month_sin","month_cos",
        "t_med_anom","t_min_anom","t_max_anom","precip_anom",
        "precip_lag1","precip_lag3","t_med_lag1","t_med_lag3","precip_roll3",
        "precip_lag6","precip_lag12","t_med_lag6",
        "precip_roll6","precip_roll12","t_med_anom_roll3",
        "station_lat","station_lon","aridity_idx",
    ]

    precip_roll3 = (pred_precip + pred_pr_lag1 + pred_pr_lag3) / 3
    aridity_idx  = pred_precip / max(pred_t_med + 10, 0.5)

    x_dict = {
        "precip": pred_precip, "t_med": pred_t_med,
        "t_min": pred_t_min, "t_max": pred_t_max,
        "month_sin": month_sin, "month_cos": month_cos,
        "t_med_anom": t_med_anom, "t_min_anom": t_min_anom,
        "t_max_anom": t_max_anom, "precip_anom": precip_anom,
        "precip_lag1": pred_pr_lag1, "precip_lag3": pred_pr_lag3,
        "t_med_lag1": pred_t_med, "t_med_lag3": pred_t_med,
        "precip_roll3": precip_roll3,
        "precip_lag6": pred_pr_lag1, "precip_lag12": pred_pr_lag3,
        "t_med_lag6": pred_t_med, "precip_roll6": precip_roll3,
        "precip_roll12": precip_roll3,
        "t_med_anom_roll3": t_med_anom,
        "station_lat": lat, "station_lon": lon,
        "aridity_idx": aridity_idx,
    }

    X_input = pd.DataFrame([x_dict])[features_v2]

    st.markdown("---")
    st.markdown("### 📊 Predicted class probabilities")

    if MODEL_AVAILABLE:
        try:
            X_imp = imputer.transform(X_input)
            X_sc  = scaler.transform(X_imp)
            probs = model.predict_proba(X_sc)[0]
            pred_class = CLASS_ORDER[np.argmax(probs)]
        except Exception as e:
            st.error(f"Prediction error: {e}")
            probs = np.ones(7) / 7
            pred_class = "Normal"
    else:
        # Demo: simulate drought scenario from low precip + high temp
        demo_dry = max(0, (30 - pred_precip) / 30)
        base = np.array([0.02, 0.05, 0.10, 0.65, 0.10, 0.05, 0.03])
        shift = np.array([demo_dry*0.2, demo_dry*0.15, demo_dry*0.1,
                          -demo_dry*0.45, -0.02, -0.01, -0.01])
        probs = np.clip(base + shift, 0, 1)
        probs /= probs.sum()
        pred_class = CLASS_ORDER[np.argmax(probs)]

    # Predicted class badge
    pred_color = CLASS_COLORS[pred_class]
    st.markdown(f"""
    <div style='background:{pred_color}22; border:1px solid {pred_color};
                border-radius:8px; padding:12px 18px; margin-bottom:16px;
                display:flex; align-items:center; gap:12px;'>
      <div style='font-size:1.4rem;'>{'🔴' if 'Dry' in pred_class else '🔵' if 'Wet' in pred_class else '🟢'}</div>
      <div>
        <div style='font-size:0.75rem; color:#8A9BAE;'>Predicted class</div>
        <div style='font-size:1.2rem; font-weight:600; color:{pred_color};'>{pred_class}</div>
        <div style='font-size:0.8rem; color:#8A9BAE;'>
          Confidence: {max(probs)*100:.1f}% · {'Calibrated model' if MODEL_AVAILABLE else 'Demo values'}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    for cls, prob in zip(CLASS_ORDER, probs):
        color = CLASS_COLORS[cls]
        width = prob * 100
        is_pred = cls == pred_class
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
          <div style='width:130px; font-size:0.78rem; color:{"#E8EAED" if is_pred else "#8A9BAE"};
                      font-weight:{"600" if is_pred else "400"};'>{cls}</div>
          <div class='prob-bar-wrap' style='flex:1;'>
            <div class='prob-bar' style='width:{width:.1f}%;background:{color};'></div>
          </div>
          <div style='width:48px; text-align:right; font-size:0.82rem;
                      color:{"#E8EAED" if is_pred else "#8A9BAE"};
                      font-weight:{"600" if is_pred else "400"};'>{prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Plotly bar chart version
    st.markdown("")
    fig_pred = go.Figure(go.Bar(
        x=CLASS_ORDER,
        y=[p * 100 for p in probs],
        marker_color=[CLASS_COLORS[c] for c in CLASS_ORDER],
        marker_line_color=["white" if c == pred_class else "rgba(0,0,0,0)"
                           for c in CLASS_ORDER],
        marker_line_width=[3 if c == pred_class else 0 for c in CLASS_ORDER],
        text=[f"{p*100:.1f}%" for p in probs],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
    ))
    fig_pred.update_layout(
        **layout(
            title=dict(
                text=f"<b>Calibrated probability distribution — {pred_station}, "
                     f"{MONTH_LABELS[pred_month-1]}</b>",
                font=dict(size=13, color=TEXT)),
            xaxis=dict(gridcolor=BORDER, tickangle=-20),
            yaxis=dict(title="Probability (%)", gridcolor=BORDER, range=[0, 100]),
            showlegend=False,
            height=280,
        )
    )
    st.plotly_chart(fig_pred, use_container_width=True)
