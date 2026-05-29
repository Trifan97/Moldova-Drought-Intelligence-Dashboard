"""
train_model.py
--------------
Trains the 7-class SPEI drought-severity classifier on the synthetic dashboard
dataset and saves the artefacts the ML Model Explorer page loads.

Replicates the validated v2 pipeline from the supervised repo:
    median imputation → SMOTE (30% of Normal) → StandardScaler
    → HistGradientBoosting (balanced sample_weight)
    → isotonic probability calibration (CalibratedClassifierCV, cv=3)

Label encoding uses a FIXED class order matching utils.constants.CLASS_ORDER,
so the calibrated model's predict_proba columns line up 1-to-1 with the order
the dashboard expects.

Outputs (→ models/):
    v2_best_model_calibrated.pkl   calibrated classifier (used by dashboard)
    v2_imputer.pkl                 fitted SimpleImputer(median)
    v2_scaler.pkl                  fitted StandardScaler
    v2_label_encoder.pkl           LabelEncoder with CLASS_ORDER ordering
    v2_feature_meta.pkl            feature lists + class names + coords
    metrics.json                   held-out test metrics (for the dashboard KPIs)

Usage
-----
    python3 train_model.py
"""

import os
import json
from collections import Counter

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)
from sklearn.inspection import permutation_importance

from utils.constants import CLASS_ORDER, STATION_COORDS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "master_dataset_v2.csv")
MODELS = os.path.join(HERE, "models")

FEATURES_V1 = ["precip", "t_med", "t_min", "t_max", "month_sin", "month_cos",
               "t_med_anom", "t_min_anom", "t_max_anom", "precip_anom",
               "precip_lag1", "precip_lag3", "t_med_lag1", "t_med_lag3",
               "precip_roll3"]
NEW_FEATURES = ["precip_lag6", "precip_lag12", "t_med_lag6",
                "precip_roll6", "precip_roll12", "t_med_anom_roll3",
                "station_lat", "station_lon", "aridity_idx"]
FEATURES_V2 = FEATURES_V1 + NEW_FEATURES


def main() -> None:
    os.makedirs(MODELS, exist_ok=True)
    df = pd.read_csv(DATA)
    print(f"Loaded {DATA}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    X = df[FEATURES_V2].copy()

    # Fixed-order label encoding (matches dashboard CLASS_ORDER) -----------------
    le = LabelEncoder()
    le.classes_ = np.array(CLASS_ORDER)
    y = le.transform(df["drought_class_7"])

    print("\nClass distribution:")
    for cls, code in zip(le.classes_, le.transform(le.classes_)):
        n = (y == code).sum()
        print(f"  {code}  {cls:<16} n={n:5d} ({100*n/len(y):.2f}%)")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"\nTrain {len(X_tr):,} | Test {len(X_te):,}")

    # Impute (median) ------------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    # SMOTE — minority classes up to 30% of Normal (class index from CLASS_ORDER)-
    from imblearn.over_sampling import SMOTE
    normal_idx = CLASS_ORDER.index("Normal")
    normal_count = Counter(y_tr)[normal_idx]
    target = int(normal_count * 0.30)
    strategy = {k: max(v, target) for k, v in Counter(y_tr).items() if k != normal_idx}
    strategy[normal_idx] = normal_count
    X_tr_sm, y_tr_sm = SMOTE(sampling_strategy=strategy, k_neighbors=5,
                             random_state=42).fit_resample(X_tr_imp, y_tr)
    print(f"SMOTE target/minority class: {target} (30% of Normal={normal_count})")
    print(f"Rows after SMOTE: {len(X_tr_sm):,}")

    # Scale ----------------------------------------------------------------------
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_sm)
    X_te_sc = scaler.transform(X_te_imp)

    sample_weights = compute_sample_weight("balanced", y_tr_sm)

    # Base model -----------------------------------------------------------------
    base = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=10, random_state=42,
    )

    # CV macro-F1 on the SMOTE training set --------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base, X_tr_sc, y_tr_sm,
                                cv=cv, scoring="f1_macro")
    print(f"\nCV macro-F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Calibrated final model -----------------------------------------------------
    print("Fitting isotonic-calibrated model (cv=3) ...")
    calibrated = CalibratedClassifierCV(estimator=base, method="isotonic", cv=3)
    calibrated.fit(X_tr_sc, y_tr_sm, sample_weight=sample_weights)

    y_pred = calibrated.predict(X_te_sc)
    y_prob = calibrated.predict_proba(X_te_sc)
    y_te_bin = label_binarize(y_te, classes=list(range(len(CLASS_ORDER))))

    macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te_bin, y_prob, multi_class="ovr", average="macro")
    report = classification_report(
        y_te, y_pred, target_names=CLASS_ORDER,
        output_dict=True, zero_division=0)

    print(f"\nHeld-out test metrics:")
    print(f"  Macro F1    : {macro_f1:.4f}")
    print(f"  Weighted F1 : {weighted_f1:.4f}")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  ROC-AUC ovr : {auc:.4f}")
    print("\n" + classification_report(
        y_te, y_pred, target_names=CLASS_ORDER, zero_division=0))

    # Permutation feature importance (on held-out test set) ----------------------
    print("Computing permutation feature importance ...")
    perm = permutation_importance(
        calibrated, X_te_sc, y_te, scoring="f1_macro",
        n_repeats=5, random_state=42, n_jobs=-1)
    imp = pd.Series(perm.importances_mean, index=FEATURES_V2)
    imp = imp.clip(lower=0)
    imp = (imp / imp.sum()) if imp.sum() > 0 else imp   # normalise to share of total
    feat_imp = {k: round(float(v), 4) for k, v in
                imp.sort_values(ascending=False).items()}

    # Save artefacts -------------------------------------------------------------
    joblib.dump(calibrated, os.path.join(MODELS, "v2_best_model_calibrated.pkl"))
    joblib.dump(imputer,    os.path.join(MODELS, "v2_imputer.pkl"))
    joblib.dump(scaler,     os.path.join(MODELS, "v2_scaler.pkl"))
    joblib.dump(le,         os.path.join(MODELS, "v2_label_encoder.pkl"))
    joblib.dump({
        "features_v2": FEATURES_V2,
        "features_v1": FEATURES_V1,
        "new_features": NEW_FEATURES,
        "class_names": CLASS_ORDER,
        "station_coords": STATION_COORDS,
    }, os.path.join(MODELS, "v2_feature_meta.pkl"))

    metrics = {
        "model_name": "HistGradientBoosting + SMOTE 30% + balanced sample_weight",
        "macro_f1": round(macro_f1, 3),
        "weighted_f1": round(weighted_f1, 3),
        "roc_auc": round(auc, 3),
        "accuracy": round(acc, 3),
        "cv_f1_mean": round(float(cv_scores.mean()), 3),
        "cv_f1_std": round(float(cv_scores.std()), 3),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 2),
                "recall": round(report[cls]["recall"], 2),
                "f1": round(report[cls]["f1-score"], 2),
                "support": int(report[cls]["support"]),
            } for cls in CLASS_ORDER
        },
        "feature_importance": feat_imp,
        "new_features": NEW_FEATURES,
        "n_rows": int(len(df)),
        "smote_normal": int(normal_count),
        "smote_target": int(target),
    }
    with open(os.path.join(MODELS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved 5 pkl artefacts + metrics.json to {MODELS}")


if __name__ == "__main__":
    main()
