# dashboard/app.py
import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Optional: shap may be expensive; lazy import below where needed
# import shap

st.set_page_config(layout="wide", page_title="Predictive Maintenance Dashboard")

# ---------------------------
# Helper functions
# ---------------------------
def find_model_file(models_dir: Path):
    """Find joblib or pkl model file in models_dir."""
    candidates = list(models_dir.glob("rf_FD001.joblib")) + list(models_dir.glob("rf_FD001.pkl"))
    if candidates:
        return candidates[0]
    # fallback: any .joblib or .pkl
    for ext in ("*.joblib", "*.pkl"):
        found = list(models_dir.glob(ext))
        if found:
            return found[0]
    return None

def load_model_and_features(models_dir: Path):
    """Load model and feature list (json). Return (model, feature_cols)."""
    model_path = find_model_file(models_dir)
    if model_path is None:
        raise FileNotFoundError(f"No model file found in {models_dir}. Expected rf_FD001.joblib or rf_FD001.pkl")

    # load model
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model file {model_path}: {e}")

    # load features json
    features_json = models_dir / "rf_FD001_features.json"
    features_txt = models_dir / "rf_FD001_features.txt"
    feature_cols = None
    if features_json.exists():
        try:
            with open(features_json, "r", encoding="utf-8") as f:
                feature_cols = json.load(f)
        except Exception:
            feature_cols = None
    elif features_txt.exists():
        try:
            with open(features_txt, "r", encoding="utf-8") as f:
                feature_cols = [line.strip() for line in f if line.strip()]
        except Exception:
            feature_cols = None

    return model, feature_cols, model_path

def detect_id_and_time_cols(df: pd.DataFrame):
    """Try to detect engine id and time/cycle columns in dataframe."""
    id_candidates = ["unit", "engine", "id", "unit_id", "engine_id"]
    time_candidates = ["cycle", "time", "t", "timestamp"]

    unit_col = None
    time_col = None

    lower_cols = {c.lower(): c for c in df.columns}

    for cand in id_candidates:
        if cand in lower_cols:
            unit_col = lower_cols[cand]
            break

    for cand in time_candidates:
        if cand in lower_cols:
            time_col = lower_cols[cand]
            break

    return unit_col, time_col

def get_latest_sample_for_unit(df: pd.DataFrame, unit_col: str, time_col: str, unit_value):
    """Return a single-row DataFrame: the sample with max(time) for unit_value."""
    sub = df[df[unit_col] == unit_value]
    if sub.empty:
        return None
    if time_col:
        # choose max time
        idx = sub[time_col].idxmax()
        return sub.loc[[idx]].copy()
    else:
        # if no time, take last row in group as fallback
        return sub.tail(1).copy()

def safe_prepare_input(sample_df: pd.DataFrame, feature_cols):
    """Ensure sample_df contains feature_cols in same order. Fill missing columns with 0 (or warn)."""
    X = sample_df.copy()
    missing = [c for c in feature_cols if c not in X.columns]
    if missing:
        raise KeyError(f"Model expects features not present in sample: {missing}")
    # select columns in right order
    X_sel = X[feature_cols].astype(float)
    return X_sel

def plot_feature_importances_bar(feature_cols, importances, top_n=25):
    df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    df = df.sort_values("importance", ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(df))))
    ax.barh(df["feature"], df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Top model feature importances")
    plt.tight_layout()
    return fig

# ---------------------------
# Paths & load
# ---------------------------
ROOT = Path.cwd().resolve()  # assume you run streamlit from project root
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"
CSV_PATH = DATA_DIR / "train_features_FD001_no_leak.csv"

st.sidebar.title("Engine Selection")
st.sidebar.markdown("This dashboard loads a trained RandomForest and shows predictions for selected engine unit.")

# load model + features - show friendly errors to the user
try:
    model, feature_cols, model_file = load_model_and_features(MODELS_DIR)
    st.sidebar.success(f"Model loaded: {model_file.name}")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# load processed CSV
if not CSV_PATH.exists():
    st.sidebar.error(f"Processed CSV not found at {CSV_PATH}\nPlace your processed CSV here: {CSV_PATH}")
    st.stop()

try:
    df_all = pd.read_csv(CSV_PATH)
    st.sidebar.success(f"Processed CSV loaded ({len(df_all)} rows).")
except Exception as e:
    st.sidebar.error(f"Failed to read processed CSV: {e}")
    st.stop()

# detect unit/time columns
unit_col, time_col = detect_id_and_time_cols(df_all)

if unit_col is None:
    st.warning("Could not detect engine id column (e.g. 'unit' or 'engine'). Please ensure processed CSV contains an engine id column.")
    st.stop()

# list unique units
units = sorted(df_all[unit_col].unique().tolist())
selected_unit = st.sidebar.selectbox("Choose Engine (unit):", units, index=0)

# decision threshold
threshold = st.sidebar.slider("Failure Threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.01)

# main layout header
st.title("Optimize Manufacturing Operations with a Predictive Maintenance Model")
st.markdown("Predictive Maintenance Model for Aircraft Turbofan Engines — NASA FD001 dataset")

st.header("Current Risk Status")

st.info("Model loaded.")

if feature_cols is None:
    st.warning("Feature list not found (models/rf_FD001_features.json missing). The model expects a set of features. You can create the JSON by exporting feature columns from your training notebook.")
else:
    st.info(f"Model expects {len(feature_cols)} features (first 12 shown): {feature_cols[:12]}")

# Fetch latest sample for selected unit
sample_row = get_latest_sample_for_unit(df_all, unit_col, time_col, selected_unit)
if sample_row is None:
    st.error(f"No samples for unit {selected_unit}.")
    st.stop()

st.subheader("Latest Sensor Snapshot for this Engine")
# Show dataframe view (only first row)
st.dataframe(sample_row.T if sample_row.shape[1] > 15 else sample_row)

# Prepare input features
if feature_cols is None:
    st.error("Feature list not available. Cannot compute prediction. Create models/rf_FD001_features.json with the training feature list.")
    st.stop()

# check that all feature cols are in CSV
missing_features = [c for c in feature_cols if c not in df_all.columns]
if missing_features:
    st.warning("Model expects features that are missing in the current dataset. See Debug panel below.")
    st.write("Missing features sample (first 10):", missing_features[:10])
else:
    st.success("All expected features found in processed CSV.")

# attempt to build X and run model
try:
    X_sample = safe_prepare_input(sample_row, feature_cols)
except KeyError as ke:
    st.error(f"Could not prepare model input: {ke}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error preparing input: {e}")
    st.stop()

# prediction
try:
    # If classifier with predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_sample)[0]  # array of class probs
        # assume class 1 is failure; check classes_ if available
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            # find index of positive class 1 if present
            if 1 in classes:
                pos_idx = classes.index(1)
            else:
                pos_idx = 1 if len(classes) > 1 else 0
        else:
            pos_idx = 1 if len(proba) > 1 else 0
        failure_prob = float(proba[pos_idx])
        pred_class = int((failure_prob >= threshold))
    else:
        # fallback to predict -> treat >0.5 as failure (not ideal)
        pred = model.predict(X_sample)[0]
        failure_prob = np.nan
        pred_class = int(pred)
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Display results
col1, col2, col3 = st.columns([1,1,1])
col1.metric("Predicted (class)", f"{pred_class} (1=failure, 0=healthy)")
col2.metric("Failure probability (class 1)", f"{failure_prob if not np.isnan(failure_prob) else 'NA'}")
col3.metric("Decision Threshold", f"{threshold:.2f}")

if pred_class == 1:
    st.error("Engine is predicted to be AT RISK (class 1) at the chosen threshold.")
else:
    st.success("Engine is predicted to be Healthy (class 0) at the chosen threshold.")

# SHAP explanation - try but guard against version/mismatch errors
st.header("Why did the model make this prediction? (SHAP explanation)")

shap_available = False
try:
    import shap
    shap_available = True
except Exception:
    shap_available = False
    st.info("SHAP not installed or failed to import. Install shap to enable per-sample explanations: pip install shap")

if shap_available:
    try:
        # Choose TreeExplainer for tree-based models
        explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
        # shap_values for classification: shap_values is list-like; for single sample use explainer.shap_values
        shap_values = explainer.shap_values(X_sample)
        # For binary classification shap_values may be [class0_arr, class1_arr]
        # we'll try to show explanation for class 1 (failure) if available
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            shap_target = 1
            shap_vals_for_failure = shap_values[shap_target][0]  # single sample
            # create bar plot of absolute SHAP values (top features)
            abs_shap = pd.Series(np.abs(shap_vals_for_failure), index=feature_cols)
            top = abs_shap.sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(top))))
            ax.barh(top.index[::-1], top.values[::-1])
            ax.set_xlabel("Absolute SHAP value")
            ax.set_title("Top SHAP features (impact on model output for class=1)")
            st.pyplot(fig)
        else:
            # shap_values is single array (regression or single-output)
            shap_vals = shap_values
            abs_shap = pd.Series(np.abs(shap_vals[0]), index=feature_cols) if shap_vals.ndim == 2 else pd.Series(np.abs(shap_vals), index=feature_cols)
            top = abs_shap.sort_values(ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(8, max(4, 0.25 * len(top))))
            ax.barh(top.index[::-1], top.values[::-1])
            ax.set_xlabel("Absolute SHAP value")
            ax.set_title("Top SHAP features (impact on model output)")
            st.pyplot(fig)
    except Exception as e:
        st.warning("SHAP explanation failed: " + str(e))
        st.info("Falling back to feature importances (global). See Debug panel for details.")
        # fallback to feature importance plot if possible below
        shap_available = False

# Global feature importance (fallback or additionally)
st.header("Model feature importance (if available)")
try:
    if hasattr(model, "feature_importances_"):
        fig = plot_feature_importances_bar(feature_cols, model.feature_importances_, top_n=25)
        st.pyplot(fig)
    else:
        st.info("Model does not expose feature_importances_. Nothing to show here.")
except Exception as e:
    st.error(f"Failed to plot feature importances: {e}")

# Debug / feature-check panel
st.markdown("---")
st.subheader("Debug / Feature-check (helpful for troubleshooting)")
st.write("Load steps & messages")

st.write(f"Loading model from: {model_file}")
st.write(f"Loaded feature list from JSON: {MODELS_DIR / 'rf_FD001_features.json' if (MODELS_DIR / 'rf_FD001_features.json').exists() else '(not found)'}")
st.write(f"Processed CSV path: {CSV_PATH}")
st.write(f"CSV exists: {CSV_PATH.exists()}")
st.write(f"Detected id column: {unit_col}")
st.write(f"Detected time column: {time_col}")

st.write("Feature list (first 30):")
st.json(feature_cols[:30] if feature_cols else [])

if missing_features:
    st.error("Model expects features that are missing from the processed CSV. You need to regenerate `train_features_FD001_no_leak.csv` with the same engineered features that were used to train the model, or update the feature JSON to match current CSV.")
    st.write("Missing features (first 30):")
    st.json(missing_features[:30])

st.write("If everything looks correct but SHAP fails, try installing a matching SHAP version and/or using `feature_perturbation='interventional'` as done above.")

st.markdown("### End of dashboard")
