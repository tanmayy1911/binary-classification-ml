
import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
SCALER_PATH = ARTIFACTS / "scaler.joblib"
META_PATH = ARTIFACTS / "train_meta.json"

st.set_page_config(page_title="Binary Classification Demo", page_icon="⚙️", layout="centered")
st.title("⚙️ Binary Classification Demo")
st.caption("Trained with scikit-learn | Demo app")

if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not META_PATH.exists():
    st.error("Model artifacts not found. Run training first: `python -m src.train`")
    st.stop()

model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
meta = pd.read_json(META_PATH)

# JSON read via pandas returns series; handle feature names
with open(META_PATH, "r") as f:
    meta_json = f.read()
import json
meta = json.loads(meta_json)
feature_names = meta["feature_names"]
target_names = meta["target_names"]

st.subheader("Enter Feature Values")
with st.form("input_form"):
    values = []
    cols = st.columns(3)
    for i, fname in enumerate(feature_names):
        col = cols[i % 3]
        values.append(col.number_input(fname, value=0.0, format="%.6f"))
    submitted = st.form_submit_button("Predict")

if submitted:
    X = np.array(values, dtype=float).reshape(1, -1)
    Xs = scaler.transform(X)
    prob = model.predict_proba(Xs)[0, 1]
    pred = int(prob >= 0.5)
    st.success(f"Prediction: **{target_names[pred]}** (probability={prob:.3f})")
    st.progress(float(prob))

st.divider()
st.caption("Tip: Click the gear icon in the top-right of Streamlit to adjust settings.")
