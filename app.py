
import io
import json
import base64
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance
import joblib

# =========================
# Load model assets
# =========================
@st.cache_resource
def load_assets():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
    return model, scaler, expected_columns

model, scaler, expected_columns = load_assets()

# =========================
# UI: App layout
# =========================
st.set_page_config(
    page_title="Heart Stroke Prediction • Priyanka",
    page_icon="🫀",
    layout="wide"
)

st.title("💓 Heart Stroke Risk by Priyanka")
st.caption("Industry-style app with probability, explanations, comparisons, lifestyle simulation, and batch scoring.")

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    ["Single Prediction", "Batch Prediction", "About & Tips"],
    index=0
)

# Helpful maps for encoding
SEX_OPTS = ["M", "F"]
CP_OPTS = ["ATA", "NAP", "TA", "ASY"]
ECG_OPTS = ["Normal", "ST", "LVH"]
ANGINA_OPTS = ["Y", "N"]
SLOPE_OPTS = ["Up", "Flat", "Down"]

# Healthy baseline for radar chart (rough guides; you can tweak)
HEALTHY_BASELINE = {
    "Age": 35,
    "RestingBP": 120,
    "Cholesterol": 180,
    "MaxHR": 170,
    "Oldpeak": 0.5
}

# Ranges for radar normalization
RANGES = {
    "Age": (18, 100),
    "RestingBP": (80, 200),
    "Cholesterol": (100, 600),
    "MaxHR": (60, 220),
    "Oldpeak": (0.0, 6.0)
}

# =========================
# Encoding utilities
# =========================
def encode_single_record(
    age, sex, chest_pain, resting_bp, cholesterol,
    fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
) -> pd.DataFrame:
    """
    Builds a 1-row DataFrame with all expected_columns, filling one-hot columns correctly.
    """
    # start with all zeros
    data = {col: 0 for col in expected_columns}

    # numeric/base columns (must match your training pipeline naming)
    base_map = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak
    }
    for k, v in base_map.items():
        if k in data:
            data[k] = v

    # one-hot columns present in your pipeline
    # names follow your scheme: Sex_M, Sex_F, ChestPainType_ASY, ...
    one_hot_keys = [
        f"Sex_{sex}",
        f"ChestPainType_{chest_pain}",
        f"RestingECG_{resting_ecg}",
        f"ExerciseAngina_{exercise_angina}",
        f"ST_Slope_{st_slope}"
    ]
    for k in one_hot_keys:
        if k in data:
            data[k] = 1

    df = pd.DataFrame([data], columns=expected_columns)
    return df

def encode_batch_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Expects raw columns (not one-hot):
    ['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
    Returns encoded DataFrame with expected_columns.
    """
    out = []
    for _, row in df_raw.iterrows():
        row_df = encode_single_record(
            age=int(row["Age"]),
            sex=str(row["Sex"]),
            chest_pain=str(row["ChestPainType"]),
            resting_bp=int(row["RestingBP"]),
            cholesterol=int(row["Cholesterol"]),
            fasting_bs=int(row["FastingBS"]),
            resting_ecg=str(row["RestingECG"]),
            max_hr=int(row["MaxHR"]),
            exercise_angina=str(row["ExerciseAngina"]),
            oldpeak=float(row["Oldpeak"]),
            st_slope=str(row["ST_Slope"]),
        )
        out.append(row_df)
    return pd.concat(out, ignore_index=True)

# =========================
# Modeling helpers
# =========================
def predict_with_proba(X_scaled: np.ndarray):
    """
    Returns (pred_class, pred_proba_float between 0 and 1).
    Works if model has predict_proba; otherwise uses distance-based proxy for KNN fallback.
    """
    pred = model.predict(X_scaled)
    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X_scaled)
        # assume positive class is 1 at index 1
        proba = p[:, 1]
    else:
        # Fallback: distance to neighbors → crude probability proxy
        try:
            nbrs = model.kneighbors(X_scaled, return_distance=True)[0]
            # smaller distance → higher risk proxy; transform to 0..1
            inv = 1 / (1 + nbrs.mean(axis=1))
            proba = inv / inv.max()
        except Exception:
            proba = np.array([np.nan] * X_scaled.shape[0])

    return pred, proba

def local_permutation_importance(model, scaler, x_row: pd.DataFrame, repeats: int = 20, seed: int = 42):
    """
    Local explanation: create a small synthetic neighborhood around x_row,
    compute permutation importance. Returns DataFrame of top features.
    """
    rng = np.random.RandomState(seed)
    # Build a small cloud around the current point
    X = np.tile(x_row.values, (80, 1)).astype(float)
    noise = rng.normal(loc=0.0, scale=0.05, size=X.shape)  # small noise
    X_noisy = X + noise

    X_noisy_scaled = scaler.transform(pd.DataFrame(X_noisy, columns=x_row.columns))
    y_hat = model.predict(X_noisy_scaled)

    # We need a scoring function; use predicted probability of class 1 if available
    if hasattr(model, "predict_proba"):
        base_scores = model.predict_proba(X_noisy_scaled)[:, 1]
    else:
        base_scores = (y_hat == 1).astype(float)

    # Define a simple scorer: correlation with base_scores
    def scorer(clf, X_scaled, y=None):
        if hasattr(clf, "predict_proba"):
            s = clf.predict_proba(X_scaled)[:, 1]
        else:
            s = (clf.predict(X_scaled) == 1).astype(float)
        # R2-like measure between s and base_scores (both length n)
        # Here use 1 - MSE for stability
        return 1.0 - np.mean((s - base_scores) ** 2)

    # permutation_importance expects X in the same representation used by the model
    result = permutation_importance(
        model,
        X_noisy_scaled,
        y=base_scores,  # placeholder, not used by scorer
        scoring=scorer,
        n_repeats=repeats,
        random_state=seed
    )
    importances = pd.DataFrame({
        "feature": x_row.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
    return importances

def normalize_for_radar(value, low, high):
    # Clip to range then min-max to 0..1
    v = max(min(value, high), low)
    return (v - low) / (high - low + 1e-9)

def radar_chart(current, baseline):
    dims = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    cur_vals = [normalize_for_radar(current[d], *RANGES[d]) for d in dims]
    base_vals = [normalize_for_radar(baseline[d], *RANGES[d]) for d in dims]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=cur_vals, theta=dims, fill='toself', name="You"))
    fig.add_trace(go.Scatterpolar(r=base_vals, theta=dims, fill='toself', name="Healthy Baseline"))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, height=420)
    return fig

def risk_badge(prob):
    if np.isnan(prob):
        return "❔ Unable to estimate probability"
    if prob >= 0.7:
        return "🔴 High Risk"
    elif prob >= 0.4:
        return "🟠 Moderate Risk"
    else:
        return "🟢 Low Risk"

def build_html_report(inputs_dict, pred_label, prob_pct, top_factors_df):
    html = f"""
    <html>
      <head><meta charset="utf-8"><title>Heart Health Report</title></head>
      <body>
        <h1>Heart Health Report</h1>
        <h2>Prediction</h2>
        <p><b>Class:</b> {'High Risk' if pred_label==1 else 'Low Risk'}</p>
        <p><b>Probability:</b> {prob_pct:.2f}%</p>
        <h2>Inputs</h2>
        <pre style="font-size:14px">{json.dumps(inputs_dict, indent=2)}</pre>
        <h2>Top Contributing Factors (local importance)</h2>
        <table border="1" cellpadding="6" cellspacing="0">
          <tr><th>Feature</th><th>Importance</th></tr>
          {''.join([f"<tr><td>{r.feature}</td><td>{r.importance_mean:.4f}</td></tr>" for r in top_factors_df.head(10).itertuples()])}
        </table>
        <hr/>
        <p style="font-size:12px;color:#555">
        This report is generated by a machine learning model and is for informational purposes only. It is not a medical diagnosis.
        </p>
      </body>
    </html>
    """
    return html.encode("utf-8")

# =========================
# PAGE: Single Prediction
# =========================
if page == "Single Prediction":
    st.subheader("Single Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.slider("Age", 18, 100, 40, help="Age in years")
        sex = st.selectbox("Sex", SEX_OPTS, index=0)
        chest_pain = st.selectbox("Chest Pain Type", CP_OPTS, index=0, help="ATA/NAP/TA/ASY")
        resting_bp = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=200, value=120, step=1)
    with c2:
        cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=1)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
        resting_ecg = st.selectbox("Resting ECG", ECG_OPTS, index=0)
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    with c3:
        exercise_angina = st.selectbox("Exercise-Induced Angina", ANGINA_OPTS, index=1)
        oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, step=0.1)
        st_slope = st.selectbox("ST Slope", SLOPE_OPTS, index=0)

    # Inline warnings for extreme values
    warn = []
    if cholesterol >= 240: warn.append("High cholesterol detected (≥240).")
    if resting_bp >= 140: warn.append("Elevated blood pressure (≥140).")
    if max_hr <= 100: warn.append("Unusually low MaxHR; check measurement context.")
    if oldpeak >= 2.0: warn.append("ST depression is elevated (≥2.0).")

    if warn:
        st.warning(" | ".join(warn))

    # Build encoded row and scale
    input_df = encode_single_record(
        age, sex, chest_pain, resting_bp, cholesterol,
        fasting_bs, resting_ecg, max_hr, exercise_angina, oldpeak, st_slope
    )
    scaled = scaler.transform(input_df)

    # Predict
    pred, prob = predict_with_proba(scaled)
    prob_val = float(prob[0]) if prob is not None else np.nan
    prob_pct = float(prob_val * 100) if not np.isnan(prob_val) else float("nan")

    # Output cards
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Prediction", "High Risk" if pred[0] == 1 else "Low Risk")
    with k2:
        if np.isnan(prob_val):
            st.metric("Risk Probability", "N/A")
        else:
            st.metric("Risk Probability", f"{prob_pct:.2f}%")
    with k3:
        st.metric("Risk Level", risk_badge(prob_val))

    st.divider()

    # Explanations & Visualization
    left, right = st.columns([1, 1])

    with left:
        st.markdown("### 🔎 Top contributing factors (local)")
        try:
            importances = local_permutation_importance(model, scaler, input_df, repeats=20)
            st.dataframe(importances.head(10), use_container_width=True)
        except Exception as e:
            st.info(f"Could not compute local importance: {e}")

    with right:
        st.markdown("### 📈 You vs Healthy Baseline (Radar)")
        current = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak
        }
        fig = radar_chart(current, HEALTHY_BASELINE)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Lifestyle simulation
    st.markdown("### 🧪 Lifestyle Simulation (What-if)")
    sim_c1, sim_c2, sim_c3, sim_c4 = st.columns(4)
    with sim_c1:
        chol_delta = st.slider("Reduce Cholesterol by (mg/dL)", 0, 150, 20, step=5)
    with sim_c2:
        bp_delta = st.slider("Reduce Resting BP by (mm Hg)", 0, 60, 10, step=5)
    with sim_c3:
        hr_delta = st.slider("Increase MaxHR by", 0, 40, 5, step=1)
    with sim_c4:
        oldpeak_delta = st.slider("Reduce Oldpeak by", 0.0, 2.0, 0.2, step=0.1)

    sim_vals = {
        "Age": age,  # unchanged
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": max(80, resting_bp - bp_delta),
        "Cholesterol": max(100, cholesterol - chol_delta),
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": min(220, max_hr + hr_delta),
        "ExerciseAngina": exercise_angina,
        "Oldpeak": max(0.0, oldpeak - oldpeak_delta),
        "ST_Slope": st_slope
    }
    sim_df = encode_single_record(
        sim_vals["Age"], sim_vals["Sex"], sim_vals["ChestPainType"],
        sim_vals["RestingBP"], sim_vals["Cholesterol"], sim_vals["FastingBS"],
        sim_vals["RestingECG"], sim_vals["MaxHR"], sim_vals["ExerciseAngina"],
        sim_vals["Oldpeak"], sim_vals["ST_Slope"]
    )
    sim_scaled = scaler.transform(sim_df)
    sim_pred, sim_prob = predict_with_proba(sim_scaled)
    sim_prob_val = float(sim_prob[0]) if sim_prob is not None else np.nan

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("What-if Prediction", "High Risk" if sim_pred[0] == 1 else "Low Risk")
    with sc2:
        st.metric("What-if Probability", "N/A" if np.isnan(sim_prob_val) else f"{sim_prob_val*100:.2f}%")
    with sc3:
        st.metric("What-if Level", risk_badge(sim_prob_val))

    st.divider()

    # Download HTML report
    st.markdown("### ⬇️ Download Report")
    inputs_for_report = {
        "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
        "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak, "ST_Slope": st_slope
    }
    try:
        top_df = local_permutation_importance(model, scaler, input_df, repeats=10).head(10)
    except Exception:
        top_df = pd.DataFrame({"feature": [], "importance_mean": [], "importance_std": []})

    html_bytes = build_html_report(inputs_for_report, int(pred[0]), prob_pct if not np.isnan(prob_val) else 0.0, top_df)
    st.download_button(
        "Download HTML Report",
        data=html_bytes,
        file_name="heart_health_report.html",
        mime="text/html"
    )

# =========================
# PAGE: Batch Prediction
# =========================
elif page == "Batch Prediction":
    st.subheader("Batch Prediction (CSV Upload)")

    st.markdown("""
    **Template columns (raw):**  
    `Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope`
    """)

    # Download template
    if st.button("Download CSV Template"):
        template = pd.DataFrame([{
            "Age": 40, "Sex": "M", "ChestPainType": "ATA", "RestingBP": 120, "Cholesterol": 200,
            "FastingBS": 0, "RestingECG": "Normal", "MaxHR": 150, "ExerciseAngina": "N",
            "Oldpeak": 1.0, "ST_Slope": "Up"
        }])
        st.download_button(
            "Save Template CSV",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="heart_batch_template.csv",
            mime="text/csv"
        )

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        try:
            df_raw = pd.read_csv(file)
            st.write("Preview:", df_raw.head())
            # Basic validation
            required_cols = ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]
            missing = [c for c in required_cols if c not in df_raw.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                X_enc = encode_batch_raw(df_raw)
                X_scaled = scaler.transform(X_enc)
                y_pred, y_proba = predict_with_proba(X_scaled)

                out = df_raw.copy()
                out["Prediction"] = np.where(y_pred == 1, "High Risk", "Low Risk")
                if y_proba is not None:
                    out["Risk_Probability"] = (y_proba * 100).round(2)
                else:
                    out["Risk_Probability"] = np.nan

                st.success("Scored successfully.")
                st.dataframe(out, use_container_width=True)

                # Download results
                st.download_button(
                    "Download Results CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="heart_batch_results.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Failed to process file: {e}")

# =========================
# PAGE: About & Tips
# =========================
else:
    st.subheader("About & Tips")
    st.markdown("""
**What makes this app different from most:**  
- Shows **probability** and **clear risk level badges**  
- **Local explanations** (permutation-based) for *your* inputs  
- **Radar chart** comparing you vs a healthy baseline  
- **Lifestyle simulation** to see how improving metrics may change risk  
- **Batch predictions** for CSVs + **exportable HTML report**

**Important notes:**  
- This app is for **education**. It does **not** replace a medical diagnosis.  
- Feature scaling is applied exactly like in training (using your saved scaler).  
- One-hot encoding matches the model’s `expected_columns` to avoid column mismatch errors.

**CSV Tips:**  
- Ensure categorical values are from these sets:  
  - Sex: `M, F`  
  - ChestPainType: `ATA, NAP, TA, ASY`  
  - RestingECG: `Normal, ST, LVH`  
  - ExerciseAngina: `Y, N`  
  - ST_Slope: `Up, Flat, Down`
""")
