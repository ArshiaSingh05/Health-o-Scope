import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, "dataset", "Cleaned_finalDataset.csv")
MODELS_DIR = os.path.join(ROOT, "models")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model(name):
    path = os.path.join(MODELS_DIR, name + ".joblib")
    if os.path.exists(path):
        return joblib.load(path)
    else:
        return None

df = load_data()

st.set_page_config(layout="wide", page_title="Health Risk Assessment", initial_sidebar_state="auto")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Home", "Predict", "EDA"])

# ---------- Helper functions ----------
def gauge_chart(value, title, vmin=0, vmax=100, steps=None, unit=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title} {unit}"},
        gauge={'axis': {'range': [vmin, vmax]},
               'bar': {'color': "darkblue"},
               'steps': steps}
    ))
    fig.update_layout(height=300)
    return fig

def percentile_overlay(series, user_value, title):
    fig = px.histogram(series.dropna(), nbins=40, opacity=0.8)
    fig.add_vline(x=user_value, line_color="red", line_width=3, annotation_text="You", annotation_position="top right")
    fig.update_layout(title=title, height=350)
    return fig

# ---------- HOME ----------
if page == "Home":
    st.title("Health Risk Assessment System")
    st.write("Choose a prediction card. Clicking a card opens only the inputs needed for that prediction.")
    cols = st.columns(3)
    cards = [
        ("High Blood Pressure", "model_high_bp"),
        ("High Cholesterol", "model_high_chol"),
        ("Heart Disease", "model_heart_disease"),
        ("Depression (PHQ-9)", "model_phq9_score"),
        ("Lifestyle Risk", "model_lifestyle_score"),
        ("BMI Prediction", "model_bmi"),
        ("LDL/HDL/Total cholesterol", "model_ldl")
    ]
    i = 0
    for label, mname in cards:
        with cols[i % 3]:
            if st.button(label):
                st.session_state["selected_card"] = (label, mname)
                st.experimental_rerun()
        i += 1

    st.markdown("---")
    st.subheader("Quick dataset stats")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (cleaned)", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Avg PHQ-9 score", round(df["phq9_score"].mean(),2))

# ---------- PREDICT ----------
elif page == "Predict":
    st.title("Make a prediction")
    # Determine selection: either session_state or show selector
    selected = st.session_state.get("selected_card", None)
    if not selected:
        sel = st.selectbox("Select a prediction", ["High Blood Pressure", "High Cholesterol",
                                                  "Heart Disease", "Depression (PHQ-9)",
                                                  "Lifestyle Risk", "BMI Prediction", "LDL/HDL/Total cholesterol"])
        # map to names
        name_map = {
            "High Blood Pressure":"model_high_bp",
            "High Cholesterol":"model_high_chol",
            "Heart Disease":"model_heart_disease",
            "Depression (PHQ-9)":"model_phq9_score",
            "Lifestyle Risk":"model_lifestyle_score",
            "BMI Prediction":"model_bmi",
            "LDL/HDL/Total cholesterol":"model_ldl"
        }
        selected = (sel, name_map[sel])

    st.write(f"Selected: **{selected[0]}**")
    model = load_model(selected[1])

    # Show only relevant inputs per selection
    # Common inputs
    col1, col2, col3 = st.columns(3)
    age = col1.number_input("Age (years)", min_value=0, max_value=120, value=30)
    gender = col1.selectbox("Gender", options=[1,2], format_func=lambda x: "Male" if x==1 else "Female")
    height_cm = col2.number_input("Height (cm)", min_value=50, max_value=250, value=170)
    weight_kg = col2.number_input("Weight (kg)", min_value=10, max_value=300, value=70)
    bmi = round(weight_kg / ((height_cm/100)**2), 1)
    col2.write(f"Computed BMI: **{bmi}**")
    waist = col3.number_input("Waist (cm)", min_value=30, max_value=200, value=80)
    # Lifestyle quick
    ever_smoke = col1.selectbox("Ever smoked?", options=[1,2], index=1, format_func=lambda x: "Yes" if x==1 else "No")
    current_smoker = col1.selectbox("Currently smoking?", options=[1,2], index=1, format_func=lambda x: "Yes" if x==1 else "No")
    ever_drink = col2.selectbox("Ever drink alcohol?", options=[1,2], index=1, format_func=lambda x: "Yes" if x==1 else "No")
    binge_drink = col3.number_input("Binge episodes (past period)", min_value=0, max_value=100, value=0)

    # Additional inputs depending on prediction
    if "Blood Pressure" in selected[0]:
        # we predict diagnosed_high_bp; user likely doesn't have physician diagnosis - we predict probability
        st.markdown("**Optional:** If you have recent blood pressure values, enter them to increase accuracy.")
        sys = st.number_input("Systolic (mmHg) - optional", value=0)
        dia = st.number_input("Diastolic (mmHg) - optional", value=0)

    if "Cholesterol" in selected[0] and "LDL/HDL" not in selected[0]:
        st.markdown("**Optional:** Provide lab values if known.")
        hdl_val = st.number_input("HDL (mg/dL)", value=0)
        ldl_val = st.number_input("LDL (mg/dL)", value=0)
        tc_val = st.number_input("Total Cholesterol (mg/dL)", value=0)

    if "Heart Disease" in selected[0]:
        st.info("This prediction uses lifestyle + demographics + BP/Cholesterol (if provided)")

    if "Depression" in selected[0]:
        st.write("Please answer the PHQ-9 style questions (0=Not at all, 3=Nearly every day).")
        dpq = {}
        phq_items = [
            "Little interest or pleasure in doing things",
            "Feeling down, depressed, or hopeless",
            "Trouble falling asleep, staying asleep, or sleeping too much",
            "Feeling tired or having little energy",
            "Poor appetite or overeating",
            "Feeling bad about yourself",
            "Trouble concentrating",
            "Moving or speaking slowly or being fidgety",
            "Thoughts that you would be better off dead"
        ]
        for idx, q in enumerate(phq_items, start=1):
            dpq[f"DPQ0{idx}"] = st.slider(q, 0, 3, 0)

    # If LDL/HDL/Total cholesterol prediction
    if "LDL/HDL/Total" in selected[0]:
        st.info("We will estimate LDL (or other labs) based on demographics and lifestyle.")

    # SUBMIT button
    if st.button("Predict"):
        # Prepare feature vector
        feat = {
            "age": age, "bmi": bmi, "waist": waist, "height": height_cm,
            "weight": weight_kg, "avg_sys": sys if 'sys' in locals() and sys>0 else df["avg_sys"].mean(),
            "avg_dia": dia if 'dia' in locals() and dia>0 else df["avg_dia"].mean(),
            "gender": gender, "race": df["race"].mode()[0], "education": df["education"].mode()[0],
            "ever_smoked": ever_smoke, "current_smoker": current_smoker, "binge_drink": binge_drink,
            "ever_drink": ever_drink
        }
        feat["lifestyle_score"] = 0
        if feat["ever_smoked"]==1: feat["lifestyle_score"]+=1
        if feat["ever_drink"]==1: feat["lifestyle_score"]+=1
        if feat["binge_drink"]>0: feat["lifestyle_score"]+=1

        X_user = pd.DataFrame([feat])

        # Special case: depression PHQ direct scoring
        if "Depression" in selected[0]:
            phq_score = sum(dpq.values())
            st.metric("PHQ-9 score", phq_score)
            cat = "None" if phq_score<=4 else ("Mild" if phq_score<=9 else ("Moderate" if phq_score<=14 else ("Moderately Severe" if phq_score<=19 else "Severe")))
            st.write("Category:", cat)
            # also attempt model estimate if available
            model_phq = load_model("model_phq9_score")
            if model_phq:
                pred = model_phq.predict(X_user)[0]
                st.write("Model-estimated PHQ-9 (regression):", round(pred,2))
                fig = gauge_chart(pred, "Estimated PHQ-9 Score", vmin=0, vmax=27, steps=[
                    {'range':[0,4], 'color':'lightgreen'},
                    {'range':[5,9], 'color':'yellow'},
                    {'range':[10,14], 'color':'orange'},
                    {'range':[15,19], 'color':'red'},
                    {'range':[20,27], 'color':'darkred'}
                ])
                st.plotly_chart(fig)
            st.stop()

        # For other predictions:
        model = load_model(selected[1])
        if model is None:
            st.error("Model not found. Please run training script first.")
            st.stop()

        # Predict / postprocess
        if "High Blood Pressure" in selected[0] or "Cholesterol" in selected[0] or "Heart Disease" in selected[0] or "Lifestyle Risk" in selected[0]:
            prob = model.predict_proba(X_user)
            # prefer class 1 probability if available
            if prob.shape[1] == 2:
                p1 = prob[0,1]*100
            else:
                # if multiclass (lifestyle), get weighted score
                p1 = np.dot(np.arange(prob.shape[1]), prob[0]) * (100/(prob.shape[1]-1))
            st.metric("Risk (0-100%)", f"{p1:.1f}%")
            # show gauge
            fig = gauge_chart(p1, selected[0], vmin=0, vmax=100, steps=[
                {'range':[0,33],'color':'lightgreen'},
                {'range':[33,66],'color':'yellow'},
                {'range':[66,100],'color':'red'}
            ])
            st.plotly_chart(fig)

            # show user position on population histogram of a related variable
            if "Blood Pressure" in selected[0]:
                st.plotly_chart(percentile_overlay(df["avg_sys"], X_user["avg_sys"].iloc[0], "Systolic BP distribution"))
            if "Cholesterol" in selected[0]:
                st.plotly_chart(percentile_overlay(df["total_cholesterol"], X_user.get("total_cholesterol", df["total_cholesterol"].mean()), "Total Cholesterol distribution"))
            if "Lifestyle" in selected[0]:
                st.plotly_chart(percentile_overlay(df["lifestyle_score"], feat["lifestyle_score"], "Lifestyle score distribution"))

        # Regression predictions (bmi, ldl/hdl/total, phq regression handled above)
        elif "BMI" in selected[0] or "LDL/HDL/Total" in selected[0] or "BMI Prediction" in selected[0]:
            pred = model.predict(X_user)[0]
            st.metric(f"{selected[0]} predicted value", round(pred,2))
            # gauge scales differ
            if "BMI" in selected[0]:
                fig = gauge_chart(pred, "Predicted BMI", vmin=10, vmax=50, steps=[
                    {'range':[10,24.9],'color':'lightgreen'},
                    {'range':[25,29.9],'color':'yellow'},
                    {'range':[30,50],'color':'red'}
                ])
                st.plotly_chart(fig)
            else:
                st.plotly_chart(gauge_chart(pred, selected[0], vmin=0, vmax=200))

# ---------- EDA ----------
elif page == "EDA":
    st.title("Dataset EDA")
    st.write("Interactive exploration. You can overlay your personal values when you visit the Predict page and submit a prediction.")
    # basic stats
    st.subheader("Distributions")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(df, x="bmi", nbins=50, title="BMI distribution"))
    c2.plotly_chart(px.histogram(df, x="phq9_score", nbins=30, title="PHQ-9 distribution"))

    st.subheader("Relationships")
    fig = px.scatter(df, x="age", y="avg_sys", color="gender", hover_data=["bmi"], title="Age vs Avg Systolic BP")
    st.plotly_chart(fig)

    st.subheader("Alcohol vs Total Cholesterol")
    fig = px.box(df, x="ever_drink", y="total_cholesterol", points="outliers", title="Alcohol vs Total Cholesterol")
    st.plotly_chart(fig)

    st.subheader("Smoking vs PHQ-9")
    fig = px.box(df, x="ever_smoked", y="phq9_score", title="Smoking vs PHQ-9 score")
    st.plotly_chart(fig)

    st.markdown("### Show your predicted point on charts")
    st.write("First make a prediction on the Predict page — the app will overlay your point on the histograms/scatter plots for comparison.")
