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

    # ---- Card Styling ----
    st.markdown("""
        <style>
        .card {
            padding: 20px;
            border-radius: 14px;
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            text-align: center;
            cursor: pointer;
            transition: 0.2s ease;
            width: 260px;
            height: 160px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin: auto;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 22px rgba(0,0,0,0.18);
        }
        .card-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 6px;
        }
        .card-desc {
            font-size: 14px;
            color: #555;
        }
        </style>
    """, unsafe_allow_html=True)

    cards = [
        ("Blood Pressure Check", "Checks your BP-related health risk.", "model_high_bp"),
        ("Cholesterol Check", "Predicts overall cholesterol risk.", "model_high_chol"),
        ("Heart Health Check", "Estimates heart disease likelihood.", "model_heart_disease"),
        ("Mood & Mental Well-Being", "Assesses mood using PHQ-9 questionnaire.", "model_phq9_score"),
        ("Daily Habits Health Score", "Rates smoking & drinking lifestyle habits.", "model_lifestyle_score"),
        ("Body Weight Indicator (BMI)", "Estimates BMI from body measures.", "model_bmi"),
        ("Detailed Cholesterol Report", "Predicts LDL, HDL & Total Cholesterol.", "model_ldl"),
    ]

    # Create rows of 3
    rows = [cards[i:i+3] for i in range(0, len(cards), 3)]

    for row in rows:
        st.write("")              # spacing between rows
        cols = st.columns(3, gap="large")

        # If this is a short row (like only 1 card), center it
        offset = (3 - len(row)) // 2

        for idx, (title, desc, model_key) in enumerate(row):
            with cols[idx + offset]:
                card_html = f"""
                <div class="card" onclick="window.location.href='?page=Predict&card={model_key}'">
                    <div class="card-title">{title}</div>
                    <div class="card-desc">{desc}</div>
                </div>
                """
                st.markdown(card_html, unsafe_allow_html=True)

                st.button(f"_{model_key}", key=model_key, disabled=True)

        # Center the last single card if row has fewer than 3
        if len(row) < 3:
            for _ in range(3 - len(row)):
                cols[_].write("")

    st.markdown("---")

    # ------------------------
    # Quick Dataset Stats
    # ------------------------
    
    #st.subheader("Quick dataset stats")
    #c1, c2, c3 = st.columns(3)
    #c1.metric("Rows (cleaned)", df.shape[0])
    #c2.metric("Columns", df.shape[1])
    #c3.metric("Avg PHQ-9 score", round(df["phq9_score"].mean(),2))

# ---------- PREDICT ----------
elif page == "Predict":
    st.title("Make a prediction")

    params = st.query_params

    # Load selection from URL if available
    if "card" in params:
        model_key = params["card"]

        for title, desc, key in cards:
            if key == model_key:
                st.session_state["selected_card"] = (title, key)
                break

    selected = st.session_state.get("selected_card")

    if not selected:
        st.warning("Please select a prediction card from the Home page.")
        st.stop()

    st.subheader(f"Selected: **{selected[0]}**")
    model = load_model(selected[1])

    # -----------------------------
    # USER INPUTS
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    age = col1.number_input("Age (years)", 0, 120, 30)
    gender = col1.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

    height_cm = col2.number_input("Height (cm)", 50, 250, 170)
    weight_kg = col2.number_input("Weight (kg)", 10, 300, 70)
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
    col2.write(f"Computed BMI: **{bmi}**")

    waist = col3.number_input("Waist (cm)", 30, 200, 80)

    # Lifestyle
    ever_smoke = col1.selectbox("Ever smoked?", [1, 2], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    current_smoker = col1.selectbox("Currently smoking?", [1, 2], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    ever_drink = col2.selectbox("Ever drink alcohol?", [1, 2], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
    binge_drink = col3.number_input("Binge episodes (past period)", 0, 100, 0)

    # For BP models
    if "Blood Pressure" in selected[0]:
        sys = st.number_input("Systolic (optional)", 0, 250, 0)
        dia = st.number_input("Diastolic (optional)", 0, 200, 0)

    # Depression PHQ-9
    if "Depression" in selected[0]:
        st.write("Answer the PHQ-9 questions (0–3):")
        dpq = {}
        questions = [
            "Little interest or pleasure in doing things",
            "Feeling down or hopeless",
            "Trouble sleeping",
            "Feeling tired",
            "Poor appetite",
            "Feeling bad about yourself",
            "Trouble concentrating",
            "Moving or speaking slowly / fidgety",
            "Thoughts of self-harm"
        ]
        for i, q in enumerate(questions, 1):
            dpq[f"DPQ0{i}"] = st.slider(q, 0, 3, 0)

    # -----------------------------
    # On Predict Click
    # -----------------------------
    if st.button("Predict"):

        # Medical averages fallback
        avg_sys_val = sys if "sys" in locals() and sys > 0 else df["avg_sys"].mean()
        avg_dia_val = dia if "dia" in locals() and dia > 0 else df["avg_dia"].mean()

        # ---- IMPORTANT ----
        # lifestyle_score comes from dataset definition
        lifestyle_score = (
            (1 if ever_smoke == 1 else 0) +
            (1 if ever_drink == 1 else 0) +
            (1 if binge_drink > 0 else 0)
        )

        feat = {
            "age": age,
            "bmi": bmi,
            "waist": waist,
            "height": height_cm,
            "weight": weight_kg,
            "avg_sys": avg_sys_val,
            "avg_dia": avg_dia_val,
            "gender": gender,
            "race": df["race"].mode()[0],
            "education": df["education"].mode()[0],
            "ever_smoked": ever_smoke,
            "current_smoker": current_smoker,
            "ever_drink": ever_drink,
            "binge_drink": binge_drink,
            "lifestyle_score": lifestyle_score
        }

        X_user = pd.DataFrame([feat])

        # ----------------------------
        # Depression special case
        # ----------------------------
        if "Depression" in selected[0]:
            phq_score = sum(dpq.values())
            st.metric("PHQ-9 Score", phq_score)

            # Category text
            levels = ["None", "Mild", "Moderate", "Moderately Severe", "Severe"]
            boundaries = [4, 9, 14, 19]
            category = levels[sum(phq_score > b for b in boundaries)]
            st.write("Category:", category)

            st.stop()

        # ----------------------------
        # Predict using loaded model
        # ----------------------------
        if model is None:
            st.error("Model not found!")
            st.stop()

        prediction = model.predict(X_user)

        # For classifiers → probability
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_user)
            risk = prob[0].max() * 100
            st.metric("Predicted Risk (%)", f"{risk:.1f}%")
            st.plotly_chart(gauge_chart(risk, selected[0], 0, 100))
        else:
            # Regression
            value = prediction[0]
            st.metric("Predicted Value", f"{value:.2f}")
            st.plotly_chart(gauge_chart(value, selected[0], 0, 200))

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
