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
    return joblib.load(path) if os.path.exists(path) else None

df = load_data()

st.set_page_config(layout="wide", page_title="Health Risk Assessment")

# ---- Navigation handled only by session_state ----
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar controls navigation
nav = st.sidebar.radio("Navigation", ["Home", "Predict", "EDA"], index=["Home", "Predict", "EDA"].index(st.session_state.page))

# Update session_state.page only when user explicitly chooses a different page
if nav != st.session_state.page:
    st.session_state.page = nav
    st.rerun()

page = st.session_state.page

# -----------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------
def gauge_chart(value, title, vmin=0, vmax=100, steps=None):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        gauge={'axis': {'range': [vmin, vmax]}, 'steps': steps},
    ))
    fig.update_layout(height=300, title=title)
    return fig

def percentile_overlay(series, user_value, title):
    fig = px.histogram(series.dropna(), nbins=40)
    fig.add_vline(x=user_value, line_color="red", line_width=3)
    fig.update_layout(title=title, height=350)
    return fig

# -----------------------------------------------------------------
# CARD DEFINITIONS
# -----------------------------------------------------------------
cards = [
    ("Blood Pressure Check", "Checks your BP-related health risk.", "model_high_bp"),
    ("Cholesterol Check", "Predicts overall cholesterol risk.", "model_high_chol"),
    ("Heart Health Check", "Estimates heart disease likelihood.", "model_heart_disease"),
    ("Mood & Mental Well-Being", "Assesses mood using PHQ-9 questionnaire.", "model_phq9_score"),
    ("Daily Habits Health Score", "Rates smoking & drinking lifestyle habits.", "model_lifestyle_score"),
    ("Body Weight Indicator (BMI)", "Estimates BMI from body measures.", "model_bmi"),
    ("Detailed Cholesterol Report", "Predicts LDL, HDL & Total Cholesterol.", "model_ldl"),
]

# -----------------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------------
if page == "Home":
    st.title("Health Risk Assessment System")
    st.write("Choose a prediction card. Clicking a card opens only the inputs needed for that prediction.")

    # Card styling
    st.markdown("""
        <style>
        .card {
            padding: 20px;
            border-radius: 14px;
            background: white;
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            width: 260px;
            height: 160px;
            text-align: center;
            transition: 0.2s;
        }
        .card:hover { transform: translateY(-4px); }
        </style>
    """, unsafe_allow_html=True)

    rows = [cards[i:i+3] for i in range(0, len(cards), 3)]

    for row in rows:
        st.write("")  # spacing between rows
        cols = st.columns(3, gap="large")

        # Center short rows (only last row)
        offset = (3 - len(row)) // 2  

        for idx, (title, desc, model_key) in enumerate(row):
            with cols[idx + offset]:

                # MAIN BUTTON — handles navigation
                if st.button(
                    f"{title}\n\n{desc}",
                    key=f"card_{model_key}",
                    use_container_width=True
                ):
                    st.session_state["selected_card"] = (title, model_key)
                    st.session_state.page = "Predict"  # switch page
                    st.rerun()  # reload UI to Predict page

        st.write("")

# -----------------------------------------------------------------
# PREDICT PAGE
# -----------------------------------------------------------------
elif page == "Predict":
    st.session_state["go_predict"] = False  # reset trigger

    selected = st.session_state.get("selected_card")
    if not selected:
        st.info("Please select a prediction card from the Home page.")
        st.stop()

    st.header(f"Prediction: {selected[0]}")
    model = load_model(selected[1])

    col1, col2, col3 = st.columns(3)

    # Basic inputs
    age = col1.number_input("Age", min_value=0, max_value=120, value=30)
    gender = col1.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

    height_cm = col2.number_input("Height (cm)", 50, 250, 170)
    weight_kg = col2.number_input("Weight (kg)", 10, 300, 70)
    bmi = round(weight_kg / ((height_cm / 100)**2), 1)
    col2.write(f"**BMI: {bmi}**")

    waist = col3.number_input("Waist (cm)", 30, 200, 80)

    ever_smoke_label = col1.selectbox("Ever smoked?", ["No", "Yes"])
    ever_smoke = 1 if ever_smoke_label == "Yes" else 2
    current_smoker_label = col1.selectbox("Currently smoking?", ["No", "Yes"])
    current_smoker = 1 if current_smoker_label == "Yes" else 2
    ever_drink_label = col2.selectbox("Ever drink alcohol?", ["No", "Yes"])
    ever_drink = 1 if ever_drink_label == "Yes" else 2
    # binge_drink = col3.number_input("How many times did you drink heavily at once?", 0, 50, 0)
    binge_map = {
    "Never": 0,
    "Once a month": 1,
    "2–3 times a month": 2,
    "Weekly": 4,
    "Multiple times weekly": 8
    }
    binge_choice = col3.selectbox(
        "Heavy drinking frequency",
        list(binge_map.keys()),
        index=0,
        help="How often you consume 4–5+ drinks in a single sitting?"
    )

    binge_drink = binge_map[binge_choice]

        # BP inputs
    if "Blood Pressure" in selected[0]:
        sys = st.number_input("Systolic BP", 0, 300, 0)
        dia = st.number_input("Diastolic BP", 0, 160, 0)
    else:
        sys = df["avg_sys"].mean()
        dia = df["avg_dia"].mean()

    # Depression questions
    if "Depression" in selected[0]:
        dpq = {}
        questions = [
            "Little interest or pleasure",
            "Feeling down",
            "Trouble sleeping",
            "Feeling tired",
            "Poor appetite",
            "Feeling bad about yourself",
            "Trouble concentrating",
            "Slow/fidgety movement",
            "Thoughts of self-harm"
        ]
        for i, q in enumerate(questions, 1):
            dpq[f"DPQ0{i}"] = st.slider(q, 0, 3, 0)

    if st.button("Predict"):

        # Depression → special case
        if "Depression" in selected[0]:
            score = sum(dpq.values())
            st.metric("PHQ-9 Score", score)
            st.stop()

        feat = {
            "age": age,
            "bmi": bmi,
            "waist": waist,
            "height": height_cm,
            "weight": weight_kg,
            "avg_sys": sys,
            "avg_dia": dia,
            "gender": gender,
            "race": df["race"].mode()[0],
            "education": df["education"].mode()[0],
            "ever_smoked": ever_smoke,
            "current_smoker": current_smoker,
            "ever_drink": ever_drink,
            "binge_drink": binge_drink,
            "lifestyle_score": (ever_smoke==1) + (ever_drink==1) + (binge_drink>0)
        }

        X = pd.DataFrame([feat])
        pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            risk = model.predict_proba(X)[0].max() * 100
            st.metric("Risk (%)", f"{risk:.1f}")
            st.plotly_chart(gauge_chart(risk, selected[0], 0, 100))
        else:
            value = pred[0]
            st.metric("Prediction", f"{value:.2f}")
            st.plotly_chart(gauge_chart(value, selected[0], 0, 200))

# -----------------------------------------------------------------
# EDA PAGE
# -----------------------------------------------------------------
elif page == "EDA":
    st.title("Dataset EDA")
    st.plotly_chart(px.histogram(df, x="bmi"))
    st.plotly_chart(px.histogram(df, x="phq9_score"))
