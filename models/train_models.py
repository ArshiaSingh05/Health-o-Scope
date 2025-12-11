"""
Train models for:
1) High Blood Pressure (classification)
2) High Cholesterol (classification)
3) Heart Disease (classification)
4) Depression category (classification from PHQ-9)
5) BMI prediction (regression)
6) LDL / HDL / Total cholesterol (regression)

Uses SMOTE for classification imbalance.
Models saved in models/ folder.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.linear_model import Ridge

DATA_PATH = r"dataset/Cleaned_finalDataset.csv"
MODELS_DIR = r"models"
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Loaded cleaned data:", df.shape)

# -------------------------
# Smaller RandomForest Settings (keeps models small)
# -------------------------
RF_CLF = RandomForestClassifier(
    n_estimators=80,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

RF_REG = RandomForestRegressor(
    n_estimators=60,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# -------------------------
# Prepare PHQ-9 categories (classification target)
# -------------------------
def phq_category(score):
    # expects numeric PHQ-9 score 0..27
    if score <= 4:
        return 0  # None
    elif score <= 9:
        return 1  # Mild
    elif score <= 14:
        return 2  # Moderate
    elif score <= 19:
        return 3  # Moderately Severe
    else:
        return 4  # Severe

# Only create category column if phq9_score exists
if "phq9_score" in df.columns:
    df["phq9_category"] = df["phq9_score"].fillna(0).apply(phq_category)
else:
    raise RuntimeError("phq9_score column not found in dataset.")

# -------------------------
# Training Functions
# -------------------------
def train_classification_model(name, features, target):
    print(f"\nTraining {name} with SMOTE")

    data = df.dropna(subset=[target])
    if data.shape[0] == 0:
        print(f"  -> No rows with target {target}, skipping.")
        return

    # Fill numeric NA with medians; categorical numeric encoding assumed already
    X = data[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = data[target].copy()

    # SMOTE requires no NaNs
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"After SMOTE: {X_res.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RF_CLF)
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    out_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, out_path)
    print(f"Saved → {out_path}")


def train_regression_model(name, features, target):
    print(f"\nTraining {name} (regression)")

    data = df.dropna(subset=[target])
    if data.shape[0] == 0:
        print(f"  -> No rows with target {target}, skipping.")
        return

    X = data[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = data[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use Ridge to keep model small for continuous targets (you can also use smaller RF)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    print(f"{name} MAE: {mae:.2f}")

    out_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, out_path)
    print(f"Saved → {out_path}")

# -------------------------
# Feature Lists
# -------------------------
basic_features = [
    "age", "bmi", "waist", "height", "weight",
    "avg_sys", "avg_dia",
    "gender", "race", "education",
    "ever_smoked", "current_smoker",
    "ever_drink", "binge_drink",
    "lifestyle_score"
]

chol_features = basic_features + ["hdl", "ldl", "total_cholesterol"]

# -------------------------------------------------
# Lifestyle Category Model (classification)
# -------------------------------------------------

# Create lifestyle numeric score
df["lifestyle_numeric"] = (
    (df["ever_smoked"] == 1).astype(int) +
    (df["current_smoker"] == 1).astype(int) +
    (df["ever_drink"] == 1).astype(int) +
    (df["binge_drink"] > 0).astype(int)
)

# Map numeric score → category
def lifestyle_category(score):
    if score == 0:
        return 0  # Excellent
    elif score == 1:
        return 1  # Good
    elif score == 2:
        return 2  # Moderate
    elif score == 3:
        return 3  # Poor
    else:
        return 4  # Very Poor

df["lifestyle_category"] = df["lifestyle_numeric"].apply(lifestyle_category)

# Train lifestyle classification model
train_classification_model(
    "model_lifestyle_score",
    basic_features,
    "lifestyle_category"
)

# -------------------------
# Train classification models (with SMOTE)
# -------------------------
train_classification_model("model_high_bp", basic_features, "diagnosed_high_bp")
train_classification_model("model_high_chol", basic_features, "diagnosed_high_chol")
train_classification_model("model_heart_disease", basic_features, "diagnosed_heart_disease")

# Train depression **classification** (PHQ-9 category)
train_classification_model("model_phq9_category", basic_features, "phq9_category")

# -------------------------
# Train regression models
# -------------------------
train_regression_model("model_bmi", basic_features, "bmi")
train_regression_model("model_ldl", chol_features, "ldl")
train_regression_model("model_hdl", chol_features, "hdl")
train_regression_model("model_totalchol", chol_features, "total_cholesterol")

print("\nAll requested models trained and saved.")
