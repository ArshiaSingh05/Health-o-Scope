"""
Train models for:
1) High Blood Pressure (classification)
2) High Cholesterol (classification)
3) Heart Disease (classification)
4) Depression PHQ-9 score (regression)
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
# Smaller RandomForest Settings
# -------------------------
RF_CLF = RandomForestClassifier(
    n_estimators=80,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

RF_REG = RandomForestRegressor(
    n_estimators=60,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)

# -------------------------
# Training Functions
# -------------------------

def train_classification_model(name, features, target):
    print(f"\nTraining {name} with SMOTE")

    data = df.dropna(subset=[target])
    X = data[features].fillna(data[features].median(numeric_only=True))
    y = data[target]

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
    print(classification_report(y_test, preds))

    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
    print(f"Saved → {name}.joblib")


def train_regression_model(name, features, target):
    print(f"\nTraining {name} (regression)")

    data = df.dropna(subset=[target])
    X = data[features].fillna(data[features].median(numeric_only=True))  # FIXED
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    print(f"{name} MAE: {mae:.2f}")

    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
    print(f"Saved → {name}.joblib")



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

train_classification_model("model_high_bp", basic_features, "diagnosed_high_bp")
train_classification_model("model_high_chol", basic_features, "diagnosed_high_chol")
train_classification_model("model_heart_disease", basic_features, "diagnosed_heart_disease")

train_regression_model("model_phq9_score", basic_features, "phq9_score")
train_regression_model("model_bmi", basic_features, "bmi")
train_regression_model("model_ldl", basic_features, "ldl")
train_regression_model("model_hdl", basic_features, "hdl")
train_regression_model("model_totalchol", basic_features, "total_cholesterol")

print("\nAll models trained successfully with smaller size!")
