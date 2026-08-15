from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# =====================================================
# PATHS
# =====================================================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

DATASET_PATH = ROOT_DIR / "train.csv"
MODEL_PATH = ROOT_DIR / "gbr_model.pkl"
FEATURE_PATH = ROOT_DIR / "feature_cols.pkl"


# =====================================================
# PAGE
# =====================================================

st.title("🧠 Model Training")

st.write(
    "Train the Gradient Boosting model using the student dataset."
)

st.markdown("---")


# =====================================================
# LOAD DATASET
# =====================================================

if not DATASET_PATH.exists():

    st.error("❌ train.csv was not found.")
    st.stop()


df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)


st.success(
    f"✅ Dataset loaded successfully: "
    f"{df.shape[0]:,} rows × {df.shape[1]} columns"
)


# =====================================================
# NORMALIZE COLUMN NAMES
# =====================================================

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("-", "_", regex=False)
)


# =====================================================
# CHECK TARGET
# =====================================================

if "exam_score" not in df.columns:

    st.error(
        "❌ 'exam_score' column was not found in train.csv."
    )

    st.write(
        "Available columns:"
    )

    st.write(
        df.columns.tolist()
    )

    st.stop()


# =====================================================
# FEATURES
# =====================================================

features = [
    "age",
    "gender",
    "course",
    "study_hours",
    "class_attendance",
    "internet_access",
    "sleep_hours",
    "sleep_quality",
    "study_method",
    "facility_rating",
    "exam_difficulty"
]


available_features = [
    feature
    for feature in features
    if feature in df.columns
]


if len(available_features) == 0:

    st.error(
        "❌ None of the expected features were found."
    )

    st.write(
        df.columns.tolist()
    )

    st.stop()


st.subheader("📋 Features Used")

st.write(
    available_features
)

st.write(
    "Target: **exam_score**"
)


# =====================================================
# PREPARE DATA
# =====================================================

model_df = df[
    available_features + ["exam_score"]
].dropna()


X_raw = model_df[
    available_features
]

y = model_df[
    "exam_score"
]


# =====================================================
# ENCODE CATEGORICAL FEATURES
# =====================================================

X = pd.get_dummies(
    X_raw,
    drop_first=False
)

feature_columns = X.columns.tolist()


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# =====================================================
# TRAINING
# =====================================================

st.subheader("🚀 Model Training")

st.write(
    f"Training rows: {len(X_train):,}"
)

st.write(
    f"Testing rows: {len(X_test):,}"
)


train_button = st.button(
    "🚀 Start Gradient Boosting Training",
    use_container_width=True
)


if train_button:

    with st.spinner(
        "Training model... Please wait."
    ):

        model = GradientBoostingRegressor(
            random_state=42,
            n_estimators=100
        )

        model.fit(
            X_train,
            y_train
        )


        predictions = model.predict(
            X_test
        )


        # Metrics

        r2 = r2_score(
            y_test,
            predictions
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        accuracy_10 = (
            np.mean(
                np.abs(
                    y_test - predictions
                ) <= 10
            ) * 100
        )


        # =================================================
        # SAVE MODEL
        # =================================================

        joblib.dump(
            model,
            MODEL_PATH
        )

        joblib.dump(
            feature_columns,
            FEATURE_PATH
        )


    # =====================================================
    # RESULTS
    # =====================================================

    st.success(
        "✅ Model trained successfully!"
    )

    st.success(
        f"💾 Saved model: {MODEL_PATH.name}"
    )

    st.success(
        f"💾 Saved features: {FEATURE_PATH.name}"
    )


    # =====================================================
    # METRICS
    # =====================================================

    st.subheader(
        "📊 Model Performance"
    )

    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    c2.metric(
        "MAE",
        f"{mae:.2f}"
    )

    c3.metric(
        "RMSE",
        f"{rmse:.2f}"
    )

    c4.metric(
        "Accuracy ±10",
        f"{accuracy_10:.2f}%"
    )


    st.markdown("---")

    st.success(
        "🎯 Training completed. "
        "The model is ready for the Prediction page."
    )