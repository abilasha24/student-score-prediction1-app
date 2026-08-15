from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# =====================================================
# PATH SETUP
# =====================================================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

TRAIN_PATH = ROOT_DIR / "train.csv"
MODEL_PATH = ROOT_DIR / "gbr_model.pkl"
FEATURE_PATH = ROOT_DIR / "feature_cols.pkl"


# =====================================================
# PAGE
# =====================================================

st.title("🤖 Model Training")

st.markdown(
    "Train a Gradient Boosting model to predict student exam scores."
)

st.markdown("---")


# =====================================================
# DATA SOURCE
# =====================================================

st.subheader("📁 Training Dataset")

source = st.radio(
    "Choose dataset source:",
    ["Upload CSV", "Use local train.csv"],
    horizontal=True
)

df = None


# =====================================================
# UPLOAD CSV
# =====================================================

if source == "Upload CSV":

    uploaded_file = st.file_uploader(
        "Upload training CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file,
                low_memory=False
            )

            st.success(
                f"✅ Dataset loaded successfully! "
                f"Rows: {len(df):,} | "
                f"Columns: {len(df.columns)}"
            )

        except Exception as e:

            st.error("❌ Failed to read uploaded CSV.")
            st.code(str(e))

            st.stop()

    else:

        st.info(
            "⬆️ Upload a CSV file to begin training."
        )

        st.stop()


# =====================================================
# LOCAL DATASET
# =====================================================

else:

    if TRAIN_PATH.exists():

        try:

            df = pd.read_csv(
                TRAIN_PATH,
                low_memory=False
            )

            st.success(
                f"✅ train.csv loaded successfully! "
                f"Rows: {len(df):,} | "
                f"Columns: {len(df.columns)}"
            )

        except Exception as e:

            st.error(
                "❌ Failed to read train.csv."
            )

            st.code(str(e))

            st.stop()

    else:

        st.warning(
            "⚠️ train.csv is not available."
        )

        st.info(
            "Please select **Upload CSV** "
            "and upload your training dataset."
        )

        st.stop()


# =====================================================
# TARGET COLUMN
# =====================================================

st.subheader("🎯 Target Column")

possible_targets = [
    "exam_score",
    "score",
    "marks",
    "final_score",
    "target"
]

available_targets = [
    col
    for col in possible_targets
    if col in df.columns
]

if not available_targets:

    st.error(
        "❌ Target column not found. "
        "Expected one of: "
        "exam_score, score, marks, final_score, target"
    )

    st.write("Available columns:")

    st.write(
        list(df.columns)
    )

    st.stop()


target_col = st.selectbox(
    "Select target column:",
    available_targets
)


# =====================================================
# BASIC CLEANING
# =====================================================

df = df.copy()


# Remove completely empty columns
df = df.dropna(
    axis=1,
    how="all"
)


# Remove rows where target is missing
df = df.dropna(
    subset=[target_col]
)


# Convert target to numeric
y = pd.to_numeric(
    df[target_col],
    errors="coerce"
)


# Keep only valid target rows
valid_rows = y.notna()

X = df.drop(
    columns=[target_col]
)

X = X.loc[
    valid_rows
].copy()

y = y.loc[
    valid_rows
].copy()


# =====================================================
# REMOVE ID COLUMN
# =====================================================

if "id" in X.columns:

    X = X.drop(
        columns=["id"]
    )


# =====================================================
# CLEAN COLUMN TYPES
# =====================================================

# Convert boolean columns to strings
for col in X.select_dtypes(
    include=["bool"]
).columns:

    X[col] = X[col].astype(str)


# Detect categorical columns
categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# Detect numeric columns
numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()


# =====================================================
# FORCE NUMERIC COLUMNS TO NUMERIC
# =====================================================

for col in numeric_cols:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


# =====================================================
# FORCE CATEGORICAL VALUES TO STRINGS
# =====================================================

for col in categorical_cols:

    X[col] = (
        X[col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


# =====================================================
# FEATURE INFORMATION
# =====================================================

st.subheader("🔎 Feature Information")

c1, c2, c3 = st.columns(3)

c1.metric(
    "Training Rows",
    f"{len(X):,}"
)

c2.metric(
    "Numeric Features",
    len(numeric_cols)
)

c3.metric(
    "Categorical Features",
    len(categorical_cols)
)


# =====================================================
# PREPROCESSING PIPELINES
# =====================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# =====================================================
# COLUMN TRANSFORMER
# =====================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_cols
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_cols
        )
    ],
    remainder="drop"
)


# =====================================================
# TRAIN BUTTON
# =====================================================

st.markdown("---")

if st.button(
    "🚀 Train Gradient Boosting Model",
    use_container_width=True
):

    try:

        with st.spinner(
            "Training model... This may take some time."
        ):

            # =========================================
            # TRAIN / TEST SPLIT
            # =========================================

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42
                )
            )


            # =========================================
            # FIT PREPROCESSOR
            # =========================================

            X_train_processed = (
                preprocessor.fit_transform(
                    X_train
                )
            )


            # =========================================
            # TRANSFORM TEST DATA
            # =========================================

            X_test_processed = (
                preprocessor.transform(
                    X_test
                )
            )


            # =========================================
            # FEATURE NAMES
            # =========================================

            feature_names = []

            # Numeric feature names
            feature_names.extend(
                numeric_cols
            )


            # Categorical feature names
            if len(categorical_cols) > 0:

                categorical_encoder = (
                    preprocessor
                    .named_transformers_[
                        "categorical"
                    ]
                    .named_steps[
                        "encoder"
                    ]
                )

                categorical_feature_names = (
                    categorical_encoder
                    .get_feature_names_out(
                        categorical_cols
                    )
                )

                feature_names.extend(
                    categorical_feature_names.tolist()
                )


            # =========================================
            # GRADIENT BOOSTING MODEL
            # =========================================

            model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=42
            )


            # =========================================
            # TRAIN MODEL
            # =========================================

            model.fit(
                X_train_processed,
                y_train
            )


            # =========================================
            # TEST PREDICTIONS
            # =========================================

            predictions = model.predict(
                X_test_processed
            )


            # =========================================
            # METRICS
            # =========================================

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
                        y_test.values
                        - predictions
                    ) <= 10
                ) * 100
            )


            # =========================================
            # MODEL BUNDLE
            # =========================================

            model_bundle = {
                "model": model,
                "preprocessor": preprocessor,
                "target_column": target_col
            }


            # =========================================
            # SAVE MODEL
            # =========================================

            joblib.dump(
                model_bundle,
                MODEL_PATH
            )


            # =========================================
            # SAVE FEATURE NAMES
            # =========================================

            joblib.dump(
                feature_names,
                FEATURE_PATH
            )


        # =================================================
        # RESULTS
        # =================================================

        st.success(
            "✅ Model trained successfully!"
        )


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


        st.info(
            f"💾 Model saved: "
            f"{MODEL_PATH.name}"
        )


        st.info(
            f"💾 Features saved: "
            f"{FEATURE_PATH.name}"
        )


        # =================================================
        # MODEL VALIDATION
        # =================================================

        st.markdown("---")

        st.subheader(
            "🔍 Model Validation"
        )

        st.write(
            "Model type:",
            type(model).__name__
        )

        st.write(
            "Preprocessor:",
            type(preprocessor).__name__
        )

        st.write(
            "Processed training features:",
            X_train_processed.shape[1]
        )

        st.write(
            "Saved model bundle:",
            "✅ Ready"
        )


    except Exception as e:

        st.error(
            "❌ Model training failed."
        )

        st.exception(e)