from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import joblib


# =====================================================
# PATHS
# =====================================================

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

MODEL_PATH = ROOT_DIR / "gbr_model.pkl"


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Student Score Prediction",
    page_icon="🎯",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("🎯 Student Exam Score Prediction")

st.write(
    "Enter student information to estimate the expected exam score."
)

st.markdown("---")


# =====================================================
# CHECK MODEL
# =====================================================

if not MODEL_PATH.exists():

    st.error("❌ gbr_model.pkl not found.")

    st.info(
        "Please train the model from the Model Training page first."
    )

    st.stop()


# =====================================================
# LOAD MODEL BUNDLE
# =====================================================

@st.cache_resource
def load_model_bundle():

    return joblib.load(MODEL_PATH)


try:

    model_bundle = load_model_bundle()

    # -------------------------------------------------
    # Validate model bundle
    # -------------------------------------------------

    if not isinstance(model_bundle, dict):

        st.error(
            "❌ Invalid model file. Expected a model bundle."
        )

        st.stop()


    required_keys = [
        "model",
        "preprocessor",
        "target_column"
    ]


    missing_keys = [
        key
        for key in required_keys
        if key not in model_bundle
    ]


    if missing_keys:

        st.error(
            "❌ Model bundle is incomplete."
        )

        st.write(
            "Missing:",
            missing_keys
        )

        st.stop()


    model = model_bundle["model"]

    preprocessor = model_bundle["preprocessor"]

    target_column = model_bundle["target_column"]


except Exception as e:

    st.error(
        "❌ Failed to load trained model."
    )

    st.code(
        str(e)
    )

    st.stop()


# =====================================================
# MODEL VALIDATION
# =====================================================

if not hasattr(model, "predict"):

    st.error(
        "❌ Loaded object is not a valid prediction model."
    )

    st.stop()


if not hasattr(preprocessor, "transform"):

    st.error(
        "❌ Saved preprocessor is invalid."
    )

    st.stop()


st.success(
    "✅ Trained Gradient Boosting model loaded successfully."
)


# =====================================================
# MODEL INFORMATION
# =====================================================

with st.expander("🔍 Model Information"):

    st.write(
        f"**Model:** `{type(model).__name__}`"
    )

    st.write(
        f"**Preprocessor:** `{type(preprocessor).__name__}`"
    )

    st.write(
        f"**Target Column:** `{target_column}`"
    )


# =====================================================
# INPUT FORM
# =====================================================

st.subheader("📝 Student Information")


col1, col2 = st.columns(2)


# =====================================================
# LEFT COLUMN
# =====================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=10,
        max_value=60,
        value=20,
        step=1
    )


    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )


    course = st.selectbox(
        "Course",
        [
            "Science",
            "Arts",
            "Commerce",
            "Engineering",
            "Medicine",
            "Computer Science",
            "Other"
        ]
    )


    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=24.0,
        value=4.0,
        step=0.5
    )


    class_attendance = st.number_input(
        "Class Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )


    internet_access = st.selectbox(
        "Internet Access",
        [
            "Yes",
            "No"
        ]
    )


# =====================================================
# RIGHT COLUMN
# =====================================================

with col2:

    sleep_hours = st.number_input(
        "Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )


    sleep_quality = st.selectbox(
        "Sleep Quality",
        [
            "Poor",
            "Average",
            "Good"
        ]
    )


    study_method = st.selectbox(
        "Study Method",
        [
            "Self Study",
            "Group Study",
            "Online Resources",
            "Tutoring",
            "Other"
        ]
    )


    facility_rating = st.slider(
        "Facility Rating",
        min_value=1,
        max_value=5,
        value=3
    )


    exam_difficulty = st.selectbox(
        "Exam Difficulty",
        [
            "Easy",
            "Medium",
            "Hard"
        ]
    )


# =====================================================
# CREATE INPUT DATAFRAME
# =====================================================

input_df = pd.DataFrame(
    {
        "age": [age],
        "gender": [gender],
        "course": [course],
        "study_hours": [study_hours],
        "class_attendance": [class_attendance],
        "internet_access": [internet_access],
        "sleep_hours": [sleep_hours],
        "sleep_quality": [sleep_quality],
        "study_method": [study_method],
        "facility_rating": [facility_rating],
        "exam_difficulty": [exam_difficulty]
    }
)


# =====================================================
# FORCE CORRECT DATA TYPES
# =====================================================

numeric_columns = [
    "age",
    "study_hours",
    "class_attendance",
    "sleep_hours",
    "facility_rating"
]


categorical_columns = [
    "gender",
    "course",
    "internet_access",
    "sleep_quality",
    "study_method",
    "exam_difficulty"
]


for column in numeric_columns:

    input_df[column] = pd.to_numeric(
        input_df[column],
        errors="coerce"
    )


for column in categorical_columns:

    input_df[column] = input_df[column].astype(str)


# =====================================================
# VALIDATE INPUT
# =====================================================

if input_df[numeric_columns].isna().any().any():

    st.error(
        "❌ Invalid numeric input."
    )

    st.stop()


# =====================================================
# PREDICTION BUTTON
# =====================================================

st.markdown("---")


predict_button = st.button(
    "🚀 Predict Exam Score",
    use_container_width=True
)


# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    try:

        # -------------------------------------------------
        # Ensure columns match training data
        # -------------------------------------------------

        transformer_columns = []

        for transformer_name, transformer, columns in (
            preprocessor.transformers
        ):

            if transformer_name != "remainder":

                if isinstance(columns, (list, tuple)):

                    transformer_columns.extend(
                        columns
                    )

                else:

                    try:

                        transformer_columns.extend(
                            list(columns)
                        )

                    except TypeError:

                        pass


        # -------------------------------------------------
        # Check required columns
        # -------------------------------------------------

        missing_columns = [
            column
            for column in transformer_columns
            if column not in input_df.columns
        ]


        if missing_columns:

            st.error(
                "❌ Input data is missing required columns."
            )

            st.write(
                missing_columns
            )

            st.stop()


        # -------------------------------------------------
        # Reorder columns
        # -------------------------------------------------

        if transformer_columns:

            input_for_model = input_df[
                transformer_columns
            ].copy()

        else:

            input_for_model = input_df.copy()


        # -------------------------------------------------
        # Final dtype protection
        # -------------------------------------------------

        for column in numeric_columns:

            if column in input_for_model.columns:

                input_for_model[column] = pd.to_numeric(
                    input_for_model[column],
                    errors="coerce"
                )


        for column in categorical_columns:

            if column in input_for_model.columns:

                input_for_model[column] = (
                    input_for_model[column]
                    .astype(str)
                )


        # -------------------------------------------------
        # Transform
        # -------------------------------------------------

        X_input = preprocessor.transform(
            input_for_model
        )


        # -------------------------------------------------
        # Model prediction
        # -------------------------------------------------

        prediction = model.predict(
            X_input
        )


        # -------------------------------------------------
        # Score
        # -------------------------------------------------

        score = float(
            prediction[0]
        )


        score = float(
            np.clip(
                score,
                0,
                100
            )
        )


        # =================================================
        # PERFORMANCE CATEGORY
        # =================================================

        if score >= 75:

            category = "Excellent"

        elif score >= 60:

            category = "Good"

        elif score >= 40:

            category = "Average"

        else:

            category = "Needs Improvement"


        # =================================================
        # PASS STATUS
        # =================================================

        if score >= 50:

            status = "Pass"

        else:

            status = "Below Pass Mark"


        # =================================================
        # RESULT
        # =================================================

        st.markdown("---")

        st.subheader(
            "🎯 Prediction Result"
        )


        c1, c2, c3 = st.columns(3)


        c1.metric(
            "Predicted Exam Score",
            f"{score:.2f} / 100"
        )


        c2.metric(
            "Performance",
            category
        )


        c3.metric(
            "Status",
            status
        )


        # -------------------------------------------------
        # Score progress
        # -------------------------------------------------

        st.progress(
            int(score)
        )


        # -------------------------------------------------
        # Result message
        # -------------------------------------------------

        if score >= 75:

            st.success(
                f"🎓 Excellent! Predicted exam score: "
                f"**{score:.2f} / 100**"
            )

        elif score >= 60:

            st.success(
                f"🎓 Good performance! Predicted exam score: "
                f"**{score:.2f} / 100**"
            )

        elif score >= 40:

            st.warning(
                f"📚 Average performance. Predicted exam score: "
                f"**{score:.2f} / 100**"
            )

        else:

            st.error(
                f"⚠️ Needs improvement. Predicted exam score: "
                f"**{score:.2f} / 100**"
            )


        # =================================================
        # INPUT SUMMARY
        # =================================================

        st.subheader(
            "📋 Student Information"
        )


        st.dataframe(
            input_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # DOWNLOAD RESULT
        # =================================================

        result_df = input_df.copy()


        result_df[
            "predicted_exam_score"
        ] = round(
            score,
            2
        )


        result_df[
            "performance"
        ] = category


        result_df[
            "status"
        ] = status


        st.download_button(

            label="⬇️ Download Prediction",

            data=result_df.to_csv(
                index=False
            ).encode("utf-8"),

            file_name="student_score_prediction.csv",

            mime="text/csv",

            use_container_width=True
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.code(
            str(e)
        )