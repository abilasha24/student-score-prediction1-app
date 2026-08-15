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
FEATURE_PATH = ROOT_DIR / "feature_cols.pkl"


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
# CHECK MODEL FILES
# =====================================================

if not MODEL_PATH.exists():

    st.error("❌ gbr_model.pkl not found.")

    st.info(
        "Please train the model from the Model Training page first."
    )

    st.stop()


if not FEATURE_PATH.exists():

    st.error("❌ feature_cols.pkl not found.")

    st.info(
        "Please train the model from the Model Training page first."
    )

    st.stop()


# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():

    return joblib.load(
        MODEL_PATH
    )


@st.cache_resource
def load_features():

    return joblib.load(
        FEATURE_PATH
    )


try:

    model = load_model()

    feature_columns = load_features()

except Exception as e:

    st.error(
        "❌ Failed to load trained model."
    )

    st.code(
        str(e)
    )

    st.stop()


st.success(
    "✅ Trained Gradient Boosting model loaded successfully."
)


# =====================================================
# INPUT FORM
# =====================================================

st.subheader("📝 Student Information")


col1, col2 = st.columns(2)


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
# CREATE INPUT
# =====================================================

input_df = pd.DataFrame({

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

})


# =====================================================
# PREPROCESS
# =====================================================

X_input = pd.get_dummies(
    input_df,
    drop_first=False
)


X_input = X_input.reindex(
    columns=feature_columns,
    fill_value=0
)


# =====================================================
# PREDICTION
# =====================================================

st.markdown("---")

predict_button = st.button(
    "🚀 Predict Exam Score",
    use_container_width=True
)


if predict_button:

    try:

        prediction = model.predict(
            X_input
        )


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
        # CATEGORY
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


        if score >= 50:

            status = "Pass"

        else:

            status = "Below Pass Mark"


        c3.metric(
            "Status",
            status
        )


        st.progress(
            int(score)
        )


        st.success(
            f"🎓 Predicted exam score: **{score:.2f} / 100**"
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
        # DOWNLOAD
        # =================================================

        result_df = input_df.copy()

        result_df[
            "predicted_exam_score"
        ] = round(
            score,
            2
        )


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