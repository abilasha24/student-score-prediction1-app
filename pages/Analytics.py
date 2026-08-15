import os
from pathlib import Path

import streamlit as st
import pandas as pd
import altair as alt

# =====================================================
# PATH SETUP
# =====================================================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

HISTORY_FILE = ROOT_DIR / "prediction_history.csv"

# =====================================================
# PAGE TITLE
# =====================================================
st.title("📊 Analytics Dashboard")
st.markdown("View prediction insights, score distribution, and model usage analytics.")
st.markdown("---")

# =====================================================
# LOAD PREDICTION HISTORY
# =====================================================
if HISTORY_FILE.exists():
    try:
        history_df = pd.read_csv(HISTORY_FILE)
        st.success(f"✅ Prediction history loaded successfully! Shape: {history_df.shape}")
    except Exception as e:
        st.error("❌ Failed to load prediction history.")
        st.code(str(e))
        st.stop()
else:
    st.warning("⚠️ prediction_history.csv not found.")
    st.info("Please go to the Prediction page and generate predictions first.")
    st.stop()

# =====================================================
# PREVIEW
# =====================================================
st.subheader("1️⃣ Prediction History Preview")
st.dataframe(history_df.head(20), use_container_width=True)

# =====================================================
# REQUIRED COLUMN CHECK
# =====================================================
required_cols = ["predicted_exam_score", "model_used", "timestamp"]

missing_cols = [col for col in required_cols if col not in history_df.columns]

if missing_cols:
    st.error(f"❌ Missing required columns in prediction_history.csv: {missing_cols}")
    st.stop()

# =====================================================
# CLEAN DATA
# =====================================================
history_df["predicted_exam_score"] = pd.to_numeric(
    history_df["predicted_exam_score"], errors="coerce"
)

history_df["timestamp"] = pd.to_datetime(
    history_df["timestamp"], errors="coerce"
)

history_df = history_df.dropna(subset=["predicted_exam_score"])

# =====================================================
# SUMMARY METRICS
# =====================================================
st.subheader("2️⃣ Analytics Summary")

total_predictions = len(history_df)
avg_score = history_df["predicted_exam_score"].mean()
max_score = history_df["predicted_exam_score"].max()
min_score = history_df["predicted_exam_score"].min()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Predictions", f"{total_predictions:,}")
c2.metric("Average Score", f"{avg_score:.2f}")
c3.metric("Highest Score", f"{max_score:.2f}")
c4.metric("Lowest Score", f"{min_score:.2f}")

# =====================================================
# SCORE DISTRIBUTION
# =====================================================
st.subheader("3️⃣ Predicted Score Distribution")

score_chart = alt.Chart(history_df).mark_bar(opacity=0.8).encode(
    x=alt.X("predicted_exam_score:Q", bin=alt.Bin(maxbins=20), title="Predicted Exam Score"),
    y=alt.Y("count()", title="Count"),
    tooltip=[alt.Tooltip("count()", title="Count")]
).properties(
    height=350,
    title="Distribution of Predicted Exam Scores"
)

st.altair_chart(score_chart, use_container_width=True)

# =====================================================
# MODEL USAGE
# =====================================================
st.subheader("4️⃣ Model Usage Analysis")

model_counts = history_df["model_used"].value_counts().reset_index()
model_counts.columns = ["Model", "Count"]

model_chart = alt.Chart(model_counts).mark_bar().encode(
    x=alt.X("Model:N", title="Model"),
    y=alt.Y("Count:Q", title="Number of Predictions"),
    tooltip=["Model", "Count"]
).properties(
    height=350,
    title="Prediction Count by Model"
)

st.altair_chart(model_chart, use_container_width=True)

# =====================================================
# SCORE CATEGORY ANALYSIS
# =====================================================
st.subheader("5️⃣ Score Category Analysis")

def categorize_score(score):
    if score >= 75:
        return "Excellent"
    elif score >= 60:
        return "Good"
    elif score >= 40:
        return "Average"
    else:
        return "Needs Improvement"

history_df["score_category"] = history_df["predicted_exam_score"].apply(categorize_score)

category_counts = history_df["score_category"].value_counts().reset_index()
category_counts.columns = ["Category", "Count"]

category_chart = alt.Chart(category_counts).mark_bar().encode(
    x=alt.X("Category:N", title="Performance Category"),
    y=alt.Y("Count:Q", title="Count"),
    tooltip=["Category", "Count"]
).properties(
    height=350,
    title="Predicted Performance Categories"
)

st.altair_chart(category_chart, use_container_width=True)

# =====================================================
# TIMESTAMP ANALYSIS
# =====================================================
st.subheader("6️⃣ Prediction Activity Over Time")

time_df = history_df.dropna(subset=["timestamp"]).copy()

if not time_df.empty:
    time_df["date"] = time_df["timestamp"].dt.date
    time_counts = time_df.groupby("date").size().reset_index(name="Count")

    time_chart = alt.Chart(time_counts).mark_line(point=True).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("Count:Q", title="Predictions"),
        tooltip=["date:T", "Count:Q"]
    ).properties(
        height=350,
        title="Prediction Activity Over Time"
    )

    st.altair_chart(time_chart, use_container_width=True)
else:
    st.info("No valid timestamp data available for time analysis.")

# =====================================================
# STATIC MODEL PERFORMANCE SECTION
# =====================================================
st.subheader("7️⃣ Model Performance Overview")

performance_df = pd.DataFrame({
    "Model": ["Gradient Boosting", "Random Forest", "Deep Neural Network"],
    "R² Score": [0.7705, 0.7441, 0.7640],
    "MAE": [7.1580, 7.5744, 7.2619],
    "RMSE": [8.9609, 9.4612, 9.0865],
    "Accuracy (±10)": [73.47, 70.80, 72.38]
})

st.dataframe(
    performance_df.style.format({
        "R² Score": "{:.4f}",
        "MAE": "{:.4f}",
        "RMSE": "{:.4f}",
        "Accuracy (±10)": "{:.2f}%"
    }),
    use_container_width=True
)

perf_chart = alt.Chart(performance_df).mark_bar().encode(
    x=alt.X("Model:N", title="Model"),
    y=alt.Y("Accuracy (±10):Q", title="Accuracy (±10 marks)"),
    tooltip=["Model", "Accuracy (±10)"]
).properties(
    height=350,
    title="Model Accuracy Comparison"
)

st.altair_chart(perf_chart, use_container_width=True)

best_model = performance_df.sort_values(by="Accuracy (±10)", ascending=False).iloc[0]["Model"]
st.success(f"🏆 Best model based on Accuracy (±10 marks): {best_model}")

# =====================================================
# DOWNLOAD HISTORY
# =====================================================
st.subheader("8️⃣ Download Analytics Data")

st.download_button(
    label="⬇️ Download Prediction History CSV",
    data=history_df.to_csv(index=False).encode("utf-8"),
    file_name="prediction_history_export.csv",
    mime="text/csv",
    use_container_width=True
)

# =====================================================
# FINAL NOTE
# =====================================================
st.info("""
This analytics dashboard provides:
- prediction history preview
- predicted score distribution
- model usage analysis
- score category analysis
- prediction activity tracking
- model performance comparison
""")