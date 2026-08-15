import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# =====================================================
# PATH SETUP
# =====================================================
CURRENT_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(CURRENT_DIR)

DEFAULT_DATASET = os.path.join(ROOT_DIR, "train.csv")

# =====================================================
# CACHE FUNCTIONS
# =====================================================
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path, low_memory=False)

@st.cache_data
def sample_dataset(df, size=20000):
    if len(df) <= size:
        return df
    return df.sample(n=size, random_state=42)

# =====================================================
# PAGE TITLE
# =====================================================
st.title("📊 Dataset & Exploratory Data Analysis")
st.markdown("---")

# =====================================================
# LOAD DATA
# =====================================================
st.subheader("1️⃣ Dataset Selection")

use_default = st.checkbox("Use default training dataset (train.csv)", value=True)

if use_default:

    if os.path.exists(DEFAULT_DATASET):

        df = load_dataset(DEFAULT_DATASET)
        df_sample = sample_dataset(df)

        st.success(f"Default dataset loaded successfully! File: train.csv | Shape: {df.shape}")

    else:
        st.error("train.csv file not found.")
        st.stop()

else:
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is None:
        st.stop()

    df = pd.read_csv(uploaded)
    df_sample = sample_dataset(df)

# =====================================================
# DATASET OVERVIEW
# =====================================================
st.subheader("2️⃣ Dataset Overview")

rows = df.shape[0]
cols = df.shape[1]
numeric_cols = df.select_dtypes(include=np.number).columns
missing = df.isnull().sum().sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{rows:,}")
c2.metric("Columns", cols)
c3.metric("Numeric Columns", len(numeric_cols))
c4.metric("Missing Values", missing)

# =====================================================
# DATASET PREVIEW
# =====================================================
st.subheader("3️⃣ Dataset Preview")

preview_rows = st.slider("Select number of rows to preview", 5, 50, 5)

st.dataframe(df.head(preview_rows), use_container_width=True)

# =====================================================
# BASIC STATS
# =====================================================
st.subheader("4️⃣ Basic Statistics")

st.dataframe(df.describe(), use_container_width=True)

# =====================================================
# FEATURE DISTRIBUTION
# =====================================================
st.subheader("5️⃣ Feature Distributions (sampled data for speed)")

numeric_cols = df_sample.select_dtypes(include=np.number).columns

selected_cols = st.multiselect(
    "Select numeric columns",
    numeric_cols,
    default=numeric_cols[:3]
)

for col in selected_cols:

    chart = alt.Chart(df_sample).mark_bar(opacity=0.75).encode(
        x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=20)),
        y="count()"
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

# =====================================================
# TARGET ANALYSIS
# =====================================================
if "exam_score" in df.columns:

    st.subheader("6️⃣ Exam Score Distribution")

    score_chart = alt.Chart(df_sample).mark_bar(opacity=0.8).encode(
        x=alt.X("exam_score:Q", bin=alt.Bin(maxbins=20)),
        y="count()"
    ).properties(height=300)

    st.altair_chart(score_chart, use_container_width=True)

# =====================================================
# CORRELATION MATRIX
# =====================================================
st.subheader("7️⃣ Correlation Matrix")

numeric_df = df_sample.select_dtypes(include=["int64","float64","int32","float32"])

if numeric_df.shape[1] >= 2:

    corr = numeric_df.corr()

    corr_df = corr.reset_index().melt(id_vars="index")
    corr_df.columns = ["Feature1","Feature2","Correlation"]

    heatmap = alt.Chart(corr_df).mark_rect().encode(
        x="Feature2:O",
        y="Feature1:O",
        color=alt.Color("Correlation:Q", scale=alt.Scale(domain=[-1,1], scheme="redblue"))
    ).properties(height=400)

    text = alt.Chart(corr_df).mark_text(size=10).encode(
        x="Feature2:O",
        y="Feature1:O",
        text=alt.Text("Correlation:Q", format=".2f")
    )

    st.altair_chart(heatmap + text, use_container_width=True)

else:
    st.info("Not enough numeric columns for correlation.")

# =====================================================
# INFO
# =====================================================
st.info(
"""
Note: Full dataset (630,000 rows) is loaded for analysis.  
For visualization speed, a random sample of 20,000 rows is used in charts.
"""
)