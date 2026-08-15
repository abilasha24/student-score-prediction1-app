from pathlib import Path

import streamlit as st
import pandas as pd

# =====================================================
# PATH SETUP
# =====================================================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

TRAIN_PATH = ROOT_DIR / "train.csv"

# =====================================================
# PAGE CONFIG
# =====================================================
st.title("📂 Dataset")
st.markdown("Upload and explore the student dataset.")
st.markdown("---")

# =====================================================
# DATA SOURCE
# =====================================================
st.subheader("📁 Dataset Source")

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
        "Upload student dataset CSV",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)

            st.success(
                f"✅ Dataset loaded successfully! "
                f"Rows: {len(df):,} | Columns: {len(df.columns)}"
            )

        except Exception as e:
            st.error("❌ Failed to read the uploaded CSV.")
            st.code(str(e))
            st.stop()

    else:
        st.info("⬆️ Upload a CSV file to explore the dataset.")
        st.stop()

# =====================================================
# LOCAL TRAIN CSV
# =====================================================
else:

    if TRAIN_PATH.exists():

        try:
            df = pd.read_csv(TRAIN_PATH, low_memory=False)

            st.success(
                f"✅ train.csv loaded successfully! "
                f"Rows: {len(df):,} | Columns: {len(df.columns)}"
            )

        except Exception as e:
            st.error("❌ Failed to read train.csv.")
            st.code(str(e))
            st.stop()

    else:

        st.warning("⚠️ train.csv is not available in the deployed app.")

        st.info(
            "Please use **Upload CSV** above to load a dataset."
        )

        st.stop()

# =====================================================
# DATASET PREVIEW
# =====================================================
st.subheader("👀 Dataset Preview")

st.dataframe(
    df.head(20),
    use_container_width=True
)

# =====================================================
# DATASET SUMMARY
# =====================================================
st.subheader("📊 Dataset Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns):,}")
c3.metric("Missing Values", f"{df.isna().sum().sum():,}")
c4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

# =====================================================
# COLUMN INFORMATION
# =====================================================
st.subheader("🧾 Column Information")

column_info = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values,
    "Missing Values": df.isna().sum().values,
    "Unique Values": [
        df[col].nunique(dropna=True)
        for col in df.columns
    ]
})

st.dataframe(
    column_info,
    use_container_width=True
)

# =====================================================
# NUMERIC SUMMARY
# =====================================================
numeric_df = df.select_dtypes(include="number")

if not numeric_df.empty:

    st.subheader("📈 Numerical Statistics")

    st.dataframe(
        numeric_df.describe().T,
        use_container_width=True
    )

# =====================================================
# DOWNLOAD
# =====================================================
st.subheader("⬇️ Download Dataset")

st.download_button(
    label="Download Current Dataset",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="student_dataset.csv",
    mime="text/csv",
    use_container_width=True
)