import streamlit as st

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Student Exam Score Prediction System",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# CUSTOM STYLING
# =====================================================
st.markdown("""
<style>
.main-title {
    font-size: 2.8rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #2563eb, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sub-title {
    text-align: center;
    color: gray;
    font-size: 1.2rem;
    margin-bottom: 20px;
}
.card {
    background-color: #f5f7ff;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 0px 8px #ddd;
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    color: gray;
    font-size: 0.9rem;
    margin-top: 20px;
}
ul li {
    margin-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# MAIN HOME PAGE CONTENT
# =====================================================
st.markdown("<div class='main-title'>🎓 Student Exam Score Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>A Machine Learning-Based Web Application for Predicting Student Exam Scores</div>", unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div class='card'>

## 💡 Project Overview
The Student Exam Score Prediction System is a machine learning-based web application developed to analyse academic and lifestyle-related data and estimate student exam scores.  
It is designed to support students, educators, and academic institutions in understanding performance-related patterns and making data-driven decisions.

## ⚙️ Technologies Used
- Python  
- Pandas and NumPy  
- Scikit-learn  
- TensorFlow / Keras  
- Streamlit  
- Joblib  
- Matplotlib / Altair  
- Kaggle Student Dataset  

## 📌 System Features
✔ Upload and explore student dataset  
✔ Perform exploratory data analysis (EDA)  
✔ Train and evaluate machine learning models  
✔ Predict student exam scores  
✔ Visualise trends and analytics  
✔ Track prediction history  

</div>
""", unsafe_allow_html=True)

with st.expander("📂 How to Use This App"):
    st.markdown("""
1️⃣ **Dataset Page:** Upload or view the student dataset and perform exploratory data analysis.  
2️⃣ **Model Training Page:** Train and evaluate the machine learning models used in the system.  
3️⃣ **Prediction Page:** Enter student details or upload input data to generate predicted exam scores.  
4️⃣ **Analytics Page:** View charts, model results, and prediction history.  
""")

student_name = st.text_input("👋 Enter your name to personalise the welcome message:", "")
if student_name:
    st.success(f"Welcome, {student_name}! Use the sidebar to navigate through the application pages.")

st.markdown("---")
st.markdown(
    "<div class='footer'>© 2026 Student Exam Score Prediction System | Developed by Abilasha Selvanayakam</div>",
    unsafe_allow_html=True
)