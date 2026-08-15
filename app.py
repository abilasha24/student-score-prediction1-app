import streamlit as st

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Student Exam Score Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.main-title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #2563eb, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.sub-title {
    text-align: center;
    color: #6b7280;
    font-size: 1.2rem;
    margin-bottom: 1.5rem;
}

.card {
    background-color: #f8fafc;
    padding: 28px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    border: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
}

.section-title {
    color: #1f2937;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 0.9rem;
    margin-top: 2rem;
}

.highlight {
    color: #2563eb;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("<div class='main-title'>🎓 Student Exam Score Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>A Machine Learning-Based Multi-Page Educational Web Application</div>", unsafe_allow_html=True)
st.markdown("---")

st.success("👋 Welcome Abi! Use the sidebar to navigate through Dataset, Model Training, Prediction, and Analytics.")

# =====================================================
# MAIN CONTENT
# =====================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class='card'>
        <div class='section-title'>📌 Project Overview</div>
        <p>
            This web application is a <span class='highlight'>Machine Learning-based Student Exam Score Prediction System</span>
            developed to help students and educators analyse student-related data and estimate exam performance.
        </p>
        <p>
            The system uses features such as <b>age, gender, course, study hours, class attendance, internet access,
            sleep hours, sleep quality, study method, facility rating, and exam difficulty</b> to predict student exam scores
            using trained machine learning and deep learning models.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='section-title'>🎯 Project Objectives</div>
        <ul>
            <li>Dataset upload and visualisation</li>
            <li>Exploratory Data Analysis (EDA)</li>
            <li>Model training and evaluation</li>
            <li>Real-time student score prediction</li>
            <li>Analytics dashboard for model insights</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
        <div class='section-title'>🛠️ Technologies Used</div>
        <ul>
            <li>Python</li>
            <li>Pandas and NumPy</li>
            <li>Scikit-learn</li>
            <li>Streamlit</li>
            <li>Joblib</li>
            <li>Matplotlib / Altair</li>
            <li>TensorFlow / Keras (DNN)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.info("""
### 📂 How to Use This App

**1. Dataset Page**  
Upload or view the dataset and explore the data.

**2. Model Training Page**  
Train and evaluate machine learning models.

**3. Prediction Page**  
Enter student details or upload test data to predict exam scores.

**4. Analytics Page**  
View charts, model comparison, and prediction history.
""")

    st.markdown("""
    <div class='card'>
        <div class='section-title'>🎓 Web App Information</div>
        <p><b>Project Title:</b> Student Exam Score Prediction System</p>
        <p><b>Category:</b> Educational Machine Learning Web Application</p>
        <p><b>Models Used:</b> Random Forest, Gradient Boosting, Deep Neural Network</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# BOTTOM SECTION
# =====================================================
st.markdown("---")

st.subheader("🚀 Key Features")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Dataset Support", "CSV Upload")
with c2:
    st.metric("ML Models", "3")
with c3:
    st.metric("Prediction Type", "Exam Score")
with c4:
    st.metric("Interface", "Multi-Page")

st.markdown("---")
st.markdown(
    "<div class='footer'>© 2026 Student Exam Score Prediction System | Developed by Abilasha Selvanayakam</div>",
    unsafe_allow_html=True
)