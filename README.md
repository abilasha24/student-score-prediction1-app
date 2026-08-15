# 🎓 Student Exam Score Prediction System

A machine-learning web application that predicts student exam performance from academic and lifestyle features, built with Scikit-learn and deployed as an interactive multi-page Streamlit app.

## 🚀 Live Demo

🔗 **App:** https://student-score-prediction1-app-an5yveweb5jgse5c7c7kcg.streamlit.app/

> Hosted on Streamlit Community Cloud, with UptimeRobot pinging every 5 minutes to prevent the free-tier app from sleeping.

## ✨ Features

- 📊 **Dataset explorer** — view and inspect the training data
- 🧠 **Model training** — trains and compares Random Forest and Gradient Boosting regressors (with train/test split, feature scaling, and evaluation metrics: R², MAE, RMSE)
- 🔮 **Prediction** — enter student details and get a predicted exam score from the trained models
- 📈 **Analytics dashboard** — score distribution and prediction-history insights, built with Altair charts
- 🏠 Clean multi-page Streamlit layout with custom CSS styling

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** Scikit-learn (Random Forest, Gradient Boosting), NumPy, Pandas
- **UI:** Streamlit (multi-page app)
- **Visualization:** Altair, Matplotlib
- **Model Persistence:** Joblib

## 🏗️ Project Structure

```text
student-score-prediction1-app
├── app.py                # Main entry point / landing page
├── requirements.txt
├── rf_model.pkl           # Trained Random Forest model
├── gbr_model.pkl          # Trained Gradient Boosting model
├── feature_cols.pkl        # Feature column reference
├── dnn_scaler.pkl          # Feature scaler
└── pages/
    ├── home.py
    ├── Dataset.py           # Dataset explorer
    ├── Model_Training.py      # Model training & evaluation
    ├── Prediction.py          # Score prediction interface
    └── Analytics.py           # Prediction analytics dashboard
```

## 🏃 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/abilasha24/student-score-prediction1-app.git
cd student-score-prediction1-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

## 📌 Project Highlights

- End-to-end ML workflow: data exploration → preprocessing → feature engineering → model training → evaluation → deployment
- Multiple regression models trained and compared side by side
- Interactive, multi-page Streamlit interface for both technical and non-technical users
- Deployed and monitored for uptime on a free-tier cloud host

## 👩‍💻 Author

**Abilasha Selvanayakam**

- GitHub: https://github.com/abilasha24
- LinkedIn: https://www.linkedin.com/in/abilashaselvanayakam2k06/
- Portfolio: https://my-portfolio-webapp-ashy.vercel.app/
