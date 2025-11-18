# 🩺 Disease Prediction System using Machine Learning  
A Complete ML Project • Kaggle Dataset • Explainable AI • Streamlit Web App • PDF Report Generator

---

## 📌 Overview

This project is a **full ML pipeline** that predicts diseases based on symptoms using a dataset of **132 symptoms and 41 diseases**.  
The system includes:

- Machine Learning model (RF + ExtraTrees Ensemble)  
- Streamlit web application  
- SHAP Explainable AI (why the model predicted a disease)  
- PDF Medical Report Export  
- Disease descriptions, tests & suggestions  
- Modern professional UI  
- Kaggle dataset integration  

This is a **lab-ready, presentation-ready** project that demonstrates real-world machine learning, interpretability, and deployment.

---

## 🌟 Features

### ✔ 41 Diseases  
### ✔ 132 Symptoms  
### ✔ Ensemble Model (RandomForest + ExtraTrees)  
### ✔ SHAP Explainability  
### ✔ Professional PDF Export  
### ✔ Disease Descriptions + Suggested Tests  
### ✔ Beautiful UI (Custom CSS Skin)  
### ✔ Top-N Predicted Diseases  
### ✔ Confidence + Risk Level  
### ✔ Patient Summary Section  
### ✔ PDF Report with SHAP Chart  
### ✔ Streamlit-based Web App  
### ✔ Fully Offline  

---

## 🧠 Architecture

            ┌────────────────────┐
            │ Kaggle Dataset     │
            │ (41 diseases,      │
            │ 132 symptoms)      │
            └─────────┬──────────┘
                      │
            ┌─────────▼──────────┐
            │ Preprocessing       │
            │ - Clean symptoms    │
            │ - Encode labels     │
            │ - Handle missing    │
            └─────────┬──────────┘
                      │
            ┌─────────▼──────────┐
            │ ML Model Training   │
            │ Ensemble: RF + ET   │
            │ Accuracy ~97%       │
            └─────────┬──────────┘
                      │
            ┌─────────▼──────────┐
            │ SHAP Explainability │
            │ Feature Importance  │
            └─────────┬──────────┘
                      │
            ┌─────────▼──────────┐
            │ Streamlit Web App   │
            │ - Symptom Input     │
            │ - Prediction        │
            │ - SHAP Graph        │
            │ - PDF Report        │
            │ - Suggestions       │
            └─────────────────────┘

---

## 🛠️ Technologies Used

- **Python 3.10+**  
- **Pandas, NumPy**  
- **Scikit-Learn**  
- **SHAP**  
- **Joblib**  
- **ReportLab (For PDF)**  
- **Streamlit**  
- **Matplotlib**  

---

## 📂 Project Structure

Disease-Prediction-Project/
│
├── data/
│ ├── kaggle_raw.csv
│ ├── kaggle_preprocessed.csv
│ └── symptoms.json
│
├── artifacts/
│ ├── model.joblib
│ ├── label_encoder.joblib
│ └── metrics.txt
│
├── src/
│ ├── app.py # Streamlit Web UI
│ ├── train.py # ML Training Pipeline
│ ├── prepare_kaggle_dataset.py
│ ├── symptoms.py
│ ├── predict.py
│ └── generate_dataset.py
│
├── scripts/
│ └── run_app.ps1 # Optional launch script
│
├── requirements.txt
└── README.md


---

## 🚀 Quick Start

### 1️⃣ Create virtual environment

```powershell
> python -m venv .venv
> .\.venv\Scripts\Activate.ps1
> pip install -r requirements.txt

Prepare Dataset (Kaggle → Training Ready)
>python src/prepare_kaggle_dataset.py
This creates:
data/kaggle_preprocessed.csv

Train the ML Model
> python src/train.py --data data/kaggle_preprocessed.csv --out artifacts/model.joblib
Outputs generated:

artifacts/model.joblib

artifacts/label_encoder.joblib

artifacts/metrics.txt

Make a CLI Prediction
> python src/predict.py --model artifacts/model.joblib --symptoms 1,0,0,1,0,0,0,0,1,0 --top 5

Run the Streamlit Web App
> streamlit run src/app.py

🎯 FULL EXECUTION SUMMARY
# 1. Create clean venv
python -m venv .venv

# 2. Activate
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit reportlab shap scikit-learn pandas numpy matplotlib joblib

# 4. Prepare dataset
python src/prepare_kaggle_dataset.py

# 5. Train model
python src/train.py --data data/kaggle_preprocessed.csv --out artifacts/model.joblib

# 6. Run Web App
python -m streamlit run src/app.py

```

🧠 Model Explainability (SHAP)

The app displays:

Top 10 symptoms influencing prediction

SHAP bar chart

Per-instance interpretability

This explains why the model predicted a specific disease.

📄 PDF Medical Report

One-click export includes:

Patient details

Selected symptoms

Predicted disease

Confidence level

Risk category

SHAP explanation image

Disease description & suggested tests

Timestamp

Professional, printable, and impressive.

📚 Disease Intelligence Layer

For every predicted disease, app shows:

Description

Common diagnostic tests

General suggestions

(Implemented through DISEASE_INFO dictionary)

🔥 Enhancements Included

Full redesigned UI

Custom CSS theme

Cards, shadows, gradients

Improved layout

Better readability

Professional feel

📈 Model Accuracy

Approximate metrics from Kaggle dataset:

Metric	Value
Accuracy	96–98%
Precision	High
Recall	High
Classes	41

See artifacts/metrics.txt after training.

📦 Future Work (Optional Ideas)

Add voice input (speech-to-text)

Use FastAPI backend for separate deployment

Convert to Android APK using WebView

Add chatbot assistant

Expand disease descriptions

Add confusion matrix visualization

Hyperparameter tuning with Optuna

👨‍🏫 Academic Notes

This project demonstrates:

Dataset preprocessing

Feature engineering

Multi-class classification

Ensemble learning

SHAP interpretability

Web UI deployment

PDF report generation

Human-centered ML design

Perfect for Machine Learning Lab, Viva, and Project Evaluation.

📝 Credits

Developed by:
Arman Ahemad Khan & Team
Silicon University, Odisha

Guided by:

Machine Learning Lab — Academic Project Submission
