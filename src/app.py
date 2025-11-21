import json
import os
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Try importing SHAP (for explainability)
try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors

from symptoms import SYMPTOM_NAMES as FALLBACK_SYMPTOMS

MODEL_PATH = "artifacts/model.joblib"
ENCODER_PATH = "artifacts/label_encoder.joblib"
SYMPTOMS_JSON = "data/symptoms.json"


# ----------------------------------------------------
# THEME / CUSTOM STYLES
# ----------------------------------------------------
def apply_custom_theme():
    st.markdown(
        """
        <style>
        /* MAIN BACKGROUND */
        .main {
            background-color: #0e1117;
            padding: 0px !important;
        }

        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background: #161a21;
        }
        section[data-testid="stSidebar"] * {
            color: #f5f5f5 !important;
        }

        /* HEADER CARD */
        .app-header {
            padding: 20px;
            background: linear-gradient(90deg, #4f8bf9, #6fc3df);
            color: white;
            border-radius: 10px;
            margin-bottom: 25px;
        }

        .app-header h1 {
            color: white;
            font-size: 32px;
            margin: 0px;
            font-weight: 800;
        }

        .app-header p {
            font-size: 15px;
            margin-top: 8px;
        }

        /* GENERIC CARD LOOK */
        .stCard {
            background: #161a21;
            padding: 20px 25px;
            border-radius: 15px;
            box-shadow: rgba(0, 0, 0, 0.35) 0px 4px 15px;
            margin-bottom: 25px;
            border: 1px solid #262d3b;
        }

        .stCard h3, .stCard h4, .stCard p, .stCard li, .stCard span {
            color: #f5f5f5 !important;
        }

        /* BUTTON */
        button[kind="primary"] {
            border-radius: 10px !important;
            padding: 12px 18px;
            font-size: 16px;
            font-weight: bold;
        }

        /* GENERAL TEXT COLOR */
        .markdown-text-container, .stMarkdown, .stText, .stDataFrame, .stTable {
            color: #f5f5f5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------
# Disease information (description, tests, general suggestions)
# Keys should match Kaggle `prognosis` labels
# --------------------------------------------------------------
DISEASE_INFO = {
    "Fungal infection": {
        "description": (
            "A skin or mucosal infection caused by fungi, often leading to itching, "
            "redness, and rashes in moist areas of the body."
        ),
        "tests": "Skin scraping & KOH exam, fungal culture, physical examination.",
        "suggestions": (
            "Keep the affected area clean and dry, avoid tight clothing, and consult "
            "a doctor or dermatologist for proper evaluation."
        ),
    },
    "Allergy": {
        "description": (
            "An overreaction of the immune system to substances such as dust, pollen, "
            "foods, or medicines, causing sneezing, rashes, or breathing difficulty."
        ),
        "tests": "Allergy skin prick test, blood IgE levels, detailed history.",
        "suggestions": (
            "Avoid suspected triggers, monitor symptoms, and seek medical advice if "
            "breathing difficulty or severe swelling occurs."
        ),
    },
    "GERD": {
        "description": (
            "Gastroesophageal Reflux Disease (GERD) occurs when stomach acid "
            "frequently flows back into the esophagus, causing heartburn and discomfort."
        ),
        "tests": "Upper GI endoscopy, pH monitoring, barium swallow X-ray.",
        "suggestions": (
            "Avoid heavy meals, spicy/oily foods, and lying down immediately after eating. "
            "Consult a doctor if symptoms are frequent or severe."
        ),
    },
    "Chronic cholestasis": {
        "description": (
            "A long-term reduction or stoppage of bile flow, leading to jaundice, itching, "
            "and digestion-related issues."
        ),
        "tests": "Liver function tests, abdominal ultrasound, MRCP, liver biopsy.",
        "suggestions": (
            "This condition needs specialist care. Consult a gastroenterologist or "
            "hepatologist for detailed evaluation."
        ),
    },
    "Drug Reaction": {
        "description": (
            "An unwanted or harmful response to a medication, which may present as rash, "
            "itching, swelling, or more severe symptoms."
        ),
        "tests": "Clinical evaluation, review of recent medications, allergy testing if needed.",
        "suggestions": (
            "Stop the suspected drug only after consulting a doctor. Seek urgent care if "
            "there is difficulty breathing, facial swelling, or widespread rash."
        ),
    },
    "Peptic ulcer diseae": {
        "description": (
            "Sores that develop on the inner lining of the stomach or upper intestine, "
            "often causing burning pain in the upper abdomen."
        ),
        "tests": "Endoscopy, H. pylori testing, stool antigen test.",
        "suggestions": (
            "Avoid painkillers on your own, reduce smoking/alcohol, and consult a doctor "
            "for proper management."
        ),
    },
    "AIDS": {
        "description": (
            "Acquired Immunodeficiency Syndrome, a late stage of HIV infection that "
            "weakens the immune system and increases risk of infections and cancers."
        ),
        "tests": "HIV antibody/antigen tests, CD4 count, viral load tests.",
        "suggestions": (
            "Requires lifelong specialist follow-up. Consult an infectious disease specialist "
            "or HIV clinic for counselling and treatment."
        ),
    },
    "Diabetes ": {
        "description": (
            "A condition where the body cannot properly regulate blood sugar, leading to "
            "high glucose levels over time."
        ),
        "tests": "Fasting blood sugar, HbA1c, oral glucose tolerance test.",
        "suggestions": (
            "Follow a healthy lifestyle, regular monitoring, and consult a doctor/endocrinologist "
            "for long-term management."
        ),
    },
    "Gastroenteritis": {
        "description": (
            "Inflammation of the stomach and intestines, commonly causing vomiting, diarrhea, "
            "and abdominal cramps."
        ),
        "tests": "Stool tests (if severe), dehydration assessment, basic blood tests.",
        "suggestions": (
            "Maintain hydration with fluids, avoid street food, and seek medical help if "
            "there is blood in stool, high fever, or severe weakness."
        ),
    },
    "Bronchial Asthma": {
        "description": (
            "A chronic condition where airways become inflamed and narrow, causing wheezing, "
            "shortness of breath, and chest tightness."
        ),
        "tests": "Spirometry (lung function test), peak flow measurement, chest exam.",
        "suggestions": (
            "Avoid triggers like smoke and dust. Regular follow-up with a doctor is important "
            "for inhaler use and long-term control."
        ),
    },
    "Hypertension ": {
        "description": (
            "Persistently high blood pressure, which increases the risk of heart disease, "
            "stroke, and kidney damage over time."
        ),
        "tests": "Blood pressure monitoring, kidney function tests, ECG, lipid profile.",
        "suggestions": (
            "Reduce salt intake, maintain a healthy weight, and consult a doctor for proper "
            "evaluation and monitoring."
        ),
    },
    "Migraine": {
        "description": (
            "A type of headache often associated with throbbing pain, sensitivity to light/sound, "
            "and sometimes nausea."
        ),
        "tests": "Mainly clinical diagnosis; sometimes CT/MRI to rule out other causes.",
        "suggestions": (
            "Identify and avoid triggers (like lack of sleep, certain foods). For frequent attacks, "
            "consult a doctor for preventive strategies."
        ),
    },
    "Cervical spondylosis": {
        "description": (
            "Age-related wear and tear affecting the joints and discs in the neck, possibly causing "
            "neck pain and stiffness."
        ),
        "tests": "X-ray of cervical spine, MRI if nerve involvement is suspected.",
        "suggestions": (
            "Posture correction, neck exercises, and medical review are important if pain persists."
        ),
    },
    "Paralysis (brain hemorrhage)": {
        "description": (
            "Loss of muscle function (often on one side of the body) due to bleeding in the brain, "
            "usually a medical emergency."
        ),
        "tests": "CT/MRI brain, neurological examination, blood pressure assessment.",
        "suggestions": (
            "This is an emergency condition. Immediate hospital care and specialist consultation are critical."
        ),
    },
    "Jaundice": {
        "description": (
            "Yellowing of skin and eyes due to high bilirubin levels, often related to liver or bile duct problems."
        ),
        "tests": "Liver function tests, ultrasound abdomen, viral hepatitis markers.",
        "suggestions": (
            "Avoid alcohol and self-medication. Timely evaluation by a doctor is important."
        ),
    },
    "Malaria": {
        "description": (
            "A mosquito-borne infection causing fever, chills, and flu-like symptoms, common in many tropical regions."
        ),
        "tests": "Peripheral blood smear, rapid malaria antigen tests.",
        "suggestions": (
            "Seek medical help early, especially with high fever and chills. Use mosquito precautions."
        ),
    },
    "Chicken pox": {
        "description": (
            "A viral infection causing itchy, blister-like rashes all over the body, often with fever."
        ),
        "tests": "Usually clinical; rarely blood tests for antibodies.",
        "suggestions": (
            "Avoid scratching lesions, maintain hygiene, and avoid contact with pregnant women or immunocompromised people."
        ),
    },
    "Dengue": {
        "description": (
            "A viral infection transmitted by mosquitoes, often causing high fever, body pain, and sometimes bleeding."
        ),
        "tests": "Dengue NS1 antigen test, IgM/IgG antibodies, platelet count.",
        "suggestions": (
            "Avoid self-medication with painkillers like NSAIDs, maintain hydration, and seek medical care for warning signs."
        ),
    },
    "Typhoid": {
        "description": (
            "A bacterial infection typically spread by contaminated food or water, causing persistent fever and abdominal symptoms."
        ),
        "tests": "Blood culture, Widal test, stool culture.",
        "suggestions": (
            "Ensure safe drinking water and proper hygiene. Medical treatment is needed for proper cure."
        ),
    },
    "hepatitis A": {
        "description": (
            "A viral infection affecting the liver, often spread via contaminated food or water."
        ),
        "tests": "Liver function tests, HAV IgM antibody test.",
        "suggestions": (
            "Rest, good hydration, and avoiding alcohol are important. Consult a doctor for evaluation."
        ),
    },
    "Hepatitis B": {
        "description": (
            "A viral liver infection that can become chronic and may lead to liver damage over time."
        ),
        "tests": "HBsAg, HBeAg, HBV DNA, liver function tests.",
        "suggestions": (
            "Requires specialist review. Follow medical advice for monitoring and treatment."
        ),
    },
    "Hepatitis C": {
        "description": (
            "A viral infection that primarily affects the liver and may lead to chronic liver disease."
        ),
        "tests": "Anti-HCV antibody test, HCV RNA, liver function tests.",
        "suggestions": (
            "Consult a liver specialist for further evaluation and long-term follow-up."
        ),
    },
    "Hepatitis D": {
        "description": (
            "A viral infection that occurs only in people infected with Hepatitis B, and can worsen liver disease."
        ),
        "tests": "Anti-HDV antibody, liver function tests.",
        "suggestions": (
            "Specialist evaluation is important for combined management of hepatitis B and D."
        ),
    },
    "Hepatitis E": {
        "description": (
            "A viral infection of the liver, typically spread through contaminated water."
        ),
        "tests": "HEV IgM antibody, liver function tests.",
        "suggestions": (
            "Usually self-limiting but requires rest and hydration. Important in pregnancy—medical advice is essential."
        ),
    },
    "Alcoholic hepatitis": {
        "description": "Liver inflammation caused by long-term heavy alcohol use.",
        "tests": "Liver function tests, ultrasound liver, history of alcohol intake.",
        "suggestions": (
            "Stopping alcohol and consulting a doctor or specialist is very important."
        ),
    },
    "Tuberculosis": {
        "description": (
            "A bacterial infection that most often affects the lungs, causing cough, fever, and weight loss."
        ),
        "tests": "Chest X-ray, sputum AFB, GeneXpert, tuberculin skin test.",
        "suggestions": (
            "Requires long-term supervised treatment. Consult a TB clinic or chest specialist."
        ),
    },
    "Common Cold": {
        "description": (
            "A mild viral infection of the nose and throat, causing sneezing, sore throat, and runny nose."
        ),
        "tests": "Usually clinical; tests rarely required.",
        "suggestions": (
            "Rest, fluids, and basic hygiene. Consult a doctor if symptoms persist or worsen."
        ),
    },
    "Pneumonia": {
        "description": (
            "Infection that inflames the air sacs in one or both lungs, causing cough, fever, and breathing difficulty."
        ),
        "tests": "Chest X-ray, blood tests, sputum culture.",
        "suggestions": (
            "Seek medical attention early, especially if breathing is difficult or fever is high."
        ),
    },
    "Dimorphic hemmorhoids(piles)": {
        "description": (
            "Swollen and inflamed veins in the rectum and anus that cause discomfort and bleeding."
        ),
        "tests": "Physical exam, proctoscopy.",
        "suggestions": (
            "High-fiber diet, adequate fluids; seek medical advice for persistent bleeding or pain."
        ),
    },
    "Heart attack": {
        "description": (
            "A medical emergency where blood flow to part of the heart is blocked, causing chest pain and potential damage."
        ),
        "tests": "ECG, cardiac enzymes, echocardiography, coronary angiography.",
        "suggestions": (
            "This is an emergency — immediate hospital care is essential if suspected."
        ),
    },
    "Varicose veins": {
        "description": (
            "Enlarged, twisted veins often seen in the legs due to faulty valves in the veins."
        ),
        "tests": "Doppler ultrasound of leg veins.",
        "suggestions": (
            "Avoid prolonged standing, consider leg elevation and medical review for severe cases."
        ),
    },
    "Hypothyroidism": {
        "description": (
            "Underactive thyroid gland leading to fatigue, weight gain, and cold intolerance."
        ),
        "tests": "TSH, Free T4 blood tests.",
        "suggestions": (
            "Regular thyroid function monitoring and medical guidance are important."
        ),
    },
    "Hyperthyroidism": {
        "description": (
            "Overactive thyroid gland causing weight loss, palpitations, anxiety, and heat intolerance."
        ),
        "tests": "TSH, Free T3, Free T4, thyroid scan.",
        "suggestions": (
            "Seek medical advice for evaluation and management of thyroid levels."
        ),
    },
    "Hypoglycemia": {
        "description": (
            "Low blood sugar levels, which can lead to dizziness, sweating, confusion, or fainting."
        ),
        "tests": "Blood glucose testing during symptoms.",
        "suggestions": (
            "Requires prompt carbohydrate intake and evaluation, especially in people on diabetes medicines."
        ),
    },
    "Osteoarthristis": {
        "description": (
            "Degenerative joint disease causing pain, stiffness, and reduced mobility, often in knees or hips."
        ),
        "tests": "X-rays of affected joints, physical examination.",
        "suggestions": (
            "Weight management, exercises, and medical advice can help manage symptoms."
        ),
    },
    "Arthritis": {
        "description": (
            "Inflammation of joints causing pain and stiffness; may have many types such as rheumatoid arthritis."
        ),
        "tests": "Joint exam, X-rays, ESR/CRP, rheumatoid factor (if suspected).",
        "suggestions": (
            "Early diagnosis and treatment can help prevent joint damage. Consult a doctor/rheumatologist."
        ),
    },
    "(vertigo) Paroymsal Positional Vertigo": {
        "description": (
            "Sudden episodes of dizziness or spinning sensation triggered by changes in head position."
        ),
        "tests": "Clinical positional tests (e.g., Dix–Hallpike), sometimes MRI to rule out other causes.",
        "suggestions": (
            "Avoid sudden head movements and consult a doctor if episodes are frequent."
        ),
    },
    "Acne": {
        "description": (
            "A common skin condition causing pimples, usually on face, chest, or back."
        ),
        "tests": "Mostly clinical; tests usually not required.",
        "suggestions": (
            "Maintain gentle skin care. For severe or scarring acne, consult a dermatologist."
        ),
    },
    "Urinary tract infection": {
        "description": (
            "Infection in any part of the urinary system, often causing burning urination and frequent urge."
        ),
        "tests": "Urine routine & culture.",
        "suggestions": (
            "Drink plenty of water and consult a doctor, especially if there is fever or back pain."
        ),
    },
    "Psoriasis": {
        "description": (
            "A chronic skin condition causing red, scaly patches, commonly on elbows, knees, or scalp."
        ),
        "tests": "Skin examination; sometimes skin biopsy.",
        "suggestions": (
            "Requires long-term skin care and sometimes specialist management."
        ),
    },
    "Impetigo": {
        "description": (
            "A contagious bacterial skin infection causing red sores, often around the nose and mouth in children."
        ),
        "tests": "Clinical exam; sometimes swab of lesions.",
        "suggestions": (
            "Maintain hygiene and avoid sharing towels. Medical treatment is usually required."
        ),
    },
}


def get_disease_info(name: str):
    """Return disease description/tests/suggestions or a generic fallback."""
    info = DISEASE_INFO.get(name)
    if not info:
        return {
            "description": "Detailed information for this disease is not available in the app.",
            "tests": "Consult a doctor for appropriate diagnostic tests.",
            "suggestions": (
                "Please seek professional medical advice for personalised recommendations."
            ),
        }
    return info


# --------------------------------------------------------------
# Load model, symptoms, encoder
# --------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH) if os.path.exists(ENCODER_PATH) else None

    if os.path.exists(SYMPTOMS_JSON):
        with open(SYMPTOMS_JSON, "r", encoding="utf-8") as f:
            symptoms = json.load(f)
    else:
        symptoms = FALLBACK_SYMPTOMS

    return model, label_encoder, symptoms


def build_feature_vector(selected_symptoms, symptom_names):
    vals = [1 if s in selected_symptoms else 0 for s in symptom_names]
    return np.array(vals).reshape(1, -1)


def get_risk_level(prob: float) -> str:
    if prob >= 0.85:
        return "🔴 Critical"
    elif prob >= 0.65:
        return "🟠 High"
    elif prob >= 0.40:
        return "🟡 Medium"
    else:
        return "🟢 Low"


# --------------------------------------------------------------
# HYBRID EXPLAINABILITY: SHAP + FEATURE IMPORTANCE FALLBACK
# --------------------------------------------------------------
def explain_prediction(model, X_instance, feature_names):
    """
    Try SHAP first. If SHAP is unavailable or the model is unsupported (e.g. VotingClassifier),
    gracefully fall back to feature-importance based explanation.
    """
    # 1. If SHAP library not available, skip directly to fallback
    if not SHAP_AVAILABLE:
        return feature_importance_figure(model, feature_names, note="SHAP not installed")

    # 2. Try SHAP TreeExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance)

        # Multi-class case: shap_values is a list [n_classes][n_samples, n_features]
        if isinstance(shap_values, list):
            # Choose the predicted class index
            pred_proba = model.predict_proba(X_instance)[0]
            pred_class_idx = np.argmax(pred_proba)
            shap_vals = shap_values[pred_class_idx][0]
        else:
            shap_vals = shap_values[0]

        abs_vals = np.abs(shap_vals)
        top_idx = np.argsort(abs_vals)[::-1][:10]

        top_feats = [feature_names[i] for i in top_idx]
        top_importance = [shap_vals[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top_feats[::-1], top_importance[::-1], color="skyblue")
        ax.set_title("Top 10 Symptoms Influencing Prediction (SHAP)")
        ax.set_xlabel("SHAP Value (Impact on Model Output)")
        plt.tight_layout()

        return fig

    except Exception:
        # 3. Fallback to feature importances
        return feature_importance_figure(
            model,
            feature_names,
            note="SHAP not supported for this ensemble model – using feature importance instead.",
        )


def feature_importance_figure(model, feature_names, note=None):
    """
    Build a bar chart using feature_importances_.
    For VotingClassifier, average importances of tree-based base estimators.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "estimators_"):
            # VotingClassifier: average importances of individual tree estimators
            est_importances = []
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    est_importances.append(est.feature_importances_)
            if not est_importances:
                return "Explainability unavailable: no feature_importances_ found."
            importances = np.mean(est_importances, axis=0)
        else:
            return "Explainability unavailable: model has no feature_importances_."

        importances = np.array(importances)
        top_idx = np.argsort(importances)[::-1][:10]

        top_feats = [feature_names[i] for i in top_idx]
        top_vals = [importances[i] for i in top_idx]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(top_feats[::-1], top_vals[::-1], color="lightgreen")
        ax.set_title("Top 10 Influencing Symptoms (Feature Importance)")
        ax.set_xlabel("Importance")
        if note:
            ax.text(
                0.5,
                -0.18,
                note,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=8,
            )
        plt.tight_layout()
        return fig

    except Exception as e:
        return f"Explainability unavailable: {e}"


# --------------------------------------------------------------
# PREMIUM PDF GENERATION (UPDATED FOR PATIENT NAME)
# --------------------------------------------------------------
def draw_section_title(pdf, x, y, title):
    pdf.setFont("Helvetica-Bold", 14)
    pdf.setFillColor(colors.HexColor("#184e77"))
    pdf.drawString(x, y, title)
    pdf.setFillColor(colors.black)
    pdf.setLineWidth(0.5)
    pdf.line(x, y - 3, x + 500, y - 3)


def generate_premium_pdf(patient_name, patient_info, prediction, disease_info, top_probs, shap_fig=None):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Header
    pdf.setFillColor(colors.HexColor("#1d3557"))
    pdf.rect(0, height - 80, width, 80, stroke=0, fill=1)

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, height - 50, "Disease Prediction Report")

    pdf.setFont("Helvetica", 10)
    pdf.drawString(400, height - 35, "Academic ML Project")
    pdf.drawString(400, height - 50, f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

    y = height - 100

    # Patient details
    draw_section_title(pdf, 40, y, "1. Patient Details")
    y -= 25

    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, f"Patient Name: {patient_name}")
    y -= 20

    pdf.setFont("Helvetica", 11)
    for line in patient_info:
        pdf.drawString(50, y, f"- {line}")
        y -= 18

    y -= 10

    # Prediction summary
    draw_section_title(pdf, 40, y, "2. Prediction Summary")
    y -= 25
    pdf.setFont("Helvetica-Bold", 11)
    for line in prediction:
        pdf.drawString(50, y, line)
        y -= 18

    y -= 10

    # Disease details
    draw_section_title(pdf, 40, y, "3. Disease Overview (Top Prediction)")
    y -= 25

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Description: {disease_info['description']}")
    y -= 18
    pdf.drawString(50, y, f"Common tests: {disease_info['tests']}")
    y -= 18
    pdf.drawString(50, y, f"General suggestions: {disease_info['suggestions']}")
    y -= 25

    # Top predictions
    draw_section_title(pdf, 40, y, "4. Model Top Predictions")
    y -= 25
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y, "Disease")
    pdf.drawString(320, y, "Probability")
    y -= 15
    pdf.line(50, y, 520, y)
    y -= 10

    pdf.setFont("Helvetica", 11)
    for disease, prob in top_probs:
        pdf.drawString(50, y, str(disease))
        pdf.drawString(320, y, f"{prob:.2%}")
        y -= 16

    y -= 10

    # SHAP Image
    if shap_fig is not None:
        if y < 250:
            pdf.showPage()
            y = height - 80

        draw_section_title(pdf, 40, y, "5. Explainability")
        y -= 20

        img_buf = io.BytesIO()
        shap_fig.savefig(img_buf, format="png", bbox_inches="tight")
        img_buf.seek(0)

        try:
            pdf.drawImage(ImageReader(img_buf), 50, y - 220, width=500, height=220)
            y -= 240
        except:
            pdf.setFont("Helvetica", 10)
            pdf.drawString(50, y - 20, "Could not render SHAP image.")
            y -= 40

    # Disclaimer
    if y < 120:
        pdf.showPage()
        y = height - 80

    draw_section_title(pdf, 40, y, "6. Important Disclaimer")
    y -= 25

    disclaimer = (
        "This report is generated as part of a machine learning academic project. "
        "It is NOT a medical diagnosis and must not be used as a substitute for "
        "professional medical advice. Consult a qualified doctor for health concerns."
    )

    pdf.setFont("Helvetica-Oblique", 9)
    text_obj = pdf.beginText(50, y)
    text_obj.textLines(disclaimer)
    pdf.drawText(text_obj)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

# --------------------------------------------------------------
# MAIN STREAMLIT APP (PATIENT NAME ADDED)
# --------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Disease Prediction from Symptoms",
        page_icon="🩺",
        layout="wide",
    )

    apply_custom_theme()

    model, label_encoder, symptom_names = load_artifacts()

    # Header
    st.markdown(
        """
        <div class="app-header">
            <h1>🩺 Disease Prediction System using Machine Learning</h1>
            <p>
            Predict possible diseases, view explainability insights,
            and generate a premium medical-style PDF report.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar – Patient Info
    st.sidebar.header("👤 Patient Information")

    patient_name = st.sidebar.text_input("Patient Name", "John Doe")

    age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.sidebar.radio("Gender", ["Male", "Female", "Other"], index=0)
    duration = st.sidebar.slider("Symptom duration (days)", 0, 30, 3)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Summary")
    st.sidebar.write(f"- Features: **{len(symptom_names)} symptoms**")
    if label_encoder:
        st.sidebar.write(f"- Diseases: **{len(label_encoder.classes_)}**")

    top_n = st.sidebar.slider("Top predictions to show", 1, 10, 5)

    # Layout columns
    col_left, col_right = st.columns([1.3, 1])

    # LEFT: Symptom selection
    with col_left:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("1️⃣ Select Symptoms")
        selected = st.multiselect(
            "Start typing to search symptoms:",
            options=symptom_names,
        )
        st.markdown(
            f"**Selected ({len(selected)}):** "
            + (", ".join(selected) if selected else "_None selected_")
        )
        predict = st.button("🔍 Predict Disease", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT: Prediction & PDF
    with col_right:
        st.markdown('<div class="stCard">', unsafe_allow_html=True)
        st.subheader("2️⃣ Prediction & Analysis")

        if predict:

            if not selected:
                st.warning("Please select at least one symptom.")
                st.markdown("</div>", unsafe_allow_html=True)
                return

            X_arr = build_feature_vector(selected, symptom_names)
            X = pd.DataFrame(X_arr, columns=symptom_names)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
            else:
                probs = None

            class_labels = label_encoder.classes_ if label_encoder else model.classes_

            if probs is not None:
                idx_sorted = np.argsort(probs)[::-1]
                pairs = [(class_labels[i], float(probs[i])) for i in idx_sorted]
            else:
                pred_idx = model.predict(X)[0]
                disease_name = label_encoder.inverse_transform([pred_idx])[0]
                pairs = [(disease_name, 1.0)]

            best_disease, best_prob = pairs[0]

            st.success(
                f"### 🧾 Predicted Disease: **{best_disease}**\n"
                f"**Confidence:** {best_prob:.2%}\n"
                f"**Risk Level:** {get_risk_level(best_prob)}"
            )

            info = get_disease_info(best_disease)

            st.markdown("### 🩻 Disease Overview")
            st.write(f"**Description:** {info['description']}")
            st.write(f"**Common Tests:** {info['tests']}")
            st.write(f"**Suggestions:** {info['suggestions']}")

            df_probs = pd.DataFrame(pairs[:top_n], columns=["Disease", "Probability"])
            st.markdown("### 📊 Top Predictions")
            st.dataframe(df_probs.style.format({"Probability": "{:.2%}"}))
            st.bar_chart(df_probs.set_index("Disease")["Probability"], use_container_width=True)

            st.markdown("### 🧠 Explainability")
            shap_fig = explain_prediction(model, X, symptom_names)
            shap_fig_obj = shap_fig if not isinstance(shap_fig, str) else None

            if shap_fig_obj:
                st.pyplot(shap_fig_obj)
            else:
                st.info(shap_fig)

            st.markdown("### 📄 Patient Summary")
            st.write(
                f"- Name: **{patient_name}**\n"
                f"- Age: **{age}**\n"
                f"- Gender: **{gender}**\n"
                f"- Duration: **{duration} day(s)**\n"
                f"- Symptoms: {', '.join(selected)}"
            )

            st.markdown("### 📥 Download Patient's PDF Report")

            patient_info = [
                f"Name: {patient_name}",
                f"Age: {age}",
                f"Gender: {gender}",
                f"Duration: {duration} day(s)",
                f"Symptoms: {', '.join(selected)}",
            ]
            prediction_info = [
                f"Predicted disease: {best_disease}",
                f"Model confidence: {best_prob:.2%}",
                f"Risk level: {get_risk_level(best_prob)}",
            ]

            pdf_bytes = generate_premium_pdf(
                patient_name,
                patient_info,
                prediction_info,
                info,
                pairs[:top_n],
                shap_fig_obj,
            )

            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_bytes,
                file_name=f"{patient_name}_Disease_Prediction_Report.pdf",
                mime="application/pdf",
            )

        else:
            st.info("Please select symptoms and click **Predict Disease**.")

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()