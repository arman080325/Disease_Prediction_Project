"""
Human-friendly symptom names used by the UI.

IMPORTANT:
- The order of SYMPTOM_NAMES MUST match the feature order used
  when generating the dataset (symptom_0 ... symptom_N-1).
- generate_dataset.py uses this list length to decide how many
  symptom columns to create in data/synthetic.csv.
"""

SYMPTOM_NAMES = [
    "fever",
    "chills",
    "cough",
    "sore throat",
    "runny / blocked nose",
    "sneezing",
    "shortness of breath",
    "chest pain or tightness",
    "fatigue / tiredness",
    "general weakness",
    "headache",
    "dizziness or light-headedness",
    "nausea",
    "vomiting",
    "diarrhea",
    "abdominal / stomach pain",
    "loss of appetite",
    "muscle pain",
    "joint pain",
    "back pain",
    "skin rash",
    "itching",
    "red or irritated eyes",
    "blurred vision",
    "increased thirst",
    "frequent urination",
    "unintentional weight loss",
    "swelling of legs / ankles",
    "heart palpitations",
    "high blood pressure (known)",
    "low blood pressure (known)",
    "night sweats",
    "difficulty sleeping (insomnia)",
    "feeling anxious / restless",
    "low mood / feeling depressed",
    "confusion or disorientation",
    "difficulty speaking clearly",
    "numbness or tingling in limbs",
    "burning sensation in limbs",
    "loss of taste or smell",
]
