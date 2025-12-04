import joblib
import pandas as pd
import numpy as np
import os

# ---- Paths (adjust if different) ----
DATA_PATH = "data/kaggle_preprocessed.csv"
MODEL_PATH = "artifacts/model.joblib"
ENCODER_PATH = "artifacts/label_encoder.joblib"

CONF_THRESHOLD = 0.90  # 90%

def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        raise SystemExit("CSV must contain a 'label' column")

    X = df.drop(columns=["label"])
    y_text = df["label"]

    # 2. Load model + encoder
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    # 3. Predict probabilities for all rows
    if not hasattr(model, "predict_proba"):
        raise SystemExit("Model has no predict_proba")

    probs = model.predict_proba(X)  # shape: [n_samples, n_classes]
    class_labels = label_encoder.classes_

    results = []

    for i in range(len(X)):
        row_probs = probs[i]
        best_idx = int(np.argmax(row_probs))
        best_prob = float(row_probs[best_idx])
        disease = class_labels[best_idx]

        if best_prob >= CONF_THRESHOLD:
            # active symptoms = columns where value == 1
            row = X.iloc[i]
            active_symptoms = [sym for sym, val in row.items() if val == 1]

            results.append({
                "row_index": i,
                "disease": disease,
                "probability": best_prob,
                "symptoms": active_symptoms,
            })

    # Sort by probability descending
    results.sort(key=lambda r: r["probability"], reverse=True)

    # Show top 10
    print(f"Found {len(results)} high-confidence cases (>= {CONF_THRESHOLD:.0%})\n")
    for r in results[:10]:
        print(f"Row {r['row_index']}:")
        print(f"  Disease: {r['disease']}")
        print(f"  Probability: {r['probability']:.2%}")
        print(f"  Symptoms ({len(r['symptoms'])}): {', '.join(r['symptoms'])}")
        print("-" * 60)

if __name__ == "__main__":
    main()
