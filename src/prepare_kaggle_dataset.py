import pandas as pd
import json

RAW_PATH = "data/kaggle_raw.csv"
OUT_PATH = "data/kaggle_preprocessed.csv"
SYMPTOMS_JSON = "data/symptoms.json"


def main():
    print("Loading Kaggle dataset...")
    df = pd.read_csv(RAW_PATH)

    print("\nColumns found in dataset:")
    print(df.columns)

    # Identify disease column
    if "prognosis" in df.columns:
        target_col = "prognosis"
    elif "Disease" in df.columns:
        target_col = "Disease"
    else:
        raise Exception("❌ Dataset must contain 'prognosis' or 'Disease' column.")

    # All other columns = symptoms
    symptom_cols = [c for c in df.columns if c != target_col]

    print("\nNumber of symptom columns:", len(symptom_cols))

    # Save symptoms list for the app
    with open(SYMPTOMS_JSON, "w", encoding="utf-8") as f:
        json.dump(symptom_cols, f)
    print(f"✅ Saved symptom list → {SYMPTOMS_JSON}")

    # Rename label column to `label` and save clean dataset
    df = df.rename(columns={target_col: "label"})
    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved clean dataset → {OUT_PATH}")


if __name__ == "__main__":
    main()
