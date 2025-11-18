"""
TRAIN.PY (FINAL & STABLE VERSION)

Ensemble model = RandomForest + ExtraTrees
→ Works on Windows + Python 3.13
→ Fully compatible with Kaggle 132-symptom dataset
→ Handles NaN values
→ No XGBoost / LightGBM / GradientBoosting needed
"""

import argparse
import os
import joblib
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)


def train_and_evaluate(data_path: str, out_path: str):
    print("📥 Loading dataset...")
    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        raise SystemExit("CSV must contain a `label` column.")

    # Replace NaN with 0 (symptom absent)
    df = df.fillna(0)

    X = df.drop(columns=["label"])
    y_text = df["label"]

    # Encode disease names → numeric labels
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    print(f"Loaded {len(df)} samples | {X.shape[1]} symptoms | {len(le.classes_)} diseases")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --------------------------
    # MODELS THAT SUPPORT NaN
    # --------------------------
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    et = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # --------------------------
    # SOFT VOTING ENSEMBLE
    # --------------------------
    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf),
            ("et", et),
        ],
        voting="soft",
        n_jobs=-1
    )

    print("\n🚀 Training Ensemble (RF + ExtraTrees)...")
    ensemble.fit(X_train, y_train)

    preds = ensemble.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"\n🎯 Ensemble Accuracy: {acc:.4f}")

    report = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(preds)
    )

    print("\n📋 Classification Report:\n")
    print(report)

    # Create directory if missing
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save model
    joblib.dump(ensemble, out_path)
    joblib.dump(le, os.path.join(os.path.dirname(out_path), "label_encoder.joblib"))

    # Save metrics
    with open(os.path.join(os.path.dirname(out_path), "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print(f"\n💾 Saved model → {out_path}")
    print(f"💾 Saved label encoder → artifacts/label_encoder.joblib")
    print(f"📝 Metrics saved → artifacts/metrics.txt")
    print("\n✅ Training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    train_and_evaluate(args.data, args.out)
