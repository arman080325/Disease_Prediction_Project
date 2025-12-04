"""
TRAIN.PY — FINAL & STABLE VERSION
----------------------------------

✔ Ensemble Model = RandomForest + ExtraTrees (Soft Voting)
✔ Works on Windows + Python 3.13
✔ Fully compatible with Kaggle 132-symptom dataset
✔ Handles NaN values
✔ Saves classification report + accuracy + model + encoder
✔ Generates ROC–AUC curve for all 41 diseases

Author: Arman Ahemad Khan
"""

import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    auc
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier


# --------------------------------------------------------------
# TRAIN + EVALUATE MODEL
# --------------------------------------------------------------
def train_and_evaluate(data_path: str, out_path: str):
    print("📥 Loading dataset...")
    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        raise SystemExit("❌ ERROR: CSV must contain a `label` column.")

    # Fill missing symptoms with 0
    df = df.fillna(0)

    X = df.drop(columns=["label"])
    y_text = df["label"]

    # Encode disease names → numbers
    le = LabelEncoder()
    y = le.fit_transform(y_text)

    print(f"\nLoaded {len(df)} samples | {X.shape[1]} symptoms | {len(le.classes_)} diseases\n")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ----------------------------------------------------------
    # BASE MODELS
    # ----------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    et = ExtraTreesClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # ----------------------------------------------------------
    # ENSEMBLE (SOFT VOTING)
    # ----------------------------------------------------------
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et)],
        voting="soft",
        n_jobs=-1
    )

    print("🚀 Training Ensemble Model (RF + ExtraTrees)...")
    ensemble.fit(X_train, y_train)

    # Predictions
    preds = ensemble.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, preds)
    print(f"\n🎯 Final Ensemble Accuracy: {acc:.4f}")

    # Classification Report
    report = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(preds)
    )
    print("\n📋 Classification Report:\n")
    print(report)

    # ----------------------------------------------------------
    # SAVE ARTIFACTS
    # ----------------------------------------------------------
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    joblib.dump(ensemble, out_path)
    joblib.dump(le, os.path.join(os.path.dirname(out_path), "label_encoder.joblib"))

    with open(os.path.join(os.path.dirname(out_path), "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print(f"\n💾 Saved Model → {out_path}")
    print("💾 Saved Label Encoder → artifacts/label_encoder.joblib")
    print("📝 Metrics Saved → artifacts/metrics.txt")

    # ----------------------------------------------------------
    # GENERATE ROC–AUC CURVE (MULTI-CLASS)
    # ----------------------------------------------------------
    print("\n📊 Generating ROC–AUC Curve…")

    y_test_bin = label_binarize(y_test, classes=list(range(len(le.classes_))))
    probs = ensemble.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(le.classes_)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot macro-average ROC
    plt.figure(figsize=(8, 6))
    for i in range(len(le.classes_)):
        plt.plot(fpr[i], tpr[i], alpha=0.2)

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC–AUC Curve (Multi-Class Ensemble)")
    plt.grid(True)

    roc_path = os.path.join(os.path.dirname(out_path), "roc_auc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print(f"📈 ROC Curve Saved → {roc_path}")
    print("\n✅ Training Completed Successfully.")


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    train_and_evaluate(args.data, args.out)
