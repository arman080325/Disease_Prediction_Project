"""Train a classifier on symptom data and save the model.

Expects CSV with columns symptom_0 ... symptom_N and `label`.
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def main(data_path, out_path):
    df = pd.read_csv(data_path)
    if 'label' not in df.columns:
        raise SystemExit('CSV must contain a `label` column')
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    joblib.dump(clf, out_path)
    print('Saved model to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to CSV dataset')
    parser.add_argument('--out', required=True, help='Output path for saved model (joblib)')
    args = parser.parse_args()
    main(args.data, args.out)
