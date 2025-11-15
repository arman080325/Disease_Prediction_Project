"""Load a saved model and predict disease from a symptoms vector.

This script accepts a comma-separated symptom vector (e.g. `1,0,0,...`) and
prints a top-N ranked list of predicted diseases with probabilities.

Usage example:
    python src/predict.py --model artifacts/model.joblib --symptoms 1,0,0,1,0,... --top 3
"""
import argparse
import joblib
import pandas as pd
import numpy as np
import sys


def parse_symptoms(symptoms_str):
    # accept a comma-separated string of integers
    try:
        vals = [int(x) for x in symptoms_str.split(',') if x.strip() != '']
    except ValueError:
        raise ValueError('Symptoms must be integers (0 or 1) separated by commas')
    return vals


def main(model_path, symptoms_str, top_n=5):
    model = joblib.load(model_path)

    vals = parse_symptoms(symptoms_str)

    # determine expected number of features from the trained model
    n_expected = None
    if hasattr(model, 'n_features_in_'):
        n_expected = int(model.n_features_in_)
    elif hasattr(model, 'feature_names_in_'):
        n_expected = len(model.feature_names_in_)

    if n_expected is not None and len(vals) != n_expected:
        raise SystemExit(f'Expected {n_expected} features, got {len(vals)}')

    # build DataFrame with feature names when available to avoid sklearn warnings
    if hasattr(model, 'feature_names_in_'):
        cols = list(model.feature_names_in_)
        X = pd.DataFrame([vals], columns=cols)
    else:
        X = pd.DataFrame([vals])

    preds = model.predict(X)
    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)

    print(f'Prediction: {preds[0]}')
    if probs is not None:
        classes = list(model.classes_)
        prob_pairs = list(zip(classes, probs[0]))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)
        print(f'Top {top_n} probabilities:')
        for c, p in prob_pairs[:top_n]:
            print(f'  {c}: {p:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict disease from symptom vector')
    parser.add_argument('--model', required=True, help='Path to saved joblib model')
    parser.add_argument('--symptoms', required=True, help='Comma-separated binary values for symptoms')
    parser.add_argument('--top', type=int, default=5, help='How many top probabilities to show')
    args = parser.parse_args()
    try:
        main(args.model, args.symptoms, top_n=args.top)
    except Exception as e:
        print('Error:', e, file=sys.stderr)
        sys.exit(2)
