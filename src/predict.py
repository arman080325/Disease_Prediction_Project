"""Load a saved model and predict disease from symptoms vector.

Usage examples:
python src/predict.py --model artifacts/model.joblib --symptoms 1,0,0,1,0,...
"""
import argparse
import joblib
import numpy as np


def main(model_path, symptoms_str):
    model = joblib.load(model_path)
    vals = [int(x) for x in symptoms_str.split(',')]
    arr = np.array(vals).reshape(1, -1)
    preds = model.predict(arr)
    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(arr)
    print('Prediction:', preds[0])
    if probs is not None:
        # show top probabilities
        classes = model.classes_
        prob_pairs = list(zip(classes, probs[0]))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)
        print('Top probabilities:')
        for c, p in prob_pairs[:5]:
            print(f'  {c}: {p:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--symptoms', required=True, help='Comma-separated binary values for symptoms')
    args = parser.parse_args()
    main(args.model, args.symptoms)
