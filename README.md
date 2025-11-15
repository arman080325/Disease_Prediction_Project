# Disease Prediction from Symptoms

This is a small, self-contained example project that demonstrates how to train a machine learning model to predict diseases from symptom presence (binary features).

Quick start

1. Create a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Generate a synthetic dataset and train a model:

```powershell
python src/generate_dataset.py --out data/synthetic.csv --n 2000
python src/train.py --data data/synthetic.csv --out artifacts/model.joblib
```

3. Run an example prediction:

```powershell
python src/predict.py --model artifacts/model.joblib --symptoms 1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

Files

- `src/generate_dataset.py`: Creates a synthetic symptoms->disease CSV.
- `src/train.py`: Trains a RandomForest, evaluates, and saves the model.
- `src/predict.py`: Loads a saved model and predicts disease from symptoms.
- `requirements.txt`: Python dependencies.

Notes

This scaffold is intended as a starting point. Replace the synthetic generator with your real dataset (CSV with symptom columns and `label` column) and adapt preprocessing as needed.
