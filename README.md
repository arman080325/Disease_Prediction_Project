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
python src/predict.py --model artifacts/model.joblib --symptoms 1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 --top 3
```

Example output:

```
Prediction: Common Cold
Top 3 probabilities:
	Common Cold: 0.362
	Allergy: 0.334
	Covid-19: 0.261
```

Notes:
- The `--symptoms` argument is a comma-separated list of binary values (0/1) matching the number of symptom features used when training the model.
- Use `--top N` to show the top N predicted diseases with probabilities.

Files

- `src/generate_dataset.py`: Creates a synthetic symptoms->disease CSV.
- `src/train.py`: Trains a RandomForest, evaluates, and saves the model.
- `src/predict.py`: Loads a saved model and predicts disease from symptoms.
- `requirements.txt`: Python dependencies.

Notes

This scaffold is intended as a starting point. Replace the synthetic generator with your real dataset (CSV with symptom columns and `label` column) and adapt preprocessing as needed.

Run the web UI (presentation-ready)

1. Install requirements (if not already):

```powershell
pip install -r requirements.txt
```

2. Start the web UI:

```powershell
python src/app.py
# OR use the helper script
.\scripts\run_app.ps1
```

Open http://127.0.0.1:5000/ in your browser. The page shows 20 symptom checkboxes and a "Predict" button; results show the top-N predicted diseases with probabilities.

Want me to add a demo PowerShell script that runs several example inputs and prints the responses? (I can add it if you like.)
