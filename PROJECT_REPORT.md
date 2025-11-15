# Project Report — Disease Prediction from Symptoms

**Project summary:**
- **Goal:** Demonstrate a small end-to-end pipeline that generates a synthetic symptoms->disease dataset, trains a classifier, and exposes prediction via CLI and a small web UI for presentation/demos.
- **Scope:** Synthetic dataset (20 binary symptom features); RandomForest classifier; command-line and Flask UI for predictions.

**Methods & components used**
- **Synthetic data generation (`src/generate_dataset.py`):**
  - Creates a dataset with 20 binary symptom columns (`symptom_0` ... `symptom_19`) and a `label` column for disease.
  - For each disease, the generator samples 3–6 signature symptoms that have higher probability of being present; other symptoms are sampled with a low background probability. This makes the dataset structured but synthetic.
  - File-writing now ensures the output directory exists before saving the CSV.

- **Model training (`src/train.py`):**
  - Uses scikit-learn's `RandomForestClassifier` (200 trees, random_state=42).
  - Uses `train_test_split` with stratification and `random_state=42` for reproducible splits.
  - Evaluation: prints `accuracy_score` and `classification_report` (precision/recall/F1 per class).
  - The trained model is serialized with `joblib.dump` to `artifacts/model.joblib`.

- **Prediction CLI (`src/predict.py`):**
  - Loads a saved joblib model and accepts a comma-separated symptom vector via `--symptoms`.
  - Validates symptom length against the trained model's expected feature count (`n_features_in_` or `feature_names_in_`).
  - Uses a pandas DataFrame with column names when available to avoid sklearn warnings.
  - Adds `--top N` to display the top-N predicted classes with probabilities.
  - Improved error handling and clearer output formatting for presentation.

- **Presentation Web UI (`src/app.py` + `src/symptoms.py`):**
  - A minimal Flask app renders a checkbox form for the 20 symptoms and a `Predict` button.
  - Uses `src/symptoms.py` for human-friendly symptom labels (default list of 20 names) and falls back to model feature names or `symptom_i` if counts mismatch.
  - Displays the top-N predicted diseases with probabilities in the browser.
  - A small helper script `scripts/run_app.ps1` was added to run the app easily on Windows.

- **Dependencies:**
  - `pandas`, `scikit-learn`, `joblib`, `python-docx` (existing), plus `flask` (added) — see `requirements.txt`.

**Key results (example runs performed during development)**
- Synthetic dataset: `data/synthetic.csv` (2,000 rows in the example).
- Training output (example): overall accuracy ~ 0.8825 and a detailed classification report across eight synthetic disease classes.
- Prediction examples (CLI):
  - Example: `python src/predict.py --model artifacts/model.joblib --symptoms 1,0,0,1,... --top 3` produced a top prediction and three probabilities.
  - All-zero and all-one symptom vectors were tested to demonstrate behavior for extreme inputs — outputs reflect the synthetic data generator's learned mappings.

**How to run (quick)**
1. Create and activate a virtual environment (PowerShell):
   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Generate data and train:
   ```powershell
   python src/generate_dataset.py --out data/synthetic.csv --n 2000
   python src/train.py --data data/synthetic.csv --out artifacts/model.joblib
   ```
3. Try CLI prediction:
   ```powershell
   python src/predict.py --model artifacts/model.joblib --symptoms 1,0,0,1,0,... --top 3
   ```
4. Run the web demo:
   ```powershell
   python src/app.py
   # open http://127.0.0.1:5000/
   ```

**Limitations & caveats**
- The dataset is synthetic and randomly generated; predictions are valid only relative to the synthetic mapping and do not reflect real clinical relationships.
- Model persistence across scikit-learn versions can raise InconsistentVersionWarning when unpickling models trained under a different scikit-learn micro-version. Retrain in the target environment for production.
- No authentication, data validation beyond simple length checks, or privacy protections are implemented — the web UI is for demo/presentation only.

**Possible next steps / enhancements**
- Replace synthetic generator with a real dataset: map real symptom names to columns and retrain for realistic predictions.
- Add a small explainability page showing `feature_importances_` for the RandomForest (top symptoms per disease).
- Add batch prediction from file and an example `scripts/demo_predict.ps1` that runs a few demo inputs for a presentation.
- Add unit tests and a simple CI job to verify training and prediction steps.

**Files added/modified during presentation prep**
- `src/generate_dataset.py` — ensure output dir exists before writing.
- `src/predict.py` — improved CLI, input validation, `--top` option.
- `src/app.py` — new Flask UI for interactive demo.
- `src/symptoms.py` — human-friendly symptom labels.
- `requirements.txt` — added `flask`.
- `README.md` — updated with web UI instructions and example outputs.

If you want, I can now:
- produce a one-page slide (PNG) or screenshot of the web UI for the presentation; or
- add a `scripts/demo_predict.ps1` that runs example inputs and prints results for an automated demo.

End of report.
