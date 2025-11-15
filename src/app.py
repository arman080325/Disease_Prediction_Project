from flask import Flask, request, render_template_string
import joblib
import pandas as pd
import numpy as np
import os
try:
  # prefer absolute import when running as a script
  from symptoms import SYMPTOM_NAMES
except Exception:
  # fallback to package-relative import (works when running as module)
  from .symptoms import SYMPTOM_NAMES

APP_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Disease Predictor</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 28px; }
      .symptoms { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; max-width:720px}
      .symptom { padding:6px; }
      .result { margin-top:18px; padding:12px; border:1px solid #ddd; background:#f9f9f9 }
    </style>
  </head>
  <body>
    <h1>Disease Predictor</h1>
    <p>Toggle symptoms (1 = present). Click <strong>Predict</strong> to run the model.</p>

    <form method="post">
      <div class="symptoms">
        {% for i in range(n_features) %}
        <label class="symptom"><input type="checkbox" name="s{{i}}" {% if defaults and defaults[i] %}checked{% endif %}> {{ symptom_labels[i] }}</label>
        {% endfor %}
      </div>
      <p>
        <label>Top N results: <input type="number" name="top" value="{{top}}" min="1" max="20"></label>
      </p>
      <button type="submit">Predict</button>
    </form>

    {% if predicted %}
    <div class="result">
      <h3>Prediction: {{predicted}}</h3>
      <h4>Top {{top}} probabilities</h4>
      <ol>
        {% for c,p in probs %}
        <li>{{c}}: {{'%.3f'|format(p)}}</li>
        {% endfor %}
      </ol>
    </div>
    {% endif %}
  </body>
</html>
"""


def create_app(model_path='artifacts/model.joblib'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'Model not found at {model_path}; run `python src/train.py` first')

    model = joblib.load(model_path)

    app = Flask(__name__)

    # Determine number of features
    n_features = getattr(model, 'n_features_in_', None)
    if n_features is None:
        # attempt to infer from feature names, else default to 20
        fn = getattr(model, 'feature_names_in_', None)
        n_features = len(fn) if fn is not None else 20

    @app.route('/', methods=['GET', 'POST'])
    def index():
        predicted = None
        probs = None
        top = 5
        defaults = [False] * n_features

        if request.method == 'POST':
            top = int(request.form.get('top', 5))
            vals = []
            for i in range(n_features):
                vals.append(1 if request.form.get(f's{i}') == 'on' else 0)

            if hasattr(model, 'feature_names_in_'):
                X = pd.DataFrame([vals], columns=list(model.feature_names_in_))
            else:
                X = pd.DataFrame([vals])

            pred = model.predict(X)[0]
            predicted = pred
            if hasattr(model, 'predict_proba'):
                p = model.predict_proba(X)[0]
                classes = list(model.classes_)
                pairs = list(zip(classes, p))
                pairs.sort(key=lambda x: x[1], reverse=True)
                probs = pairs[:top]
            defaults = vals

        # choose labels: prefer user-friendly SYMPTOM_NAMES when available
        if len(SYMPTOM_NAMES) == n_features:
            symptom_labels = SYMPTOM_NAMES
        else:
            # try model feature names, then fallback to generic names
            if hasattr(model, 'feature_names_in_'):
                symptom_labels = list(model.feature_names_in_)
            else:
                symptom_labels = [f'symptom_{i}' for i in range(n_features)]

        return render_template_string(
            APP_HTML,
            n_features=n_features,
            predicted=predicted,
            probs=probs,
            top=top,
            defaults=defaults,
            symptom_labels=symptom_labels,
        )

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(host='127.0.0.1', port=5000, debug=False)
