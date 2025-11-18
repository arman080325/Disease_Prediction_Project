"""Generate a synthetic symptoms->disease dataset.

Output CSV columns: symptom_0 ... symptom_(N-1), label

- Uses symptoms.SYMPTOM_NAMES to define how many features we have.
- Each disease gets its own "signature" symptoms that are much more likely
  to be present when that disease is the label.
- Non-signature symptoms appear as low-probability background noise.

This keeps the dataset synthetic (no real patient data) but structured
enough that the ML model can learn clear mappings and reach high accuracy.
"""

import argparse
import csv
import os
import random

from symptoms import SYMPTOM_NAMES

# You can extend or rename diseases as you like.
DISEASES = [
    "Common Cold",
    "Seasonal Flu",
    "Migraine",
    "Allergic Rhinitis",
    "Gastroenteritis",
    "Hypertension",
    "Type 2 Diabetes",
    "Covid-19",
    "Anxiety Disorder",
    "Depressive Disorder",
    "Asthma",
    "Pneumonia",
]

N_SYMPTOMS = len(SYMPTOM_NAMES)


def build_disease_profiles(n_symptoms: int = N_SYMPTOMS):
    """
    For each disease, choose a subset of 'signature' symptoms that are
    much more likely to be present when that disease is the label.

    Returns:
        dict[disease] -> set(indices of signature symptoms)
    """
    profiles = {}
    for disease in DISEASES:
        # 5–10 signature symptoms per disease (tune if you like)
        count = random.randint(5, 10)
        profiles[disease] = set(random.sample(range(n_symptoms), count))
    return profiles


def generate_sample(profiles, n_symptoms: int = N_SYMPTOMS):
    """
    Generate one synthetic patient record:

    - Pick a random disease.
    - For that disease's signature symptoms: set to 1 with high probability.
    - For all other symptoms: set to 1 with low background probability.
    """
    disease = random.choice(list(profiles.keys()))
    signature = profiles[disease]
    symptoms = [0] * n_symptoms

    for i in range(n_symptoms):
        if i in signature:
            # Signature symptoms: very likely to be present
            symptoms[i] = 1 if random.random() < 0.80 else 0
        else:
            # Non-signature symptoms: low chance of appearing
            symptoms[i] = 1 if random.random() < 0.05 else 0

    return symptoms, disease


def main(out_path: str, n: int = 3000, seed: int = 42):
    random.seed(seed)
    profiles = build_disease_profiles(N_SYMPTOMS)

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"symptom_{i}" for i in range(N_SYMPTOMS)] + ["label"]
        writer.writerow(header)

        for _ in range(n):
            symptoms, disease = generate_sample(profiles, N_SYMPTOMS)
            writer.writerow(symptoms + [disease])

    print(f"✅ Generated {n} samples with {N_SYMPTOMS} symptoms -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--n", type=int, default=3000, help="Number of samples to generate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    main(args.out, n=args.n, seed=args.seed)
