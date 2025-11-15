"""Generate a synthetic symptoms->disease dataset.

Output CSV columns: symptom_0 ... symptom_19, label
"""
import argparse
import csv
import random
import os


DISEASES = [
    "Common Cold",
    "Flu",
    "Migraine",
    "Allergy",
    "Gastroenteritis",
    "Hypertension",
    "Diabetes",
    "Covid-19",
]


def build_disease_profiles(n_symptoms=20):
    # For each disease, choose a small subset of symptoms that are more likely
    profiles = {}
    for d in DISEASES:
        # 3-6 signature symptoms per disease
        count = random.randint(3, 6)
        profiles[d] = set(random.sample(range(n_symptoms), count))
    return profiles


def generate_sample(profiles, n_symptoms=20):
    disease = random.choice(list(profiles.keys()))
    sig = profiles[disease]
    symptoms = [0] * n_symptoms
    # signature symptoms: high probability
    for i in range(n_symptoms):
        if i in sig:
            symptoms[i] = 1 if random.random() < 0.75 else 0
        else:
            symptoms[i] = 1 if random.random() < 0.08 else 0
    return symptoms, disease


def main(out_path, n=1000, n_symptoms=20):
    profiles = build_disease_profiles(n_symptoms)
    # ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"symptom_{i}" for i in range(n_symptoms)] + ["label"]
        writer.writerow(header)
        for _ in range(n):
            symptoms, disease = generate_sample(profiles, n_symptoms)
            writer.writerow(symptoms + [disease])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples to generate")
    args = parser.parse_args()
    main(args.out, n=args.n)
