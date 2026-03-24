"""
predict.py  –  CLI predictor for the fraud-detection model.
Usage:
    python predict.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "fraud_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        print("❌  fraud_model.pkl not found. Run train_model.py first.")
        sys.exit(1)
    return joblib.load(MODEL_PATH)


def get_float(prompt, allow_negative=False):
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("   ⚠  Please enter a valid number.")
            continue
        if not allow_negative and value < 0:
            print("   ⚠  Value must be non-negative.")
            continue
        return value


def main():
    print("=" * 50)
    print("  Fraud Detection — CLI Predictor")
    print("=" * 50)
    print("  Enter transaction details below.")
    print("  (V1–V28 PCA features default to 0 if unknown)\n")

    model = load_model()

    amount     = get_float("Transaction amount ($):  ")
    time_value = get_float("Time since first transaction (seconds):  ")

    # Build 30-feature vector: [Time, V1..V28, Amount]
    feature_values = [time_value] + [0.0] * 28 + [amount]
    features = np.array([feature_values], dtype=float)

    prediction = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    fraud_prob = float(proba[1])
    result     = "Fraud" if int(prediction) == 1 else "Safe"
    confidence = fraud_prob * 100 if result == "Fraud" else (1 - fraud_prob) * 100

    print("\n" + "─" * 50)
    icon = "🚨" if result == "Fraud" else "✅"
    print(f"  {icon}  Result:     {result}")
    print(f"      Confidence: {confidence:.1f}%")
    print(f"      Fraud probability: {fraud_prob*100:.1f}%")
    print("─" * 50 + "\n")


if __name__ == "__main__":
    main()
