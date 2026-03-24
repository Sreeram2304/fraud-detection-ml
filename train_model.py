"""
train_model.py
Trains a fraud-detection pipeline with:
  - StandardScaler for feature normalisation
  - class_weight='balanced' to handle class imbalance
  - Cross-validated model selection (Logistic Regression vs Random Forest)
  - Full evaluation report: accuracy, precision, recall, F1, ROC-AUC
  - Saves the best pipeline as fraud_model.pkl via joblib
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "fraud_model.pkl"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"dataset.csv not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "Class" not in df.columns:
        raise ValueError("dataset.csv must contain a 'Class' column (0=Safe, 1=Fraud).")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    print(f"Dataset loaded: {len(df)} rows | Features: {X.shape[1]} | Fraud: {y.sum()} ({y.mean()*100:.1f}%)")
    return X, y


def build_candidates():
    """Return a dict of named sklearn Pipeline candidates."""
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                max_depth=10,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }


def select_best_model(candidates, X_train, y_train):
    """Cross-validate all candidates; return the name & pipeline with best ROC-AUC."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_name, best_pipeline, best_score = None, None, -1

    print("\n── Cross-Validation (5-fold ROC-AUC) ───────────────────────────")
    for name, pipeline in candidates.items():
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        mean_score = scores.mean()
        print(f"  {name:<25} ROC-AUC: {mean_score:.4f}  (±{scores.std():.4f})")
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = pipeline

    print(f"\n  ✅ Best model: {best_name}  (ROC-AUC={best_score:.4f})")
    return best_name, best_pipeline


def evaluate(pipeline, X_test, y_test):
    """Print a full evaluation report on the hold-out test set."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n── Hold-out Test Evaluation ─────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"  Confusion Matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"  ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  F1 (fraud class): {f1_score(y_test, y_pred):.4f}")


def main():
    print("=" * 60)
    print("  Fraud Detection — Model Training")
    print("=" * 60)

    X, y = load_data()

    # Stratified split preserves the fraud ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    candidates = build_candidates()
    best_name, best_pipeline = select_best_model(candidates, X_train, y_train)

    # Fit winner on full training set
    best_pipeline.fit(X_train, y_train)
    evaluate(best_pipeline, X_test, y_test)

    # Persist the pipeline (scaler + model baked in)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")
    print("   (Pipeline includes StandardScaler — no separate scaler needed at inference)")


if __name__ == "__main__":
    main()
