"""
app.py  –  Fraud Detection Flask Web App (production-ready)
"""

import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_model.pkl"

# DB_PATH can be overridden via env var — useful on cloud where the working
# dir may be read-only but /tmp is always writable.
DB_PATH = Path(os.environ.get("DB_PATH", str(BASE_DIR / "database.db")))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        # Auto-train on first deploy if .pkl is missing
        print("⚙️  fraud_model.pkl missing — running train_model.py …", flush=True)
        subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], check=True)
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    model_load_error = None
except Exception as exc:
    model = None
    model_load_error = str(exc)


# ── Database helpers ──────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                amount     REAL,
                time_val   REAL,
                result     TEXT,
                confidence REAL,
                created_at TEXT
            )
        """)


init_db()


# ── Input parsing ─────────────────────────────────────────────────────────────
def parse_float(value, field_name, allow_negative=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"Invalid {field_name}: please enter a number."
    if not allow_negative and number < 0:
        return None, f"{field_name.capitalize()} must be non-negative."
    return number, None


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html", prediction=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template(
            "index.html",
            prediction=None,
            error=f"Model not available: {model_load_error}",
        )

    amount, err = parse_float(request.form.get("amount", ""), "amount")
    if err:
        return render_template("index.html", prediction=None, error=err)

    time_value, err = parse_float(request.form.get("time", ""), "time")
    if err:
        return render_template("index.html", prediction=None, error=err)

    # 30-feature vector: [Time, V1..V28, Amount]  — V1-V28 default to 0
    features = np.array([[time_value] + [0.0] * 28 + [amount]], dtype=float)

    prediction = model.predict(features)[0]
    proba      = model.predict_proba(features)[0]
    fraud_prob = float(proba[1])
    result     = "Fraud" if int(prediction) == 1 else "Safe"
    confidence = round(fraud_prob * 100 if result == "Fraud" else (1 - fraud_prob) * 100, 1)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO transactions (amount, time_val, result, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (amount, time_value, result, confidence, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence,
        fraud_prob=round(fraud_prob * 100, 1),
        error=None,
        amount=amount,
        time=time_value,
    )


@app.route("/history")
def history():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, amount, time_val, result, confidence, created_at "
            "FROM transactions ORDER BY id DESC LIMIT 20"
        ).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/health")
def health():
    """Health-check endpoint — used by Render/Railway to confirm the app is up."""
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(debug=debug)