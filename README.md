# 🛡️ FraudGuard — Credit Card Fraud Detection

A Flask web app + CLI tool that detects fraudulent credit card transactions using a machine-learning pipeline trained on a realistic synthetic dataset.

## Features

- **30-feature model** — Time, V1–V28 (PCA-style), Amount (mirrors real credit card fraud datasets)
- **Balanced pipeline** — `class_weight='balanced'` + `StandardScaler` inside the sklearn Pipeline
- **Cross-validated model selection** — automatically picks the best model (Logistic Regression vs Random Forest) by ROC-AUC
- **Confidence scores** — shows fraud probability % alongside every prediction
- **Prediction history** — last 20 predictions stored in SQLite with timestamps
- **CLI predictor** — `predict.py` for quick terminal use
- **Clean UI** — responsive design with colour-coded result cards and mini confidence bars

---

## Project Structure

```
fraud-detection-ml/
├── app.py              # Flask app & routes
├── train_model.py      # Training + cross-validated model selection
├── predict.py          # CLI predictor
├── dataset.csv         # Synthetic dataset (1 500 rows, 30 features, 2% fraud)
├── requirements.txt    # Pinned dependencies
├── .gitignore
├── templates/
│   ├── index.html      # Prediction UI
│   └── history.html    # History page
└── static/
    └── style.css       # Responsive styling
```

> `fraud_model.pkl` and `database.db` are generated locally and are **not** committed to the repo.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python train_model.py
```

This will:
1. Load `dataset.csv`
2. Perform 5-fold stratified cross-validation on Logistic Regression and Random Forest
3. Select the best model by ROC-AUC
4. Evaluate it on the hold-out test set (classification report, confusion matrix, ROC-AUC)
5. Save the full pipeline (scaler + model) as `fraud_model.pkl`

---

## Run the Web App

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## CLI Prediction

```bash
python predict.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SECRET_KEY` | `dev-secret-change-me` | Flask session secret — **change in production** |
| `FLASK_DEBUG` | `""` (off) | Set to `1` to enable debug mode |

---

## Notes

- The dataset is **synthetic** and for demonstration purposes only.
- For a production model, replace `dataset.csv` with the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284 807 rows, same 30-feature structure).
- In the web UI, V1–V28 features are set to `0.0` (neutral baseline) since users only enter Amount and Time. Supply all 30 features via the API or CLI for higher accuracy.
