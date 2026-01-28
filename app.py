from flask import Flask, render_template, request
import pickle
import sqlite3

app = Flask(__name__)

# Load ML model
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

# Create DB table
def init_db():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            amount REAL,
            time REAL,
            result TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    amount = float(request.form["amount"])
    time = float(request.form["time"])

    prediction = model.predict([[amount, time]])

    if prediction[0] == 1:
        result = "Fraud"
    else:
        result = "Safe"

    # Save to database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO transactions (amount, time, result) VALUES (?, ?, ?)",
        (amount, time, result)
    )
    conn.commit()
    conn.close()

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
