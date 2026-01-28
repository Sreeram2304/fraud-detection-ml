import pickle

# Load model
with open("fraud_model.pkl", "rb") as file:
    model = pickle.load(file)

amount = float(input("Enter transaction amount: "))
time = float(input("Enter transaction time: "))

transaction = [[amount, time]]

prediction = model.predict(transaction)

if prediction[0] == 1:
    print("🚨 Fraud Transaction")
else:
    print("✅ Safe Transaction")
