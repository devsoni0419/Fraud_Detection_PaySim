import joblib
import pandas as pd

bundle = joblib.load("models/model.pkl")
model = bundle["model"]
preprocessor = bundle["preprocessor"]

sample = pd.DataFrame([{
    "step": 10,
    "type": "TRANSFER",
    "amount": 850000,
    "oldbalanceOrg": 900000,
    "newbalanceOrig": 50000,
    "oldbalanceDest": 0,
    "newbalanceDest": 0
}])

X = preprocessor.transform(sample)
prob = model.predict_proba(X)[0][1]

print("Fraud Probability:", prob)
