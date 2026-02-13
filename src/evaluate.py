import joblib
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import load_data, split_and_preprocess

bundle = joblib.load("models/model.pkl")
model = bundle["model"]

df = load_data("data/paysim.csv")
_, X_test, _, y_test, _ = split_and_preprocess(df)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
