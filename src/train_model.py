import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_data, split_and_preprocess

DATA_PATH = "data/paysim.csv"

df = load_data(DATA_PATH)

X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df)

model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)
print('Training Started')

model.fit(X_train, y_train)
print('Training Completed')
joblib.dump(
    {"model": model, "preprocessor": preprocessor},
    "models/model.pkl"
)

print("Model saved")
