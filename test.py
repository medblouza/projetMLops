import joblib
import pandas as pd

model = joblib.load("models/churn_model.joblib")
preprocessor = joblib.load("data/processed/preprocessor.joblib")

X_test = pd.read_csv("data/processed/X_test_raw.csv")
y_test = pd.read_csv("data/processed/y_test.csv")

X_processed = preprocessor.transform(X_test)
preds = model.predict(X_processed)

print("Unique predictions:", set(preds))
