from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Charger modèle & preprocessor
model = joblib.load("models/churn_model.joblib")
preprocessor = joblib.load("data/processed/preprocessor.joblib")

# 🔑 Charger une référence du schéma d'entraînement
X_REFERENCE = pd.read_csv("data/processed/X_train_raw.csv")


class CustomerInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerInput):

    input_df = X_REFERENCE.iloc[[0]].copy()

    for col, value in customer.model_dump().items():
        input_df[col] = value

    X_processed = preprocessor.transform(input_df)

    proba = model.predict_proba(X_processed)[0][1]
    prediction = 1 if proba >= 0.4 else 0

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(proba), 3),
        "decision_threshold": 0.4
    }

