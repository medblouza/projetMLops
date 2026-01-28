from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from datetime import datetime

# ---------------- CONFIG ----------------
MONITORING_PATH = "data/monitoring/current.csv"
os.makedirs("data/monitoring", exist_ok=True)

# ---------------- APP ----------------
app = FastAPI(title="Customer Churn Prediction API")

# Charger modèle & preprocessor
model = joblib.load("models/churn_model.joblib")
preprocessor = joblib.load("data/processed/preprocessor.joblib")

# Référence schéma d'entraînement
X_REFERENCE = pd.read_csv("data/processed/X_train_raw.csv")

# ---------------- SCHEMA ----------------
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

    # 1️⃣ Construire un DataFrame valide
    input_df = X_REFERENCE.iloc[[0]].copy()
    for col, value in customer.model_dump().items():
        input_df[col] = value

    # 2️⃣ Transformation + prédiction
    X_processed = preprocessor.transform(input_df)
    proba = model.predict_proba(X_processed)[0][1]
    prediction = int(proba >= 0.4)

    # 3️⃣ Logging pour monitoring
    log_df = input_df.copy()
    log_df["prediction"] = prediction
    log_df["churn_probability"] = proba
    log_df["timestamp"] = datetime.utcnow()

    if os.path.exists(MONITORING_PATH):
        log_df.to_csv(MONITORING_PATH, mode="a", header=False, index=False)
    else:
        log_df.to_csv(MONITORING_PATH, index=False)

    # 4️⃣ Réponse API
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": round(float(proba), 3),
        "decision_threshold": 0.4
    }
