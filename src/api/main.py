from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction API")

# Charger modèle & preprocessor
model = joblib.load("models/churn_model.joblib")
preprocessor = joblib.load("data/processed/preprocessor.joblib")

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

    df = pd.DataFrame([customer.model_dump()])

    X_processed = preprocessor.transform(df)
    prediction = model.predict(X_processed)[0]

    return {
        "churn_prediction": "Yes" if prediction == 1 else "No"
    }

