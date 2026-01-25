import pandas as pd
import joblib
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = "data/processed"
MODEL_PATH = "models/churn_model.joblib"
REPORT_DIR = "reports"

def main():
    X_test = pd.read_csv(f"{DATA_DIR}/X_test_raw.csv")
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

    preprocessor = joblib.load(f"{DATA_DIR}/preprocessor.joblib")
    model = joblib.load(MODEL_PATH)

    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    os.makedirs(REPORT_DIR, exist_ok=True)

    with open(f"{REPORT_DIR}/evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Evaluation metrics saved")

if __name__ == "__main__":
    main()
