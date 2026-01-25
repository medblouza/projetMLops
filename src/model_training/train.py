import pandas as pd
import joblib
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = "data/processed"
MODEL_DIR = "models"

def main():
    X_train = pd.read_csv(f"{DATA_DIR}/X_train_raw.csv")
    X_test = pd.read_csv(f"{DATA_DIR}/X_test_raw.csv")
    y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").values.ravel()
    y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").values.ravel()

    preprocessor = joblib.load(f"{DATA_DIR}/preprocessor.joblib")

    X_train_p = preprocessor.fit_transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    with mlflow.start_run():

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_p, y_train)

        y_pred = model.predict(X_test_p)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log params
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("MLflow run completed")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, f"{MODEL_DIR}/churn_model.joblib")

if __name__ == "__main__":
    main()
