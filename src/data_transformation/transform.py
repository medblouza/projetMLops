import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DATA_PATH = "data/processed/churn_ingested.csv"
OUTPUT_DIR = "data/processed"

def main():
    df = pd.read_csv(DATA_PATH)

    y = df["Churn"].map({"Yes": 1, "No": 0})
    X = df.drop(columns=["Churn", "customerID"])

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sauvegarder uniquement y + preprocessor
    y_train.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

    joblib.dump(preprocessor, f"{OUTPUT_DIR}/preprocessor.joblib")

    # Sauvegarder X bruts (OK)
    X_train.to_csv(f"{OUTPUT_DIR}/X_train_raw.csv", index=False)
    X_test.to_csv(f"{OUTPUT_DIR}/X_test_raw.csv", index=False)

    print("Data transformation completed")

if __name__ == "__main__":
    main()
