import pandas as pd
import json
import os

DATA_PATH = "data/processed/churn_ingested.csv"
REPORT_PATH = "data/processed/validation_report.json"

def main():
    df = pd.read_csv(DATA_PATH)

    report = {}

    # 1. Valeurs manquantes
    report["missing_values"] = df.isnull().sum().to_dict()

    # 2. Types des colonnes
    report["dtypes"] = df.dtypes.astype(str).to_dict()

    # 3. Distribution de la cible
    report["target_distribution"] = df["Churn"].value_counts().to_dict()

    os.makedirs("data/processed", exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print("Data validation completed")

if __name__ == "__main__":
    main()
