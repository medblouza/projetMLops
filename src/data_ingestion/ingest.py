import pandas as pd
import os

RAW_DATA_PATH = "data/raw/telco_churn.csv"
OUTPUT_PATH = "data/processed/churn_ingested.csv"

def main():
    # vérifier que le fichier existe
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError("Dataset not found")

    # charger les données
    df = pd.read_csv(RAW_DATA_PATH)

    # afficher infos basiques
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # créer le dossier si nécessaire
    os.makedirs("data/processed", exist_ok=True)

    # sauvegarder
    df.to_csv(OUTPUT_PATH, index=False)
    print("Data ingestion completed")

if __name__ == "__main__":
    main()
