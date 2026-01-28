import pandas as pd

df = pd.read_csv("data/processed/X_train_raw.csv")

df.to_csv("data/reference/reference.csv", index=False)

print("Reference dataset created")
