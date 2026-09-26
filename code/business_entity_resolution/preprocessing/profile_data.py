import os
import pandas as pd
from load_data import load_dataset, config, base_dir


def profile_dataset(name, df):
    print(f"\n{'='*50}")
    print(f"PROFILE: {name}")
    print(f"{'='*50}")

    print("Shape:", df.shape)
    print("\nDtypes:\n", df.dtypes)

    print("\nMissing values (%):")
    print((df.isnull().mean() * 100).round(2))

    print("\nDuplicate rows:", df.duplicated().sum())

    if "entity_id" in df.columns:
        print("Duplicate entity_id:", df["entity_id"].duplicated().sum())
        print("Unique entity_id count:", df["entity_id"].nunique())

    # Check likely text columns for noise
    text_cols = df.select_dtypes(include="object").columns
    for col in text_cols:
        print(f"\n--- Column: {col} ---")
        print("Unique values:", df[col].nunique())
        print("Sample values:")
        print(df[col].dropna().sample(min(5, df[col].dropna().shape[0]), random_state=1).tolist())

        # flag suspicious characters
        has_weird_chars = df[col].dropna().astype(str).str.contains(r"[^\x00-\x7F]").sum()
        print(f"Rows with non-ASCII characters: {has_weird_chars}")


if __name__ == "__main__":
    s1 = load_dataset(config["data"]["train_s1"])
    s2 = load_dataset(config["data"]["train_s2"])
    s3 = load_dataset(config["data"]["train_s3"])

    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        profile_dataset(name, df)