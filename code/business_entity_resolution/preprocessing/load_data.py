import os
import pandas as pd
import yaml

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
config_path = os.path.join(base_dir, "configs", "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)


def load_dataset(relative_path):
    """Load a TSV file, tolerating bad rows and encoding issues."""
    full_path = os.path.join(base_dir, relative_path)

    for encoding in ("utf-8", "latin1"):
        try:
            df = pd.read_csv(
                full_path,
                sep="\t",
                engine="python",
                on_bad_lines="skip",
                encoding=encoding,
            )
            return df
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not read {full_path} with utf-8 or latin1 encoding")


def profile(name, df):
    print(f"\n--- {name} ---")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Missing values:\n", df.isnull().sum())
    print(df.head(3))


if __name__ == "__main__":
    s1 = load_dataset(config["data"]["train_s1"])
    s2 = load_dataset(config["data"]["train_s2"])
    s3 = load_dataset(config["data"]["train_s3"])

    for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        profile(name, df)