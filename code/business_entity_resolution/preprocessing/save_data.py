import os
from load_data import load_dataset, config, base_dir
from clean_data import clean_dataframe

output_dir = os.path.join(base_dir, "data", "processed")
os.makedirs(output_dir, exist_ok=True)

if __name__ == "__main__":
    for split, key in [("s1", "train_s1"), ("s2", "train_s2"), ("s3", "train_s3")]:
        print(f"\nProcessing {split}...")
        df = load_dataset(config["data"][key])
        df_clean = clean_dataframe(df)

        out_path = os.path.join(output_dir, f"{split}_clean.tsv")
        df_clean.to_csv(out_path, sep="\t", index=False)
        print(f"Saved {split} -> {out_path} | shape: {df_clean.shape} | columns: {list(df_clean.columns)}")