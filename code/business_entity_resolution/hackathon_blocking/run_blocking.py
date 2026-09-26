import os
import pandas as pd
from src.blocking import EntityBlocker

def main():
    # 1. Define paths to Person 1's cleaned datasets
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    
    s1_path = os.path.join(data_dir, "s1_clean.tsv")
    s2_path = os.path.join(data_dir, "s2_clean.tsv")

    # 2. Check if Person 1's cleaned files exist
    if not os.path.exists(s1_path) or not os.path.exists(s2_path):
        print(f"[!] Error: Could not find clean data files at '{data_dir}'.")
        print("Please run 'git pull origin main' to fetch Person 1's cleaned datasets from GitHub.")
        return

    # 3. Load Person 1's processed datasets
    print("[*] Loading cleaned datasets...")
    df_s1 = pd.read_csv(s1_path, sep="\t")
    df_s2 = pd.read_csv(s2_path, sep="\t")

    # 4. Initialize Blocker
    blocker = EntityBlocker(prefix_len=3, top_k=2)
    
    # 5. Run Candidate Pair Generation on full dataset
    print("[*] Generating candidate pairs...")
    candidates = blocker.generate_candidate_pairs(
        df_a=df_s1, 
        df_b=df_s2, 
        match_col='company_name',  # Adjust column name if Person 1 named it differently (e.g., 'name')
        id_col_a='id',
        id_col_b='id'
    )

    # 6. Export output TSV for Person 3 & Person 4
    output_path = "candidate_pairs.tsv"
    candidates.to_csv(output_path, sep="\t", index=False)
    print(f"[✓] Output exported successfully to '{output_path}'.")

if __name__ == "__main__":
    main()