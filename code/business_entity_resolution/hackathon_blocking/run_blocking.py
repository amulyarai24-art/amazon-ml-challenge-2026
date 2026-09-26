import pandas as pd
from src.blocking import EntityBlocker

def main():
    # Replace these sample dataframes with your actual hackathon datasets!
    data_a = {
        'id': [101, 102, 103, 104],
        'company_name': ['Acme Corp', 'Google LLC', 'Microsoft Corporation', 'Amazon Inc']
    }
    data_b = {
        'id': [201, 202, 203, 204],
        'company_name': ['Akme Corporation', 'Google', 'Micro soft Corp', 'Apple Inc']
    }

    df_left = pd.DataFrame(data_a)
    df_right = pd.DataFrame(data_b)

    # Initialize Blocker
    blocker = EntityBlocker(prefix_len=3, top_k=2)
    
    # Run Candidate Pair Generation
    candidates = blocker.generate_candidate_pairs(
        df_a=df_left, 
        df_b=df_right, 
        match_col='company_name',
        id_col_a='id',
        id_col_b='id'
    )

    # Output CSV for Person 1 (DL / Model training module)
    output_path = "candidate_pairs.csv"
    candidates.to_csv(output_path, index=False)
    print(f"[✔] Output exported successfully to '{output_path}'.")

if __name__ == "__main__":
    main()
    