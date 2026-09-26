import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import recordlinkage

class EntityBlocker:
    """
    Hackathon Entity Matching Blocker (Person 2 Deliverable)
    Combines exact prefix blocking with top-K TF-IDF character n-gram cosine matching
    to prune non-matching record pairs efficiently.
    """
    def __init__(self, prefix_len=3, top_k=5, char_ngram_range=(2, 4)):
        self.prefix_len = prefix_len
        self.top_k = top_k
        self.ngram_range = char_ngram_range

    def _normalize_series(self, series: pd.Series) -> pd.Series:
        """Standardizes text for robust blocking."""
        return (
            series.astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", "", regex=True)
            .str.strip()
        )

    def generate_candidate_pairs(self, df_a: pd.DataFrame, df_b: pd.DataFrame, match_col: str, id_col_a: str = 'id', id_col_b: str = 'id'):
        """
        Generates candidate pairs combining deterministic and similarity-based blocking.
        Returns a Pandas DataFrame of paired record IDs.
        """
        print(f"[*] Starting candidate pair generation for {len(df_a)} x {len(df_b)} records...")
        start_time = time.time()
        if df_a.empty or df_b.empty:
            return pd.DataFrame(columns=[f"{id_col_a}_a", f"{id_col_b}_b", "left_index", "right_index"])

        # Step 1: Clean & Normalize
        norm_a = self._normalize_series(df_a[match_col])
        norm_b = self._normalize_series(df_b[match_col])

        # Step 2: Deterministic / Prefix Blocking
        print("[*] Running Prefix/Exact Blocking...")
        df_a_block = df_a.copy()
        df_b_block = df_b.copy()
        
        df_a_block['block_key'] = norm_a.str[:self.prefix_len]
        df_b_block['block_key'] = norm_b.str[:self.prefix_len]

        indexer = recordlinkage.Index()
        indexer.block(left_on='block_key', right_on='block_key')
        
        prefix_pairs = indexer.index(df_a_block, df_b_block)
        print(f"    -> Prefix blocking identified {len(prefix_pairs):,} candidate pairs.")

        # Step 3: Top-K TF-IDF Char N-Gram Cosine Nearest Neighbors
        print(f"[*] Running TF-IDF Top-{self.top_k} K-NN Blocking...")
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=self.ngram_range)
        
        tfidf_a = vectorizer.fit_transform(norm_a)
        tfidf_b = vectorizer.transform(norm_b)

        k_neighbors = min(self.top_k, tfidf_b.shape[0])
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine', algorithm='brute', n_jobs=-1)
        nn.fit(tfidf_b)
        
        _, indices = nn.kneighbors(tfidf_a)

        knn_tuple_list = []
        for idx_a, row_indices in enumerate(indices):
            for idx_b in row_indices:
                knn_tuple_list.append((idx_a, idx_b))

        knn_pairs = pd.MultiIndex.from_tuples(knn_tuple_list, names=["left_index", "right_index"])
        print(f"    -> TF-IDF KNN identified {len(knn_pairs):,} candidate pairs.")

        # Step 4: Union & Deduplicate Candidate Pairs
        all_candidate_indices = prefix_pairs.union(knn_pairs)
        print(f"[+] Total Unique Candidate Pairs: {len(all_candidate_indices):,}")

        # Map back to original dataset Record IDs
        pairs_df = pd.DataFrame({
            f'{id_col_a}_a': df_a.iloc[all_candidate_indices.get_level_values(0)][id_col_a].values,
            f'{id_col_b}_b': df_b.iloc[all_candidate_indices.get_level_values(1)][id_col_b].values,
            'left_index': all_candidate_indices.get_level_values(0),
            'right_index': all_candidate_indices.get_level_values(1)
        })

        elapsed = time.time() - start_time
        print(f"[✔] Candidate pair generation complete in {elapsed:.2f} seconds.")
        return pairs_df
    