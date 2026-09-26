from rapidfuzz import fuzz
import numpy as np

def compute_structured_features(s1_rec: dict, cand_rec: dict) -> np.ndarray:
    n1, n2 = s1_rec.get('name_norm', ''), cand_rec.get('name_norm', '')
    return np.array([fuzz.ratio(n1, n2) / 100.0, fuzz.token_sort_ratio(n1, n2) / 100.0], dtype=np.float32)