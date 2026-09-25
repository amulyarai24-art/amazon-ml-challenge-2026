import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from typing import Dict, List, Tuple

def compute_structured_features(s1_rec: Dict, cand_rec: Dict) -> np.ndarray:
    """
    Computes a 10-dimensional feature vector comparing two business records.
    """
    n1 = str(s1_rec.get("name_norm", "") or "")
    n2 = str(cand_rec.get("name_norm", "") or "")
    
    a1 = str(s1_rec.get("address_norm", "") or "")
    a2 = str(cand_rec.get("address_norm", "") or "")
    
    p1 = str(s1_rec.get("postal", "") or "")
    p2 = str(cand_rec.get("postal", "") or "")

    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    
    addr_ratio = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    addr_token_sort = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0

    postal_exact = 1.0 if (p1 and p2 and p1 == p2) else 0.0
    postal_prefix = 1.0 if (p1 and p2 and len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3]) else 0.0
    
    t1 = set(n1.split())
    t2 = set(n2.split())
    jaccard = len(t1.intersection(t2)) / max(len(t1.union(t2)), 1)
    len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)

    return np.array([
        name_ratio,
        name_token_sort,
        name_token_set,
        name_partial,
        addr_ratio,
        addr_token_sort,
        postal_exact,
        postal_prefix,
        jaccard,
        len_diff
    ], dtype=np.float32)

def train_stage1_lightgbm(X_train: np.ndarray, y_train: np.ndarray, model_save_path: str):
    """
    Trains a precision-weighted LightGBM Gradient Boosted Decision Tree model.
    """
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'verbose': -1
    }
    
    # Negative samples receive 2x weight to penalize false positives for F0.5 metric
    sample_weights = np.where(y_train == 0, 2.0, 1.0)
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    
    gbm = lgb.train(
        params,
        train_data,
        num_boost_round=150
    )
    
    gbm.save_model(model_save_path)
    print(f"[INFO] LightGBM Stage 1 model successfully saved to {model_save_path}")
    return gbm