import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from typing import Dict


def _numeric_tokens(text: str) -> set[str]:
    return {tok for tok in str(text).split() if any(ch.isdigit() for ch in tok)}


def _acronym(text: str) -> str:
    tokens = [t for t in str(text).split() if t]
    return "".join(t[0] for t in tokens)


def compute_structured_features(s1_rec: Dict, cand_rec: Dict) -> np.ndarray:
    """Compute robust name/address/geographic similarity features."""
    n1 = str(s1_rec.get("name_norm", "") or "")
    n2 = str(cand_rec.get("name_norm", "") or "")
    a1 = str(s1_rec.get("address_norm", "") or "")
    a2 = str(cand_rec.get("address_norm", "") or "")
    p1 = str(s1_rec.get("postal", "") or "")
    p2 = str(cand_rec.get("postal", "") or "")
    c1 = str(s1_rec.get("country", "") or "").strip().lower()
    c2 = str(cand_rec.get("country", "") or "").strip().lower()

    name_compact_1 = n1.replace(" ", "")
    name_compact_2 = n2.replace(" ", "")
    name_tokens_1 = set(n1.split())
    name_tokens_2 = set(n2.split())
    addr_tokens_1 = set(a1.split())
    addr_tokens_2 = set(a2.split())

    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_partial = fuzz.partial_ratio(n1, n2) / 100.0
    compact_name_ratio = fuzz.ratio(name_compact_1, name_compact_2) / 100.0 if name_compact_1 and name_compact_2 else 0.0

    addr_ratio = fuzz.ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    addr_token_sort = fuzz.token_sort_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0
    addr_token_set = fuzz.token_set_ratio(a1, a2) / 100.0 if a1 and a2 else 0.0

    postal_exact = float(bool(p1 and p2 and p1 == p2))
    postal_prefix = float(bool(p1 and p2 and len(p1) >= 3 and len(p2) >= 3 and p1[:3] == p2[:3]))
    country_exact = float(bool(c1 and c2 and c1 == c2))
    country_mismatch = float(bool(c1 and c2 and c1 != c2))

    name_jaccard = len(name_tokens_1 & name_tokens_2) / max(len(name_tokens_1 | name_tokens_2), 1)
    name_overlap = len(name_tokens_1 & name_tokens_2) / max(min(len(name_tokens_1), len(name_tokens_2)), 1)
    addr_jaccard = len(addr_tokens_1 & addr_tokens_2) / max(len(addr_tokens_1 | addr_tokens_2), 1)
    numeric_1 = _numeric_tokens(a1)
    numeric_2 = _numeric_tokens(a2)
    numeric_overlap = len(numeric_1 & numeric_2) / max(len(numeric_1 | numeric_2), 1)
    exact_name = float(bool(n1 and n2 and n1 == n2))
    exact_address = float(bool(a1 and a2 and a1 == a2))
    acronym_match = float(bool(_acronym(n1) and _acronym(n2) and _acronym(n1) == _acronym(n2)))
    name_len_diff = abs(len(n1) - len(n2)) / max(len(n1), len(n2), 1)
    address_len_diff = abs(len(a1) - len(a2)) / max(len(a1), len(a2), 1)

    return np.array([
        name_ratio,
        name_token_sort,
        name_token_set,
        name_partial,
        compact_name_ratio,
        exact_name,
        acronym_match,
        name_jaccard,
        name_overlap,
        addr_ratio,
        addr_token_sort,
        addr_token_set,
        addr_jaccard,
        exact_address,
        numeric_overlap,
        postal_exact,
        postal_prefix,
        country_exact,
        country_mismatch,
        name_len_diff,
        address_len_diff,
    ], dtype=np.float32)


def train_stage1_lightgbm(X_train: np.ndarray, y_train: np.ndarray, model_save_path: str):
    """Train a precision-weighted LightGBM matcher."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": 6,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "min_child_samples": 30,
        "verbose": -1,
        "seed": 42,
        "feature_pre_filter": False,
    }

    sample_weights = np.where(y_train == 0, 3.0, 1.0)
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)

    gbm = lgb.train(params, train_data, num_boost_round=250)
    gbm.save_model(model_save_path)
    print(f"[INFO] LightGBM Stage 1 model successfully saved to {model_save_path}")
    return gbm
