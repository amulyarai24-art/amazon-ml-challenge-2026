import math
import json
from typing import List, Dict, Tuple
from collections import defaultdict

class FellegiSunterEvidenceEngine:
    """
    Probabilistic Fellegi-Sunter Linkage Engine.
    Computes log-likelihood ratios w_i = log2(m_i / u_i) where:
      m_i = P(Agreement_i | Match)
      u_i = P(Agreement_i | Non-Match)
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.m_probs: Dict[str, float] = {}
        self.u_probs: Dict[str, float] = {}
        self.weights: Dict[str, float] = {}

    def fit(self, positive_pairs: List[Dict], negative_pairs: List[Dict]):
        """
        Calculates m_i and u_i probabilities from training ground truth pairs.
        """
        pos_counts = defaultdict(int)
        n_pos = max(len(positive_pairs), 1)
        
        for pair in positive_pairs:
            for feature_key, agreed in pair.get("agreements", {}).items():
                if agreed:
                    pos_counts[feature_key] += 1
                    
        for feat, count in pos_counts.items():
            self.m_probs[feat] = min(max(count / n_pos, self.eps), 1.0 - self.eps)

        neg_counts = defaultdict(int)
        n_neg = max(len(negative_pairs), 1)
        
        for pair in negative_pairs:
            for feature_key, agreed in pair.get("agreements", {}).items():
                if agreed:
                    neg_counts[feature_key] += 1
                    
        for feat, count in neg_counts.items():
            self.u_probs[feat] = min(max(count / n_neg, self.eps), 1.0 - self.eps)

        self._compute_weights()

    def _compute_weights(self):
        all_features = set(self.m_probs.keys()).union(set(self.u_probs.keys()))
        for feat in all_features:
            m = self.m_probs.get(feat, 0.90)
            u = self.u_probs.get(feat, 0.01)
            # Log2 evidence weight
            self.weights[feat] = math.log2(m / u)

    def score_pair(self, agreements: Dict[str, bool]) -> float:
        score = 0.0
        for feat, agreed in agreements.items():
            if agreed and feat in self.weights:
                score += self.weights[feat]
            elif not agreed and feat in self.weights:
                # Disagreement weight penalty
                m = self.m_probs.get(feat, 0.90)
                u = self.u_probs.get(feat, 0.01)
                disagree_weight = math.log2((1.0 - m) / (1.0 - u))
                score += disagree_weight
        return score

    def save_weights(self, filepath: str):
        data = {
            "m_probs": self.m_probs,
            "u_probs": self.u_probs,
            "weights": self.weights
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_weights(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.m_probs = data.get("m_probs", {})
        self.u_probs = data.get("u_probs", {})
        self.weights = data.get("weights", {})