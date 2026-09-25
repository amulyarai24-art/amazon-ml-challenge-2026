from collections import defaultdict
from typing import List, Set

class MultiViewMultilingualBlocker:
    def __init__(self, max_candidates: int = 100):
        self.max_candidates = max_candidates
        self.name_index = defaultdict(list)
        self.postal_index = defaultdict(list)

    def fit_candidates(self, candidate_records: List[dict]):
        for rec in candidate_records:
            cid = rec['entity_id']
            if rec.get('name_norm'): self.name_index[rec['name_norm']].append(cid)
            if rec.get('postal'): self.postal_index[rec['postal']].append(cid)

    def retrieve_candidates(self, s1_rec: dict) -> List[str]:
        candidates: Set[str] = set()
        if s1_rec.get('name_norm') in self.name_index: candidates.update(self.name_index[s1_rec['name_norm']])
        if s1_rec.get('postal') in self.postal_index: candidates.update(self.postal_index[s1_rec['postal']])
        return list(candidates)[:self.max_candidates]