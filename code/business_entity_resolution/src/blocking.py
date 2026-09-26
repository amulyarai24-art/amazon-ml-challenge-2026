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
            name = rec.get('name_norm') or rec.get('name_normalized') or rec.get('business_name') or rec.get('company_name')
            postal = rec.get('postal') or rec.get('postal_code')
            if name: self.name_index[str(name)].append(cid)
            if postal: self.postal_index[str(postal)].append(cid)

    def retrieve_candidates(self, s1_rec: dict) -> List[str]:
        candidates: Set[str] = set()
        name = s1_rec.get('name_norm') or s1_rec.get('name_normalized') or s1_rec.get('business_name') or s1_rec.get('company_name')
        postal = s1_rec.get('postal') or s1_rec.get('postal_code')
        if name is not None: candidates.update(self.name_index.get(str(name), []))
        if postal is not None: candidates.update(self.postal_index.get(str(postal), []))
        return list(candidates)[:self.max_candidates]