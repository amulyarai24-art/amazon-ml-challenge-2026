import networkx as nx
from typing import List, Tuple

class TripartiteGraphConsensus:
    def __init__(self, confidence_threshold: float = 0.72, margin_threshold: float = 0.18):
        self.threshold = confidence_threshold
        self.margin_threshold = margin_threshold

    def resolve_consensus(self, s1_id: str, candidate_scores: List[Tuple[str, float]]) -> List[str]:
        if not candidate_scores:
            return []
        sorted_cands = sorted(candidate_scores, key=lambda x: x[1], reverse=True)
        if sorted_cands[0][1] < self.threshold:
            return []
        accepted = []
        s2_sel, s3_sel = False, False
        for cid, score in sorted_cands:
            if score >= self.threshold:
                if cid.startswith("S2") and not s2_sel:
                    accepted.append(cid)
                    s2_sel = True
                elif cid.startswith("S3") and not s3_sel:
                    accepted.append(cid)
                    s3_sel = True
        return accepted