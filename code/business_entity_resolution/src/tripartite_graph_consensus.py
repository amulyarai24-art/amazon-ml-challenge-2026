"""
tripartite_graph_consensus.py

Owned by: Person 4 (Graph Consensus, Calibration & Submission QA Lead)

Resolves cross-source (S1 <-> S2 <-> S3) matching contradictions using a
maximum-weight clique consensus over the candidate graph, applies Isotonic
Regression probability calibration to raw matcher scores, and enforces a
margin-based confidence gate before committing to any match (singleton
protection).

Integration contract with the rest of the pipeline
---------------------------------------------------
- Candidate generation (blocking) produces, per Source-1 entity, a list of
  candidate Source-2 / Source-3 ids (candidate_pairs.tsv).
- The matching model scores each S1-candidate pair and produces a RAW
  (uncalibrated) similarity/probability score for that pair.
- This module turns those raw scores into a final, precision-safe decision:

      raw scores --> calibrate_probabilities() --> resolve_consensus() --> final ids
"""

from __future__ import annotations

import itertools
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scikit-learn is required for isotonic calibration. "
        "Install with `pip install scikit-learn`."
    ) from exc


# --------------------------------------------------------------------------- #
# Probability calibration (Isotonic Regression)
# --------------------------------------------------------------------------- #

DEFAULT_CALIBRATOR_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "isotonic_calibrator.pkl"
)


class ProbabilityCalibrator:
    """Wraps a fitted Isotonic Regression model that maps a raw matcher
    score -> a calibrated probability that the pair is a true match.

    Fit once on the training set (raw score, is_true_match) pairs, then
    reused (pickled) at inference time on the test set.
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._fitted = False

    def fit(self, raw_scores: Iterable[float], labels: Iterable[int]) -> "ProbabilityCalibrator":
        raw_scores_arr = np.asarray(list(raw_scores), dtype=float)
        labels_arr = np.asarray(list(labels), dtype=float)
        if raw_scores_arr.shape[0] != labels_arr.shape[0]:
            raise ValueError("raw_scores and labels must be the same length")
        self._model.fit(raw_scores_arr, labels_arr)
        self._fitted = True
        return self

    def transform(self, raw_scores: Iterable[float]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ProbabilityCalibrator has not been fit or loaded yet.")
        raw_scores_arr = np.asarray(list(raw_scores), dtype=float)
        return self._model.predict(raw_scores_arr)

    def save(self, path: Path = DEFAULT_CALIBRATOR_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self._model, fh)

    @classmethod
    def load(cls, path: Path = DEFAULT_CALIBRATOR_PATH) -> "ProbabilityCalibrator":
        instance = cls()
        with open(path, "rb") as fh:
            instance._model = pickle.load(fh)
        instance._fitted = True
        return instance


_calibrator_singleton: Optional[ProbabilityCalibrator] = None


def calibrate_probabilities(raw_scores: np.ndarray) -> np.ndarray:
    """Convert raw matcher scores into calibrated match probabilities.

    Loads the fitted Isotonic Regression calibrator (trained offline on the
    training split via fit_calibrator_from_training_data) the first time
    it's called, then reuses it for the rest of the process.
    """
    global _calibrator_singleton
    if _calibrator_singleton is None:
        _calibrator_singleton = ProbabilityCalibrator.load()
    raw_scores_arr = np.asarray(raw_scores, dtype=float)
    return _calibrator_singleton.transform(raw_scores_arr)


def fit_calibrator_from_training_data(
    raw_scores: Iterable[float],
    labels: Iterable[int],
    save_path: Path = DEFAULT_CALIBRATOR_PATH,
) -> ProbabilityCalibrator:
    """One-off training-time helper: fit the isotonic calibrator on the
    labelled training candidate pairs (raw_score, is_true_match) and persist
    it so calibrate_probabilities can load it during test-time inference.
    """
    calibrator = ProbabilityCalibrator().fit(raw_scores, labels)
    calibrator.save(save_path)
    return calibrator


# --------------------------------------------------------------------------- #
# Tripartite graph consensus
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    entity_id: str
    source: str
    score: float


class TripartiteGraphConsensus:
    """Resolves S1 <-> S2 <-> S3 cycle contradictions with a max-weight
    clique consensus and a margin-based confidence gate.
    """

    def __init__(
        self,
        cross_source_scores: Optional[Dict[FrozenSet[str], float]] = None,
        consistency_threshold: float = 0.3,
        default_consistency: float = 0.5,
        margin_threshold: float = 0.18,
        min_score_threshold: float = 0.5,
        max_exact_clique_size: int = 18,
    ) -> None:
        self.cross_source_scores = cross_source_scores or {}
        self.consistency_threshold = consistency_threshold
        self.default_consistency = default_consistency
        self.margin_threshold = margin_threshold
        self.min_score_threshold = min_score_threshold
        self.max_exact_clique_size = max_exact_clique_size

    def resolve_consensus(self, s1_id: str, candidate_scores: List[dict]) -> List[str]:
        if not candidate_scores:
            return []

        candidates = [
            Candidate(
                entity_id=c["entity_id"],
                source=self._source_of(c["entity_id"]),
                score=float(c["score"]),
            )
            for c in candidate_scores
        ]

        candidates = [c for c in candidates if c.score >= self.min_score_threshold]
        if not candidates:
            return []

        candidates.sort(key=lambda c: c.score, reverse=True)
        clique = self._max_weight_consensus_clique(candidates)

        if not clique:
            return []

        if len(clique) == 1 and not self._passes_margin_gate(clique[0], candidates):
            return []

        return [c.entity_id for c in clique]

    @staticmethod
    def _source_of(entity_id: str) -> str:
        if entity_id.startswith("S2-"):
            return "S2"
        if entity_id.startswith("S3-"):
            return "S3"
        raise ValueError(f"Unexpected entity id (not S2-/S3-): {entity_id!r}")

    def _passes_margin_gate(self, chosen: Candidate, all_candidates: List[Candidate]) -> bool:
        runner_up = next((c for c in all_candidates if c.entity_id != chosen.entity_id), None)
        if runner_up is None:
            return True
        return (chosen.score - runner_up.score) > self.margin_threshold

    def _consistency(self, a: Candidate, b: Candidate) -> float:
        if a.source == b.source:
            return 1.0
        key = frozenset((a.entity_id, b.entity_id))
        return self.cross_source_scores.get(key, self.default_consistency)

    def _max_weight_consensus_clique(self, candidates: List[Candidate]) -> List[Candidate]:
        n = len(candidates)
        compatible = [[True] * n for _ in range(n)]

        for i, j in itertools.combinations(range(n), 2):
            ok = self._consistency(candidates[i], candidates[j]) >= self.consistency_threshold
            compatible[i][j] = compatible[j][i] = ok

        weights = [c.score for c in candidates]

        if n <= self.max_exact_clique_size:
            best_subset = self._exact_max_weight_clique(compatible, weights)
        else:
            best_subset = self._greedy_max_weight_clique(compatible, weights)

        return [candidates[i] for i in best_subset]

    @staticmethod
    def _exact_max_weight_clique(
        compatible: List[List[bool]], weights: List[float]
    ) -> List[int]:
        n = len(weights)
        best = {"subset": [], "weight": 0.0}

        order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        suffix_max = [0.0] * (n + 1)

        for idx in range(n - 1, -1, -1):
            suffix_max[idx] = suffix_max[idx + 1] + weights[order[idx]]

        def branch(pos: int, chosen: List[int], chosen_weight: float) -> None:
            if chosen_weight > best["weight"]:
                best["weight"] = chosen_weight
                best["subset"] = list(chosen)

            if pos == n:
                return

            if chosen_weight + suffix_max[pos] <= best["weight"]:
                return

            candidate_idx = order[pos]

            if all(compatible[candidate_idx][c] for c in chosen):
                chosen.append(candidate_idx)
                branch(pos + 1, chosen, chosen_weight + weights[candidate_idx])
                chosen.pop()

            branch(pos + 1, chosen, chosen_weight)

        branch(0, [], 0.0)
        return best["subset"]

    @staticmethod
    def _greedy_max_weight_clique(
        compatible: List[List[bool]], weights: List[float]
    ) -> List[int]:
        order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
        chosen: List[int] = []

        for idx in order:
            if all(compatible[idx][c] for c in chosen):
                chosen.append(idx)

        return chosen
