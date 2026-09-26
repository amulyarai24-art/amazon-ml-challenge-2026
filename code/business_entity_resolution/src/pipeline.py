"""
pipeline.py

Owned by: Person 4 (Graph Consensus, Calibration & Submission QA Lead)

End-to-end orchestration: takes the blocking stage's candidate pairs plus
the matching model's raw scores, calibrates them, runs tripartite graph
consensus + margin gating per Source-1 entity, and writes the final
leaderboard submission file.

Expected inputs (produced by the rest of the team's stages)
-------------------------------------------------------------
dataset_dir/
    test/test_source1.tsv, test_source2.tsv, test_source3.tsv

output_dir/
    candidate_pairs.tsv    (source1_entity_id, candidate_entity_ids)
                           -- the final blocking/candidate-generation output
    candidate_scores.tsv   (source1_entity_id, candidate_entity_id, raw_score)
                           -- raw matcher score per S1-candidate pair
    s2_s3_scores.tsv       (entity_id_a, entity_id_b, raw_score)  [optional]
                           -- raw matcher score for S2<->S3 candidate pairs,
                              used for cross-source consistency checks

NOTE: `candidate_scores.tsv` / `s2_s3_scores.tsv` are not part of the
official submission format (only matching_results.tsv and
candidate_pairs.tsv are) -- they are this pipeline's internal hand-off
files between the matching-model stage and this consensus stage. Adjust
the loader functions below if your teammates' matching model instead
exposes scores via a different file name / in-memory object.

Produces
--------
output_dir/matching_results.tsv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple

from tripartite_graph_consensus import TripartiteGraphConsensus, calibrate_probabilities


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _load_s1_ids(dataset_dir: Path) -> List[str]:
    rows = _read_tsv(dataset_dir / "test" / "test_source1.tsv")
    return [row["entity_id"] for row in rows]


def _load_candidate_pairs(output_dir: Path) -> Dict[str, List[str]]:
    rows = _read_tsv(output_dir / "candidate_pairs.tsv")
    result: Dict[str, List[str]] = {}
    for row in rows:
        ids = row.get("candidate_entity_ids", "").strip()
        result[row["source1_entity_id"]] = ids.split(",") if ids else []
    return result


def _load_raw_scores(output_dir: Path) -> Dict[str, Dict[str, float]]:
    """Returns {source1_entity_id: {candidate_entity_id: raw_score}}."""
    rows = _read_tsv(output_dir / "candidate_scores.tsv")
    result: Dict[str, Dict[str, float]] = {}
    for row in rows:
        result.setdefault(row["source1_entity_id"], {})[row["candidate_entity_id"]] = float(
            row["raw_score"]
        )
    return result


def _load_s2_s3_scores(output_dir: Path) -> Dict[FrozenSet[str], float]:
    path = output_dir / "s2_s3_scores.tsv"
    if not path.exists():
        return {}
    rows = _read_tsv(path)
    result: Dict[FrozenSet[str], float] = {}
    for row in rows:
        key = frozenset((row["entity_id_a"], row["entity_id_b"]))
        result[key] = float(row["raw_score"])
    return result


def run_pipeline(dataset_dir: str, output_dir: str) -> None:
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    s1_ids = _load_s1_ids(dataset_path)
    candidate_pairs = _load_candidate_pairs(output_path)
    raw_scores = _load_raw_scores(output_path)
    raw_s2_s3_scores = _load_s2_s3_scores(output_path)
    if not raw_scores:
        raise ValueError("candidate_scores.tsv contains no scores; run the matching stage before graph consensus.")

    # Calibrate the S2<->S3 consistency scores with the same isotonic model
    # so they live on the same probability scale as the S1<->candidate scores.
    calibrated_s2_s3: Dict[FrozenSet[str], float] = {}
    if raw_s2_s3_scores:
        keys = list(raw_s2_s3_scores.keys())
        calibrated_values = calibrate_probabilities([raw_s2_s3_scores[k] for k in keys])
        calibrated_s2_s3 = dict(zip(keys, calibrated_values))

    consensus = TripartiteGraphConsensus(cross_source_scores=calibrated_s2_s3)

    results_rows: List[Tuple[str, str]] = []
    for s1_id in s1_ids:
        candidate_ids = [cid for cid in candidate_pairs.get(s1_id, []) if cid]
        scores_for_s1 = raw_scores.get(s1_id, {})

        candidate_scores: List[dict] = []
        if candidate_ids:
            raw_values = [scores_for_s1.get(cid, 0.0) for cid in candidate_ids]
            calibrated_values = calibrate_probabilities(raw_values)
            candidate_scores = [
                {"entity_id": cid, "score": float(prob)}
                for cid, prob in zip(candidate_ids, calibrated_values)
            ]

        matched_ids = consensus.resolve_consensus(s1_id, candidate_scores)
        results_rows.append((s1_id, ",".join(matched_ids)))

    _write_matching_results(output_path / "matching_results.tsv", results_rows)


def _write_matching_results(path: Path, rows: List[Tuple[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id, matched in rows:
            writer.writerow([s1_id, matched])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the entity resolution pipeline end-to-end.")
    parser.add_argument("--dataset-dir", required=True, help="Path to the challenge dataset/ directory")
    parser.add_argument("--output-dir", required=True, help="Path to write matching_results.tsv into")
    args = parser.parse_args()

    run_pipeline(args.dataset_dir, args.output_dir)
    print(f"Wrote {Path(args.output_dir) / 'matching_results.tsv'}")
