#!/usr/bin/env python3
"""
validate_submission.py

Owned by: Person 4 (Graph Consensus, Calibration & Submission QA Lead)

Stdlib-only validator for matching_results.tsv and candidate_pairs.tsv
against every rule in the challenge's submission format spec. Run this
before uploading to the leaderboard so a formatting mistake doesn't cost a
submission attempt.

Usage
-----
python3 src/validate_submission.py \\
    --matching output/matching_results.tsv \\
    --candidate output/candidate_pairs.tsv \\
    --test-dir dataset/test

Prints "RESULT: PASS" (exit 0) if the files are safe to submit, or
"RESULT: FAIL" plus a numbered list of issues (exit 1).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Set


def _load_entity_ids(path: Path) -> Set[str]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        return {row["entity_id"] for row in reader}


def _parse_id_list(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    return [x.strip() for x in raw.split(",")]


def _validate_result_file(
    path: Path,
    id_column: str,
    all_s1_ids: Set[str],
    valid_secondary_ids: Set[str],
    file_label: str,
) -> List[str]:
    issues: List[str] = []

    if not path.exists():
        return [f"{file_label}: file not found at {path}"]

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = reader.fieldnames or []
        expected_cols = {"source1_entity_id", id_column}
        if not expected_cols.issubset(set(fieldnames)):
            issues.append(
                f"{file_label}: expected columns {sorted(expected_cols)}, found {fieldnames}"
            )
            return issues

        seen_s1_ids: Dict[str, int] = {}

        for line_no, row in enumerate(reader, start=2):  # header is line 1
            s1_id = row["source1_entity_id"].strip()
            id_list = _parse_id_list(row.get(id_column, ""))

            if not s1_id:
                issues.append(f"{file_label} line {line_no}: empty source1_entity_id")
                continue

            seen_s1_ids[s1_id] = seen_s1_ids.get(s1_id, 0) + 1

            if s1_id not in all_s1_ids:
                issues.append(
                    f"{file_label} line {line_no}: source1_entity_id '{s1_id}' "
                    f"does not exist in the test set"
                )

            if len(set(id_list)) != len(id_list):
                dupes = sorted({x for x in id_list if id_list.count(x) > 1})
                issues.append(
                    f"{file_label} line {line_no} ({s1_id}): duplicate ids in list: {dupes}"
                )

            for entity_id in id_list:
                if entity_id == s1_id:
                    issues.append(
                        f"{file_label} line {line_no} ({s1_id}): self-match to its "
                        f"own id is not allowed"
                    )
                elif entity_id.startswith("S1-"):
                    issues.append(
                        f"{file_label} line {line_no} ({s1_id}): id list contains a "
                        f"Source-1 id ('{entity_id}'); only Source-2/Source-3 ids are allowed"
                    )
                elif entity_id not in valid_secondary_ids:
                    issues.append(
                        f"{file_label} line {line_no} ({s1_id}): id '{entity_id}' "
                        f"does not exist in the test set"
                    )

        duplicate_s1 = sorted(s1 for s1, count in seen_s1_ids.items() if count > 1)
        if duplicate_s1:
            issues.append(f"{file_label}: duplicate source1_entity_id rows: {duplicate_s1}")

        missing_s1 = all_s1_ids - set(seen_s1_ids.keys())
        if missing_s1:
            missing_sorted = sorted(missing_s1)
            issues.append(
                f"{file_label}: missing rows for {len(missing_s1)} source1 entities, "
                f"e.g. {missing_sorted[:5]}"
            )

    return issues


def _validate_matches_subset_of_candidates(
    matching_path: Path,
    candidate_path: Path,
    id_column_matching: str,
    id_column_candidate: str,
) -> List[str]:
    issues: List[str] = []
    if not matching_path.exists() or not candidate_path.exists():
        return issues

    def load(path: Path, col: str) -> Dict[str, Set[str]]:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            return {
                row["source1_entity_id"].strip(): set(_parse_id_list(row.get(col, "")))
                for row in reader
            }

    matched = load(matching_path, id_column_matching)
    candidates = load(candidate_path, id_column_candidate)

    for s1_id, matched_ids in matched.items():
        cand_ids = candidates.get(s1_id, set())
        extra = matched_ids - cand_ids
        if extra:
            issues.append(
                f"matching_results.tsv ({s1_id}): matched id(s) {sorted(extra)} never "
                f"appeared in candidate_pairs.tsv -- likely a pipeline bug"
            )
    return issues


def validate(matching_path: Path, candidate_path: Path, test_dir: Path) -> List[str]:
    issues: List[str] = []

    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"

    for p in (s1_path, s2_path, s3_path):
        if not p.exists():
            issues.append(f"test data file not found: {p}")
    if issues:
        return issues

    all_s1_ids = _load_entity_ids(s1_path)
    valid_secondary_ids = _load_entity_ids(s2_path) | _load_entity_ids(s3_path)

    issues += _validate_result_file(
        matching_path, "matched_entity_ids", all_s1_ids, valid_secondary_ids,
        "matching_results.tsv",
    )
    issues += _validate_result_file(
        candidate_path, "candidate_entity_ids", all_s1_ids, valid_secondary_ids,
        "candidate_pairs.tsv",
    )
    issues += _validate_matches_subset_of_candidates(
        matching_path, candidate_path, "matched_entity_ids", "candidate_entity_ids",
    )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a challenge submission.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to dataset/test directory")
    args = parser.parse_args()

    issues = validate(Path(args.matching), Path(args.candidate), Path(args.test_dir))

    if issues:
        print(f"RESULT: FAIL ({len(issues)} issue(s) found)\n")
        for i, issue in enumerate(issues, start=1):
            print(f"{i}. {issue}")
        return 1

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
