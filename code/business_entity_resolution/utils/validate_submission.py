"""Offline validator for the official Amazon ML Challenge submission files."""

import argparse
import csv
import os
import sys


def _read_ids(path, id_column):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if id_column not in (rows[0].keys() if rows else []):
        raise ValueError(f"{path}: missing column {id_column!r}")
    return rows


def validate(matching_file, candidate_file, test_dir):
    errors = []

    test_s1 = os.path.join(test_dir, "test_source1.tsv")
    test_s2 = os.path.join(test_dir, "test_source2.tsv")
    test_s3 = os.path.join(test_dir, "test_source3.tsv")

    for path in (matching_file, candidate_file, test_s1, test_s2, test_s3):
        if not os.path.exists(path):
            errors.append(f"Missing file: {path}")

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        return 1

    try:
        s1_rows = _read_ids(test_s1, "entity_id")
        s2_rows = _read_ids(test_s2, "entity_id")
        s3_rows = _read_ids(test_s3, "entity_id")
        matching_rows = _read_ids(matching_file, "source1_entity_id")
        candidate_rows = _read_ids(candidate_file, "source1_entity_id")

        s1_ids = {r["entity_id"] for r in s1_rows}
        target_ids = {r["entity_id"] for r in s2_rows} | {r["entity_id"] for r in s3_rows}

        if len(matching_rows) != len(s1_rows):
            errors.append(f"matching_results has {len(matching_rows)} rows; expected {len(s1_rows)}.")
        if len(candidate_rows) != len(s1_rows):
            errors.append(f"candidate_pairs has {len(candidate_rows)} rows; expected {len(s1_rows)}.")

        def check_rows(rows, column, label):
            seen_s1 = set()
            parsed = {}
            for row in rows:
                sid = row.get("source1_entity_id", "")
                if sid in seen_s1:
                    errors.append(f"{label}: duplicate Source-1 row: {sid}")
                seen_s1.add(sid)
                if sid not in s1_ids:
                    errors.append(f"{label}: unknown Source-1 ID: {sid}")
                raw = row.get(column, "")
                ids = [x.strip() for x in raw.split(",") if x.strip()]
                if len(ids) != len(set(ids)):
                    errors.append(f"{label}: duplicate candidate/match ID for {sid}")
                bad = [x for x in ids if x not in target_ids]
                if bad:
                    errors.append(f"{label}: invalid target IDs for {sid}: {bad[:3]}")
                parsed[sid] = ids
            return parsed

        matches = check_rows(matching_rows, "matched_entity_ids", "matching_results")
        candidates = check_rows(candidate_rows, "candidate_entity_ids", "candidate_pairs")

        missing_s1_match = s1_ids - set(matches)
        missing_s1_candidate = s1_ids - set(candidates)
        if missing_s1_match:
            errors.append(f"matching_results missing {len(missing_s1_match)} Source-1 IDs.")
        if missing_s1_candidate:
            errors.append(f"candidate_pairs missing {len(missing_s1_candidate)} Source-1 IDs.")

        for sid, ids in matches.items():
            if not set(ids).issubset(set(candidates.get(sid, []))):
                errors.append(f"matching_results contains an ID not present in candidate_pairs for {sid}.")

    except Exception as exc:
        errors.append(str(exc))

    print("==================================================")
    print(" Amazon ML Challenge 2026 Submission Validator")
    print("==================================================")
    if errors:
        for i, error in enumerate(errors[:50], 1):
            print(f"[FAIL {i}] {error}")
        if len(errors) > 50:
            print(f"... and {len(errors) - 50} more issue(s).")
        print("RESULT: FAIL")
        return 1

    print("[PASS] Format, ID, row-count, duplicate, and candidate-subset checks passed.")
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", default="output/matching_results.tsv")
    parser.add_argument("--candidate", default="output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", default="dataset/test")
    args = parser.parse_args()
    sys.exit(validate(args.matching, args.candidate, args.test_dir))
