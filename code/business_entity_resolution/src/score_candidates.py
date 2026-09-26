"""Score real blocked candidate pairs with the trained LightGBM matcher.

Run from code/business_entity_resolution:
    python src/score_candidates.py --dataset-dir ../dataset --output-dir ../output --model-dir ../models
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import lightgbm as lgb
import polars as pl

from cascade_ranker import compute_structured_features


def _read_tsv(path: Path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _find_col(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def _records(path: Path):
    df = pl.read_csv(path, separator="\t", ignore_errors=True)
    id_col = _find_col(df, ["entity_id", "id", "source1_entity_id", "source2_entity_id"])
    name_col = _find_col(df, ["name_norm", "name_normalized", "business_name", "company_name", "name"])
    addr_col = _find_col(df, ["address_norm", "address_normalized", "business_address", "address"])
    post_col = _find_col(df, ["postal", "postal_code", "zip", "postcode"])
    if not id_col:
        raise ValueError(f"Could not identify entity ID column in {path}")

    rows = {}
    for row in df.to_dicts():
        rows[str(row[id_col])] = {
            "name_norm": row.get(name_col, "") if name_col else "",
            "address_norm": row.get(addr_col, "") if addr_col else "",
            "postal": row.get(post_col, "") if post_col else "",
        }
    return rows


def score_candidates(dataset_dir: str, output_dir: str, model_dir: str) -> None:
    dataset = Path(dataset_dir)
    output = Path(output_dir)
    models = Path(model_dir)

    s1 = _records(dataset / "test" / "test_source1.tsv")
    s2 = _records(dataset / "test" / "test_source2.tsv")
    s3 = _records(dataset / "test" / "test_source3.tsv")
    candidates = {**s2, **s3}
    candidate_path = output / "candidate_pairs.tsv"
    model_path = models / "lgbm_stage1_ranker.txt"

    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing blocking output: {candidate_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained matcher: {model_path}")

    model = lgb.Booster(model_file=str(model_path))
    rows = _read_tsv(candidate_path)

    scored = []
    for row in rows:
        sid = str(row["source1_entity_id"])
        candidate_ids = [
            cid.strip()
            for cid in row.get("candidate_entity_ids", "").split(",")
            if cid.strip()
        ]
        if sid not in s1:
            raise ValueError(f"Candidate file references unknown S1 ID: {sid}")

        for cid in candidate_ids:
            if cid not in candidates:
                raise ValueError(f"Candidate file references unknown S2/S3 ID: {cid}")
            features = compute_structured_features(s1[sid], candidates[cid]).reshape(1, -1)
            raw_score = float(model.predict(features)[0])
            scored.append((sid, cid, raw_score))

    output.mkdir(parents=True, exist_ok=True)
    out_path = output / "candidate_scores.tsv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "raw_score"])
        writer.writerows(scored)

    print(f"[SUCCESS] Wrote {len(scored)} candidate scores to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score blocked S1-S2 candidate pairs.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()
    score_candidates(args.dataset_dir, args.output_dir, args.model_dir)
