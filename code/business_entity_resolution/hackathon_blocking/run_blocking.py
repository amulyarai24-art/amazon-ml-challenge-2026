import os
from pathlib import Path

import pandas as pd

from src.blocking import EntityBlocker


def _pick_column(df, candidates, label):
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find {label} column. Tried {candidates}; available columns: {list(df.columns)}")


def _load_processed(base_dir, source):
    candidates = [
        Path(base_dir).parent / "data" / "processed" / f"{source}_clean.tsv",
        Path(base_dir) / "data" / "processed" / f"{source}_clean.tsv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path, sep="\t")
    raise FileNotFoundError(f"Processed {source} data not found. Checked: {candidates}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    df_s1 = _load_processed(base_dir, "s1")
    df_s2 = _load_processed(base_dir, "s2")

    s1_id = _pick_column(df_s1, ["entity_id", "id", "source1_entity_id"], "Source-1 id")
    s2_id = _pick_column(df_s2, ["entity_id", "id", "source2_entity_id"], "Source-2 id")
    name_s1 = _pick_column(df_s1, ["name_norm", "name_normalized", "company_name", "business_name", "name"], "Source-1 name")
    name_s2 = _pick_column(df_s2, ["name_norm", "name_normalized", "company_name", "business_name", "name"], "Source-2 name")

    left = df_s1[[s1_id, name_s1]].rename(columns={s1_id: "entity_id", name_s1: "match_name"})
    right = df_s2[[s2_id, name_s2]].rename(columns={s2_id: "entity_id", name_s2: "match_name"})

    blocker = EntityBlocker(prefix_len=3, top_k=5)
    pairs = blocker.generate_candidate_pairs(left, right, "match_name", "entity_id", "entity_id")

    grouped = (
        pairs.groupby("entity_id_a")["entity_id_b"]
        .apply(lambda values: ",".join(dict.fromkeys(map(str, values))))
        .reset_index()
        .rename(columns={"entity_id_a": "source1_entity_id", "entity_id_b": "candidate_entity_ids"})
    )

    all_s1 = pd.DataFrame({"source1_entity_id": df_s1[s1_id].astype(str)})
    output = all_s1.merge(grouped, on="source1_entity_id", how="left")
    output["candidate_entity_ids"] = output["candidate_entity_ids"].fillna("")

    project_output = Path(base_dir).parent / "output"
    project_output.mkdir(parents=True, exist_ok=True)
    output_path = project_output / "candidate_pairs.tsv"
    output.to_csv(output_path, sep="\t", index=False)
    print(f"[OK] Wrote {len(output):,} S1 rows to {output_path}")


if __name__ == "__main__":
    main()
