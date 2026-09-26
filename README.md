# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains the team's runnable solution for the Amazon ML Challenge 2026 Business Entity Resolution problem.

The pipeline uses only the provided challenge data. Large/raw datasets, trained models, and generated TSV outputs are kept local and ignored by Git.

## Pipeline

```text
Training data
    ↓
Preprocessing / normalization
    ↓
Blocking: S1 → S2 + S3 candidate pairs
    ↓
LightGBM pair scoring
    ↓
Score calibration + tripartite graph consensus
    ↓
matching_results.tsv
```

### Components

- `preprocessing/` — cleaning, normalization, postal extraction, and profiling.
- `hackathon_blocking/` — candidate generation for S1 against S2/S3.
- `src/train.py` — trains the LightGBM matcher from the official training ground truth and saves calibration parameters.
- `src/score_candidates.py` — scores the final blocked candidates.
- `src/pipeline.py` — calibrates scores, applies graph consensus/margin gating, and writes final matches.
- `utils/validate_submission.py` — official-format submission validator.

The current runnable matching pipeline is LightGBM + calibration + tripartite graph consensus. Older transformer/Ditto placeholders are not part of the active pipeline.

## Dataset layout

Keep the challenge data locally:

```text
code/business_entity_resolution/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── data/processed/       # generated cleaned files
├── models/               # generated model/calibration files
└── output/               # generated candidate/match files
```

Do **not** commit the raw challenge datasets, generated models, or generated TSV outputs.

## Setup

From `code/business_entity_resolution`:

```bash
pip install -r requirements.txt
```

## Run the complete pipeline

Run these stages in order.

### 1. Train the matcher

From `code/business_entity_resolution/src`:

```bash
python train.py --train_dir ../dataset/train --model_dir ../models
```

This creates the LightGBM model and calibration files under `models/`.

### 2. Run preprocessing / blocking

Make sure Person 1's cleaned files exist under `code/business_entity_resolution/data/processed/`.

Then from `code/business_entity_resolution/hackathon_blocking`:

```bash
python run_blocking.py
```

This creates:

```text
output/candidate_pairs.tsv
```

### 3. Score blocked candidates

From `code/business_entity_resolution/src`:

```bash
python score_candidates.py --dataset-dir ../dataset --output-dir ../output --model-dir ../models
```

This creates the internal hand-off file:

```text
output/candidate_scores.tsv
```

### 4. Produce final submission

From `code/business_entity_resolution/src`:

```bash
python pipeline.py --dataset-dir ../dataset --output-dir ../output
```

This creates:

```text
output/matching_results.tsv
```

### 5. Validate before uploading

From the repository root:

```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

A successful validation should report `PASS`.

## Official submission files

Only these two generated files belong in the final submission output:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

`candidate_scores.tsv` is an internal pipeline file and is not an official submission output.

The final results must contain every test Source-1 entity exactly once, with only valid Source-2/Source-3 IDs and no duplicate IDs.

## Team workflow

The intended hand-off is:

```text
Person 1 → cleaned data
Person 2 → candidate_pairs.tsv
Person 3 → trained matcher + candidate_scores.tsv
Person 4 → calibration + graph consensus + validation
```

Keep raw challenge data local and use Git for source code/configuration rather than large generated artifacts.

## Compliance

The solution is designed around the challenge rules:

- TSV input/output with explicit tab separation.
- Source 1 is the reference source; matches are only from Source 2 and Source 3.
- Every test Source-1 entity is preserved in the final output.
- No external business-identity lookup is used.
- Generated datasets/models/outputs are excluded from Git via `.gitignore`.

