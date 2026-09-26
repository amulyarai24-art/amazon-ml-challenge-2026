# ⚡ Business Entity Resolution Engine 🏢🔍

> **Amazon ML Challenge 2026** | A high-throughput, multi-stage Machine Learning framework for large-scale record linkage, company entity matching, and cross-source record disambiguation. 🚀✨

---

# 🧠 System Architecture & Design

## 📌 What is Business Entity Resolution?

Business Entity Resolution is the process of determining whether records from different datasets refer to the **same real-world business entity**.

In real-world datasets, the same company may appear in multiple forms:

| Source | Company Name | Address | Other Information |
|---|---|---|---|
| Dataset A | `ABC Technologies Pvt Ltd` | `12 MG Road, Bangalore` | Website |
| Dataset B | `ABC Technology Private Limited` | `12 Mahatma Gandhi Rd, Bengaluru` | Phone |
| Dataset C | `A.B.C. Technologies` | `12 MG Rd., Bengaluru` | Email |

A simple exact string comparison would consider these records different.

A business entity resolution system must instead learn that:

```text
ABC Technologies Pvt Ltd
ABC Technology Private Limited
A.B.C. Technologies
```

may all represent the same underlying company.

The problem becomes substantially harder when dealing with:

- multilingual company names
- spelling variations
- abbreviations
- punctuation differences
- missing fields
- noisy addresses
- inconsistent phone numbers
- different formatting conventions
- transliteration
- duplicate records
- conflicting information
- large datasets containing millions of possible record pairs

The goal of this project is therefore not merely to calculate string similarity.

It is to build a **coarse-to-fine matching system** that combines deterministic rules, probabilistic linkage, gradient-boosted ranking, transformer-based semantic matching, and graph-based global consistency.

---

# 🏗️ Overall Architecture

The system follows a **multi-stage cascade**.

Instead of comparing every record against every other record using an expensive neural model, inexpensive techniques are applied first to reduce the search space. More computationally expensive models are then applied only to increasingly ambiguous candidate pairs.

```text
                         ┌──────────────────────────────┐
                         │       Raw Source Data        │
                         │                              │
                         │ Names / Address / Phone /    │
                         │ Email / Website / Metadata   │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Stage 0                      │
                         │ Multilingual Normalization   │
                         │                              │
                         │ Unicode normalization        │
                         │ Suffix normalization         │
                         │ Punctuation cleanup          │
                         │ Field standardization        │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Stage 1                      │
                         │ Candidate Blocking           │
                         │                              │
                         │ Exact token blocks           │
                         │ Geographic/postal buckets   │
                         │ Token-overlap retrieval      │
                         └──────────────┬───────────────┘
                                        │
                             Candidate Pair Set
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Stage 2                      │
                         │ Fellegi-Sunter Linkage       │
                         │                              │
                         │ Probabilistic evidence       │
                         │ Match / Non-match likelihood │
                         │ Ratio-based scoring           │
                         └──────────────┬───────────────┘
                                        │
                         ┌──────────────┴───────────────┐
                         │                              │
                    High confidence                Ambiguous
                         │                              │
                         ▼                              ▼
                  ┌──────────────┐       ┌──────────────────────────────┐
                  │ Early Match  │       │ Stage 3                      │
                  │ / Decision   │       │ LightGBM Cascade Ranker      │
                  └──────────────┘       │                              │
                                         │ Fuzzy similarities            │
                                         │ Structural features           │
                                         │ Cross-field evidence          │
                                         └──────────────┬───────────────┘
                                                        │
                                             ┌──────────┴──────────┐
                                             │                     │
                                        Confident             Ambiguous
                                             │                     │
                                             ▼                     ▼
                                      ┌──────────────┐  ┌───────────────────────┐
                                      │ Pair Match   │  │ Stage 4               │
                                      │ Decision     │  │ Ditto Transformer     │
                                      └──────────────┘  │ Cross-Encoder         │
                                                        │                       │
                                                        │ BERT / DistilBERT     │
                                                        │ Cross-attention       │
                                                        └───────────┬───────────┘
                                                                    │
                                                                    ▼
                                                        ┌───────────────────────┐
                                                        │ Stage 5               │
                                                        │ Tripartite Graph      │
                                                        │ Consensus             │
                                                        │                       │
                                                        │ Global consistency    │
                                                        │ Connected components  │
                                                        │ Conflict resolution   │
                                                        └───────────┬───────────┘
                                                                    │
                                                                    ▼
                                                        ┌───────────────────────┐
                                                        │ Final Entity Groups   │
                                                        │ / Resolved Records    │
                                                        └───────────────────────┘
```

---

# 🔄 Why a Cascading Architecture?

A naïve entity resolution system could attempt to compare every record with every other record.

For `N` records, exhaustive pairwise comparison creates a quadratic search space:

```text
N × N
```

For large datasets, this becomes computationally impractical, particularly when the final matcher is a transformer model.

The proposed architecture therefore follows a **coarse-to-fine strategy**.

| Stage | Purpose | Relative Cost |
|---|---|---:|
| Normalization | Make equivalent representations comparable | Very Low |
| Blocking | Generate plausible candidates | Low |
| Fellegi-Sunter | Probabilistic evidence filtering | Low |
| LightGBM | Strong tabular ranking | Medium |
| Ditto | Deep semantic comparison | High |
| Graph Consensus | Global consistency | Medium |

The expensive model is therefore reserved for the pairs that actually require deeper reasoning.

This design provides two important benefits:

### ⚡ Computational Efficiency

Most obviously incorrect pairs can be eliminated before transformer inference.

### 🎯 Precision Preservation

Ambiguous cases are not forced through simplistic rules. They can be escalated to stronger models.

---

# 🧹 Stage 0 — Multilingual Text Normalization & Cleaning

The first component converts heterogeneous raw records into a more consistent representation.

Normalization is important because many apparent mismatches are caused by formatting rather than actual entity differences.

## Main Operations

### 1. Unicode Normalization

Unicode representations can contain visually equivalent but internally different character sequences.

The system applies normalization such as:

```python
unicodedata.normalize("NFKD", text)
```

This helps create a more consistent representation before downstream matching.

---

### 2. Business Legal Suffix Normalization

Companies frequently appear with different legal suffixes.

Examples:

```text
ABC Technologies Pvt Ltd
ABC Technologies Private Limited
ABC Technologies Ltd
ABC Technologies Limited
```

These suffixes can be normalized into a common representation where appropriate.

---

### 3. Punctuation Normalization

Formatting differences such as:

```text
A.B.C. Technologies
ABC Technologies
ABC-Technologies
```

can create unnecessary string differences.

The normalization layer removes or standardizes punctuation while preserving meaningful alphanumeric information.

This is particularly important when fields contain identifiers.

---

### 4. Whitespace Normalization

Repeated or irregular whitespace is standardized.

For example:

```text
"ABC   Technologies"
```

becomes:

```text
"ABC Technologies"
```

---

### 5. Field-Level Standardization

Different fields can require different preprocessing.

Examples include:

```text
Company Name
Address
Phone
Email
Website
Postal Code
```

The system therefore treats normalization as a field-aware preprocessing stage rather than simply applying one generic string transformation everywhere.

---

# 🧱 Stage 1 — Candidate Pair Blocking Engine

Blocking is one of the most important scalability components.

The purpose of blocking is **not to decide whether two records match**.

Instead, blocking answers a different question:

> "Which records are plausible enough to compare further?"

This produces a candidate pair set that is much smaller than the full pairwise search space.

---

## 🔑 Blocking Strategy

The project uses multiple blocking signals.

### Exact Token Blocks

Important normalized tokens can be used to place records into common candidate buckets.

For example:

```text
ABC Technologies Bangalore
```

may produce tokens such as:

```text
abc
technologies
bangalore
```

Records sharing informative tokens can enter the same candidate block.

---

### 🌍 Geographic / Postal Bucketing

Location information provides another useful signal.

Records sharing a postal or geographic bucket can be considered together.

For example:

```text
560001
```

can be used as a blocking signal for records associated with the same locality.

---

### 🔤 Token-Overlap Retrieval

Instead of requiring complete string equality, records can be retrieved based on overlapping normalized tokens.

This allows candidate generation to survive moderate spelling and formatting differences.

---

## Blocking Philosophy

Blocking should be:

```text
Broad enough to retain true matches
BUT
Restrictive enough to avoid unnecessary candidate pairs
```

A blocker that is too restrictive can produce **false negatives before the ML models even see the pair**.

A blocker that is too broad creates an unnecessarily large candidate set.

Therefore, blocking quality is an important part of the overall pipeline.

---

# 📊 Stage 2 — Fellegi-Sunter Probabilistic Linkage

After candidate generation, the system can use probabilistic record linkage to estimate how strong the observed evidence is for a match.

The Fellegi-Sunter framework compares field-level agreement patterns under two hypotheses:

```text
M = Pair is a true match
U = Pair is a non-match
```

For an observed comparison pattern `γ`, the system evaluates how much more likely that pattern is under `M` than under `U`.

---

## Mathematical Intuition

For a comparison feature `i`:

```text
mᵢ = P(γᵢ | Match)
```

and:

```text
uᵢ = P(γᵢ | Non-Match)
```

The corresponding evidence can be expressed through a log-likelihood ratio:

```text
wᵢ = log(mᵢ / uᵢ)
```

Across multiple fields, the evidence can be combined:

```text
W = Σᵢ wᵢ
```

A large positive score indicates that the observed agreement pattern is more characteristic of a true match.

A low or negative score indicates that the evidence is more consistent with a non-match.

---

## Why Probabilistic Linkage?

Pure string similarity can answer:

```text
"How similar are these strings?"
```

Probabilistic linkage asks:

```text
"How much does this observed similarity change our belief that
these two records represent the same entity?"
```

This provides a useful intermediate layer between deterministic blocking and machine-learning ranking.

---

# 🌳 Stage 3 — LightGBM Cascade Ranker

The next stage uses a gradient-boosted decision tree model to combine multiple pairwise features.

Rather than relying on a single similarity score, the ranker can evaluate a richer feature vector.

---

## 🔢 String Similarity Features

The feature generation layer can include fuzzy matching measurements such as:

### RapidFuzz Ratio

Measures overall character-level similarity.

### Partial Ratio

Useful when one string is contained within or closely resembles a portion of another.

### Token Sort Ratio

Useful when words appear in a different order.

Example:

```text
ABC Technologies India
India ABC Technologies
```

### Token Set Ratio

Useful when one string contains additional repeated or reordered tokens.

---

## 📐 Additional Similarity Features

Other useful signals include:

```text
Normalized Levenshtein similarity
Jaro-Winkler similarity
Token overlap
Length differences
Exact-match indicators
Field-presence indicators
```

---

## 🧩 Structural Features

The model can also use non-textual evidence such as:

```text
Same postal code?
Same country?
Same domain?
Same phone suffix?
Same normalized identifier?
Same address components?
```

This allows LightGBM to reason over multiple heterogeneous signals simultaneously.

---

## 🚦 Cascade Behaviour

The LightGBM layer acts as an intermediate decision point.

Conceptually:

```text
Candidate Pair
      │
      ▼
LightGBM Score
      │
      ├── High confidence ──► Accept / Resolve
      │
      └── Ambiguous ────────► Deep Transformer
```

Thresholds should be treated as configurable model parameters rather than universal constants.

For example, an implementation may define regions such as:

```text
Score ≥ high_threshold
        → confident match

Score ≤ low_threshold
        → confident non-match

Otherwise
        → escalate
```

The exact thresholds should be selected using validation data and the target evaluation metric.

---

# 🤖 Stage 4 — Ditto Transformer Cross-Encoder

The deepest pairwise model in the architecture is a transformer-based cross-encoder inspired by the Ditto entity matching approach.

Unlike independent text embeddings, a cross-encoder receives both records together and allows the transformer to directly model interactions between them.

---

## Serialized Pair Representation

A pair of records can be represented as a serialized sequence such as:

```text
[CLS]
field_1_of_record_A
[SEP]
field_2_of_record_A
[SEP]
...
field_1_of_record_B
[SEP]
field_2_of_record_B
[SEP]
...
```

Conceptually:

```text
Record A
        +
Record B
        ↓
Serialized Pair
        ↓
Transformer
        ↓
Match Probability
```

---

## Why Cross-Encoder?

Suppose two records contain:

```text
ABC Technology Pvt Ltd
```

and:

```text
ABC Technologies Private Limited
```

A cross-encoder can directly model interactions between the two sequences.

The transformer can learn relationships such as:

- corresponding business names
- reordered words
- abbreviations
- contextual token similarity
- address relationships
- multilingual patterns
- field interactions

This is more expressive than treating each record as an isolated embedding.

---

# 🧠 BERT / DistilBERT Backbone

The transformer stage can use a pretrained language model such as:

```text
BERT
```

or a lighter variant such as:

```text
DistilBERT
```

The choice creates a trade-off between:

```text
Model capacity
        ↕
Inference cost
        ↕
Latency
        ↕
Memory usage
```

The model is fine-tuned for the binary entity matching task.

---

# 🔥 Custom Contrastive / Focal Loss

Entity matching datasets can contain substantially more non-matching pairs than matching pairs.

This class imbalance makes standard binary classification potentially suboptimal.

A focal-style objective can reduce the influence of easy examples and focus learning more strongly on difficult examples.

The standard focal-loss formulation is:

```text
L_Focal = -αₜ (1 - pₜ)^γ log(pₜ)
```

where:

```text
pₜ = model probability assigned to the true class
αₜ = class weighting factor
γ  = focusing parameter
```

The focusing term:

```text
(1 - pₜ)^γ
```

reduces the contribution of already-easy examples.

This can be useful when the model needs to focus on ambiguous record pairs.

The exact loss configuration should be treated as an implementation detail and should be verified against the training code when reporting experimental results.

---

# 🕸️ Stage 5 — Tripartite Graph Consensus & Clustering

Pairwise matching alone can produce inconsistent decisions.

Consider three records:

```text
A
B
C
```

Suppose:

```text
A ↔ B = Match
B ↔ C = Match
A ↔ C = Uncertain
```

A purely independent pairwise system may leave these decisions inconsistent.

The graph stage introduces global structure.

---

## Graph Representation

The entity resolution problem can be represented as:

```text
G = (V, E)
```

where:

```text
V = records / entities
E = predicted matching relationships
```

An edge can represent a sufficiently strong pairwise match.

---

## Connected Components

Once high-confidence match edges are created, connected components can be extracted.

For example:

```text
A ─── B
     │
     C
```

can form one entity cluster:

```text
{A, B, C}
```

This effectively performs a form of transitive consolidation.

---

## Why Graph Consensus?

Pairwise models reason locally:

```text
Record A ↔ Record B
```

The graph stage reasons globally:

```text
A ↔ B
B ↔ C
A ↔ C
```

This allows the final system to consider the consistency of relationships across an entire entity group.

---

## Conflict Resolution

Graph-based post-processing can also help identify problematic structures such as:

```text
Strong match
      ↓
Conflict
      ↓
Weak edge
```

Graph density, confidence values, and connectivity can be used to identify relationships that should be retained or reconsidered.

The graph layer should therefore be viewed as a **global consistency mechanism**, not simply another independent binary classifier.

---

# 🔄 Complete End-to-End Data Flow

The complete pipeline can be understood as:

```text
RAW RECORDS
    │
    ▼
┌─────────────────────────────┐
│ Normalization               │
│                             │
│ Unicode                     │
│ Legal suffixes              │
│ Punctuation                 │
│ Whitespace                  │
│ Field standardization       │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Blocking                    │
│                             │
│ Exact tokens                │
│ Geographic buckets          │
│ Token overlap               │
└──────────────┬──────────────┘
               │
               ▼
        Candidate Pairs
               │
               ▼
┌─────────────────────────────┐
│ Fellegi-Sunter              │
│                             │
│ Probabilistic evidence      │
│ Match vs Non-Match          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ LightGBM Ranker             │
│                             │
│ Fuzzy similarity            │
│ Structural features         │
│ Cross-field evidence        │
└──────────────┬──────────────┘
               │
          Ambiguous Pairs
               │
               ▼
┌─────────────────────────────┐
│ Ditto Cross-Encoder         │
│                             │
│ BERT / DistilBERT           │
│ Pair serialization          │
│ Cross-attention             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Graph Consensus             │
│                             │
│ Match edges                 │
│ Connected components        │
│ Global consistency          │
└──────────────┬──────────────┘
               │
               ▼
      FINAL ENTITY GROUPS
```

---

# 🎯 Core Design Principles

## 1. Coarse-to-Fine Matching

Cheap operations happen before expensive operations.

```text
Cheap
 ↓
Blocking
 ↓
Probabilistic linkage
 ↓
Tree model
 ↓
Transformer
 ↓
Global graph reasoning
Expensive
```

---

## 2. Multiple Sources of Evidence

No single feature should be responsible for every decision.

The system combines:

```text
Name similarity
Address similarity
Phone information
Email information
Website information
Geographic information
Probabilistic evidence
Structural signals
Semantic transformer evidence
Graph relationships
```

---

## 3. Confidence-Based Escalation

A pair that is obviously a match does not necessarily need a transformer.

Likewise, a pair that is obviously unrelated should not consume expensive inference resources.

The architecture therefore follows:

```text
Certain
  ↓
Resolve quickly

Uncertain
  ↓
Spend more computation
```

---

## 4. Local + Global Reasoning

The system combines:

### Local reasoning

```text
Does Record A match Record B?
```

with:

### Global reasoning

```text
Are all these matching decisions mutually consistent?
```

This is the purpose of the graph consensus stage.

---

# 🧰 Core Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core implementation |
| PyTorch | Deep learning |
| HuggingFace Transformers | BERT / DistilBERT models |
| LightGBM | Gradient-boosted ranking |
| Scikit-Learn | Machine-learning utilities |
| RapidFuzz | Efficient fuzzy string matching |
| Polars | High-performance tabular processing |
| Pandas | Data manipulation |
| NetworkX | Graph construction and analysis |
| PyYAML | Configuration management |
| Custom Python CLI | Training and inference orchestration |

---

# 🔍 Matching Evidence

The system uses several categories of evidence.

## Textual Evidence

```text
Company name
Address
Website
Email
```

## Character-Level Evidence

```text
Levenshtein similarity
RapidFuzz Ratio
Partial Ratio
Jaro-Winkler
```

## Token-Level Evidence

```text
Token overlap
Token Set Ratio
Token Sort Ratio
```

## Structural Evidence

```text
Postal code equality
Country equality
Identifier equality
Field presence
Domain consistency
```

## Probabilistic Evidence

```text
m-probabilities
u-probabilities
Likelihood ratios
```

## Semantic Evidence

```text
Transformer representations
Cross-attention
Contextual interactions
```

## Global Evidence

```text
Graph connectivity
Connected components
Cluster consistency
```

---

# 🚦 Confidence-Based Cascade

A useful conceptual representation of the decision system is:

```text
                 Candidate Pair
                       │
                       ▼
              Fellegi-Sunter
                       │
                       ▼
              LightGBM Ranker
                       │
              ┌────────┴────────┐
              │                 │
        High confidence      Ambiguous
              │                 │
              ▼                 ▼
        Early decision      Transformer
                                  │
                                  ▼
                            Final pair score
                                  │
                                  ▼
                           Graph consensus
```

The actual thresholds should be calibrated experimentally rather than assumed to be universally optimal.

---

# 📈 Evaluation Strategy

For an entity resolution system, evaluation should consider the cost of false positives and false negatives.

A particularly useful metric for precision-sensitive matching is **F0.5**.

The F-score family is:

```text
Fβ = (1 + β²) × Precision × Recall
     --------------------------------
     (β² × Precision) + Recall
```

For:

```text
β = 0.5
```

the resulting metric gives greater relative importance to precision than recall.

This is useful when incorrectly merging two different businesses is particularly undesirable.

---

## Precision

```text
Precision =
True Positives
-------------------------
True Positives + False Positives
```

It measures how many predicted matches were actually correct.

---

## Recall

```text
Recall =
True Positives
-------------------------
True Positives + False Negatives
```

It measures how many of the true matches were successfully identified.

---

## F0.5

```text
F0.5 =
1.25 × Precision × Recall
-------------------------
0.25 × Precision + Recall
```

The exact competition evaluation procedure should always be treated as the authoritative definition when reporting final challenge results.

---

# 🧪 Training Strategy

The training process can be conceptually separated into multiple stages.

```text
Raw Training Data
       │
       ▼
Normalization
       │
       ▼
Candidate Generation
       │
       ▼
Pair Construction
       │
       ▼
Feature Generation
       │
       ├──────────────► LightGBM Training
       │
       └──────────────► Transformer Pair Dataset
                              │
                              ▼
                       Transformer Fine-Tuning
```

---

# 🐛 Error Analysis

Improving entity resolution is not only about increasing model complexity.

The most useful improvements often come from understanding **why the system is wrong**.

A useful error-analysis table is:

| Error Type | Example | Possible Cause |
|---|---|---|
| False Positive | Different companies with similar names | Insufficient discriminative evidence |
| False Negative | Same company with highly different names | Blocking failure / semantic variation |
| Address mismatch | Same entity, different address format | Normalization |
| Transliteration mismatch | Different scripts | Multilingual preprocessing |
| Abbreviation | `Pvt Ltd` vs `Private Limited` | Suffix handling |
| Missing data | Phone absent | Over-reliance on one field |
| Graph conflict | Inconsistent edges | Pairwise uncertainty |

---

# ⚠️ Important Failure Modes

## Blocking Failure

If the true pair never enters the candidate set:

```text
True Match
   ↓
Not blocked together
   ↓
Never reaches classifier
   ↓
False Negative
```

This makes blocking recall extremely important.

---

## Over-Aggressive Normalization

Normalization can remove useful information if it is too aggressive.

Therefore:

```text
Normalization ≠ Information Destruction
```

The goal is to remove irrelevant variation while preserving discriminative information.

---

## Similar Company Names

Two different companies may legitimately share very similar names.

For example:

```text
ABC Technologies
ABC Technologies India
```

Name similarity alone is therefore insufficient.

---

## Common Business Names

Names containing generic words such as:

```text
Global
India
Solutions
Technologies
Services
Enterprises
International
```

may produce high lexical similarity across unrelated companies.

Additional fields become important.

---

## Missing Fields

A record may contain:

```text
Name only
```

while another contains:

```text
Name + Address + Phone + Website
```

The pipeline must therefore be robust to incomplete records.

---

# 📁 Repository Structure

```text
amazon-ml-challenge-2026/
│
├── README.md
├── models/
│
└── code/
    └── business_entity_resolution/
        │
        ├── README.md
        ├── requirements.txt
        ├── Documentation_template.md
        │
        ├── configs/
        │   └── config.yaml
        │
        ├── preprocessing/
        │
        ├── hackathon_blocking/
        │   ├── src/
        │   │   └── blocking.py
        │   ├── run_blocking.py
        │   ├── requirements.txt
        │   └── README.md
        │
        ├── src/
        │   ├── blocking.py
        │   ├── cascade_ranker.py
        │   ├── contrastive_focal_loss.py
        │   ├── ditto_cross_encoder.py
        │   ├── fellegi_sunter_engine.py
        │   ├── multilingual_normalization.py
        │   ├── pipeline.py
        │   ├── train.py
        │   └── tripartite_graph_consensus.py
        │
        └── utils/
            └── validate_submission.py
```

---

# 🧩 Module Responsibility Map

## `multilingual_normalization.py`

Responsible for:

```text
Unicode normalization
Legal suffix handling
Punctuation processing
Whitespace normalization
Field preprocessing
```

---

## `blocking.py`

Responsible for:

```text
Candidate generation
Blocking keys
Token-based retrieval
Candidate pair construction
```

---

## `fellegi_sunter_engine.py`

Responsible for:

```text
Probabilistic comparison
m/u probabilities
Likelihood-ratio evidence
Intermediate linkage scoring
```

---

## `cascade_ranker.py`

Responsible for:

```text
Feature construction
Fuzzy matching features
LightGBM ranking
Confidence-based routing
```

---

## `ditto_cross_encoder.py`

Responsible for:

```text
Record serialization
Transformer input
BERT / DistilBERT inference
Pair classification
```

---

## `contrastive_focal_loss.py`

Responsible for:

```text
Custom loss computation
Class imbalance handling
Focal weighting
Hard-example emphasis
```

---

## `tripartite_graph_consensus.py`

Responsible for:

```text
Graph construction
Match edges
Connected components
Global consistency
Cluster formation
Conflict handling
```

---

## `pipeline.py`

Acts as the higher-level orchestration layer connecting the processing stages.

Conceptually:

```text
Input
 ↓
Normalize
 ↓
Block
 ↓
Score
 ↓
Rank
 ↓
Deep Match
 ↓
Graph Consensus
 ↓
Output
```

---

## `train.py`

Responsible for model-training workflows and associated training configuration.

---

## `validate_submission.py`

Used to validate the generated submission/output against the expected format before final submission.

---

# ⚙️ Configuration

The project keeps configuration separate from the core implementation.

Example configuration location:

```text
code/business_entity_resolution/configs/config.yaml
```

Configuration can be used to control items such as:

```text
Dataset paths
Model paths
Thresholds
Feature configuration
Training parameters
Transformer configuration
Inference settings
```

This keeps experiments reproducible without requiring source-code changes for every parameter adjustment.

---

# 🚀 Quickstart

## 1. Clone the Repository

```bash
git clone https://github.com/amulyarai24-art/amazon-ml-challenge-2026.git
cd amazon-ml-challenge-2026
```

---

## 2. Enter the Project

```bash
cd code/business_entity_resolution
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For the dedicated blocking implementation:

```bash
cd hackathon_blocking
pip install -r requirements.txt
```

---

# 🏋️ Model Training

From:

```text
code/business_entity_resolution/
```

the training workflow can be invoked using:

```bash
python src/train.py \
    --train_dir "../../dataset/train" \
    --model_dir "../../models"
```

The exact arguments should match the implementation currently present in the repository.

---

# 🔮 Inference Pipeline

The pipeline can be executed using the configured inference workflow:

```bash
python src/pipeline.py \
    --config configs/config.yaml
```

Conceptually, the pipeline performs:

```text
Load data
   ↓
Normalize
   ↓
Generate candidates
   ↓
Apply probabilistic evidence
   ↓
Generate ranking features
   ↓
Run cascade
   ↓
Escalate ambiguous pairs
   ↓
Run transformer
   ↓
Construct graph
   ↓
Resolve clusters
   ↓
Generate final output
```

---

# 🧪 Submission Validation

Before producing the final challenge submission, output validation should be performed.

The repository provides:

```text
utils/validate_submission.py
```

A validation layer is useful because a model can be correct while the generated submission is still invalid due to:

```text
Incorrect columns
Incorrect identifiers
Missing rows
Unexpected duplicates
Incorrect formatting
Invalid values
```

The validation step therefore acts as a final interface check between the ML system and the competition submission format.

---

# 📦 Data Policy

Large challenge datasets should remain local during development.

Raw challenge data should **not** be committed to GitHub.

Instead:

```text
.gitignore
```

should be used to prevent large/raw dataset files from being accidentally pushed.

The repository should contain:

```text
Code
Configuration
Documentation
Utilities
Model definitions
Training/inference logic
```

while large challenge datasets remain local.

This keeps the Git repository manageable and avoids unnecessarily publishing challenge data.

---

# 🔁 Reproducibility

A reproducible ML project should keep track of:

```text
Dataset version
Configuration
Feature definitions
Model architecture
Training parameters
Random seeds
Thresholds
Evaluation procedure
Dependency versions
```

The configuration file provides a central place for experiment parameters.

The codebase separates:

```text
Preprocessing
Blocking
Feature generation
Model training
Inference
Graph processing
Validation
```

which makes individual components easier to test and modify.

---

# 📊 Results

This repository should report **actual measured results only**.

Do not insert fabricated accuracy, F0.5, precision, recall, runtime, or leaderboard values.

When experiments are completed, results can be documented using a table such as:

| Experiment | Blocking | Ranker | Transformer | Graph | Precision | Recall | F0.5 |
|---|---|---|---|---|---:|---:|---:|
| Baseline | — | — | — | — | — | — | — |
| Experiment 1 | ✓ | ✓ | — | — | — | — | — |
| Experiment 2 | ✓ | ✓ | ✓ | — | — | — | — |
| Full Pipeline | ✓ | ✓ | ✓ | ✓ | — | — | — |

This makes the improvement contributed by each stage measurable.

---

# 🧪 Experimentation Framework

The architecture naturally supports ablation experiments.

For example:

### Experiment A

```text
Blocking
+
Basic similarity
```

### Experiment B

```text
Blocking
+
Fellegi-Sunter
+
LightGBM
```

### Experiment C

```text
Blocking
+
Fellegi-Sunter
+
LightGBM
+
Transformer
```

### Experiment D

```text
Full cascade
+
Graph consensus
```

Comparing these configurations can reveal which components provide meaningful improvements.

---

# 🔬 Recommended Error-Analysis Workflow

A practical improvement loop is:

```text
1. Run validation
       ↓
2. Collect false positives
       ↓
3. Collect false negatives
       ↓
4. Categorize errors
       ↓
5. Identify the failed stage
       ↓
6. Modify the relevant component
       ↓
7. Re-run validation
       ↓
8. Compare metrics
```

The important point is to avoid changing every component simultaneously.

If a false negative is caused by blocking, improving the transformer will not fix it because the transformer never receives the pair.

Likewise, if a false positive survives every pairwise stage but creates a bad cluster, the graph layer may be the appropriate place to investigate.

---

# 🧭 Stage-Level Debugging

The pipeline makes it possible to ask:

```text
Where did this decision go wrong?
```

For each pair:

```text
Raw Records
    ↓
Normalized Records
    ↓
Was candidate generated?
    ↓
Fellegi-Sunter evidence
    ↓
LightGBM score
    ↓
Transformer score
    ↓
Graph edge
    ↓
Final cluster
```

This makes debugging significantly easier than using a single end-to-end black-box model.

---

# 🏎️ Scalability Considerations

The central scalability principle is:

> **Do not spend expensive computation on pairs that can be rejected cheaply.**

Blocking handles candidate generation.

Probabilistic linkage provides inexpensive evidence.

LightGBM combines many structured signals efficiently.

The transformer is reserved for difficult cases.

Graph processing consolidates the final relationships.

Therefore the architecture is intentionally designed around:

```text
High recall in candidate generation
+
Efficient intermediate filtering
+
Deep reasoning for ambiguous cases
+
Global consistency at the end
```

---

# 🧠 Why Not Use Only BERT?

A transformer is powerful, but using it for every possible pair is expensive.

Suppose a dataset contains a large number of records.

The number of possible record pairs grows rapidly.

Running:

```text
Record A + Record B
```

through a transformer for every possible combination would create unnecessary computational cost.

The cascade instead does:

```text
Millions of possible pairs
        ↓
Blocking
        ↓
Much smaller candidate set
        ↓
Probabilistic filtering
        ↓
Smaller candidate set
        ↓
LightGBM
        ↓
Ambiguous subset
        ↓
Transformer
```

This is the central engineering idea behind the system.

---

# 🧠 Why Not Use Only String Similarity?

String similarity is useful but limited.

Consider:

```text
ABC Technologies Pvt Ltd
```

and:

```text
ABC Technology Private Limited
```

The strings are not identical, but may represent the same company.

Conversely:

```text
ABC Technologies
```

and:

```text
ABC Technologies India
```

may have extremely high textual similarity while representing different businesses.

Therefore:

```text
String similarity
≠
Entity identity
```

Entity resolution requires multiple signals and context.

---

# 🧠 Why Add Probabilistic Linkage?

Probabilistic linkage provides an interpretable intermediate layer.

Instead of simply saying:

```text
Similarity = 0.87
```

the system can reason in terms of:

```text
How frequently does this type of agreement occur
among true matches?

How frequently does it occur among non-matches?
```

This provides statistical evidence that can complement the ML models.

---

# 🧠 Why Add LightGBM?

LightGBM is well suited to structured tabular features.

The entity matching problem naturally produces features such as:

```text
Name similarity
Address similarity
Phone equality
Postal-code equality
Website similarity
Token overlap
Length difference
Probabilistic score
```

Gradient-boosted trees can learn nonlinear relationships between these features.

For example:

```text
High name similarity
+
Same postal code
+
Same website domain
```

may provide substantially stronger evidence than:

```text
High name similarity
```

alone.

---

# 🧠 Why Add a Transformer?

Traditional similarity measures mainly operate on surface-level relationships.

Transformers provide contextual representations and direct interaction between the two records.

This is particularly valuable for:

```text
Complex name variations
Multilingual records
Different word order
Contextual relationships
Partial information
Semantically similar representations
```

The transformer therefore serves as the high-capacity reasoning stage for difficult cases.

---

# 🧠 Why Add Graph Consensus?

Entity resolution ultimately produces groups of records representing entities.

This is not purely a collection of independent pairwise decisions.

The graph stage allows the system to transition from:

```text
Pair-level predictions
```

to:

```text
Entity-level structure
```

This is important because the final objective is often to identify consistent entity groups rather than merely label isolated record pairs.

---

# 🧩 System-Level View

The complete architecture can be summarized as:

```text
                DATA
                 │
                 ▼
       ┌──────────────────┐
       │ NORMALIZATION     │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ BLOCKING          │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ FELLEGI-SUNTER    │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ LIGHTGBM          │
       └────────┬─────────┘
                │
         ┌──────┴──────┐
         │             │
      Certain       Ambiguous
         │             │
         │             ▼
         │      ┌───────────────┐
         │      │ DITTO         │
         │      │ TRANSFORMER   │
         │      └───────┬───────┘
         │              │
         └──────┬───────┘
                │
                ▼
       ┌──────────────────┐
       │ GRAPH CONSENSUS   │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ ENTITY CLUSTERS   │
       └──────────────────┘
```

---

# 👥 Team Development Workflow

The repository is structured so that multiple team members can work simultaneously.

A practical division of responsibilities is:

```text
Member 1
→ Preprocessing + Normalization

Member 2
→ Blocking + Candidate Generation

Member 3
→ Ranking + Transformer

Member 4
→ Graph Consensus + Pipeline Integration
```

Each member can work on their component independently.

The final integration happens through Git.

---

## Recommended Git Workflow

Each member should:

```bash
git pull
```

create or switch to their working branch:

```bash
git checkout -b feature/<component-name>
```

make changes:

```bash
git add .
git commit -m "Implement <component>"
```

and push:

```bash
git push origin feature/<component-name>
```

The branch can then be reviewed and merged into the main branch.

---

# 🔀 Integration Strategy

The most important rule for team development is:

```text
Do not independently create conflicting versions of the same files.
```

Each component should have a clearly defined interface.

For example:

```text
Blocking
    ↓
candidate_pairs

Fellegi-Sunter
    ↓
probabilistic_features

LightGBM
    ↓
ranked_candidates

Transformer
    ↓
deep_match_scores

Graph
    ↓
entity_clusters
```

This makes integration much easier.

---

# 📚 Documentation Map

The repository contains documentation at multiple levels.

## Root README

```text
README.md
```

Provides the complete architecture and project overview.

---

## Project README

```text
code/business_entity_resolution/README.md
```

Provides project-level implementation information.

---

## Blocking Documentation

```text
code/business_entity_resolution/hackathon_blocking/README.md
```

Documents the dedicated blocking implementation.

---

## Configuration

```text
code/business_entity_resolution/configs/config.yaml
```

Contains configurable experiment and pipeline parameters.

---

## Documentation Template

```text
code/business_entity_resolution/Documentation_template.md
```

Provides the structure for implementation/documentation details.

---

# 🚀 Future Improvements

The architecture can be extended in several directions.

## 1. Better Multilingual Models

A multilingual transformer could improve cross-language and transliteration matching.

---

## 2. Learned Blocking

Instead of relying entirely on manually defined blocking keys, learned retrieval methods could generate candidates using semantic representations.

---

## 3. Approximate Nearest Neighbor Retrieval

Embedding-based retrieval could be used for scalable semantic candidate generation.

---

## 4. Better Hard-Negative Mining

Transformer training could focus on difficult non-match pairs that look highly similar.

Example:

```text
Same company name
Different address
Different website
```

These are much more informative than obviously unrelated pairs.

---

## 5. Threshold Calibration

Instead of choosing thresholds manually, thresholds can be optimized using validation data.

---

## 6. Field-Specific Models

Different fields may benefit from specialized preprocessing or similarity functions.

For example:

```text
Name
→ token + semantic similarity

Address
→ component-aware similarity

Phone
→ normalized exact comparison

Email
→ local-part + domain comparison

Website
→ domain normalization
```

---

## 7. Better Graph Optimization

Graph-based clustering can be enhanced with:

```text
Edge confidence
Cluster consistency
Conflict penalties
Community detection
Constraint-based clustering
```

---

# 🧪 Experimental Ablation Plan

A strong experimentation plan can evaluate the architecture incrementally.

```text
Baseline
   ↓
+ Normalization
   ↓
+ Blocking
   ↓
+ Fellegi-Sunter
   ↓
+ LightGBM
   ↓
+ Transformer
   ↓
+ Graph Consensus
```

For every experiment, record:

```text
Configuration
Dataset split
Precision
Recall
F0.5
Runtime
Candidate count
Transformer invocation count
Error categories
```

This makes it possible to understand both:

```text
Accuracy improvement
```

and:

```text
Computational trade-offs
```

---

# 📌 Project Goal

The goal of this project is to construct a robust and scalable business entity resolution framework that can transform noisy, heterogeneous business records into consistent entity groups.

The architecture combines:

```text
Multilingual normalization
        +
Candidate blocking
        +
Probabilistic linkage
        +
Gradient-boosted ranking
        +
Transformer-based semantic matching
        +
Graph-based global consensus
```

The central philosophy is:

> **Use the cheapest reliable evidence first, reserve expensive models for ambiguity, and use global structure to make the final entity assignments consistent.**

---

# 🏁 Final Architecture Summary

The complete system can be summarized in five major matching stages plus preprocessing:

```text
┌──────────────────────────────────────────────────────────────┐
│                    BUSINESS ENTITY DATA                     │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 0 — MULTILINGUAL NORMALIZATION                        │
│                                                              │
│ Unicode • Suffixes • Punctuation • Whitespace • Fields      │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 1 — CANDIDATE BLOCKING                                │
│                                                              │
│ Exact Tokens • Geographic Buckets • Token Overlap           │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 2 — FELLEGI-SUNTER PROBABILISTIC LINKAGE              │
│                                                              │
│ Match Probability • Non-Match Probability • LLR Evidence    │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 3 — LIGHTGBM CASCADE RANKER                           │
│                                                              │
│ Fuzzy Similarity • Structural Features • Cross-Field Data   │
└────────────────────────────┬─────────────────────────────────┘
                             │
                       Ambiguous Pairs
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 4 — DITTO TRANSFORMER CROSS-ENCODER                   │
│                                                              │
│ BERT / DistilBERT • Serialized Pairs • Cross-Attention      │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│ STAGE 5 — TRIPARTITE GRAPH CONSENSUS                        │
│                                                              │
│ Match Edges • Connected Components • Global Consistency     │
└────────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────┐
│                  FINAL ENTITY CLUSTERS                       │
│                                                              │
│        Records resolved to their real-world entities         │
└──────────────────────────────────────────────────────────────┘
```

---

# ⚡ Final Takeaway

This project is designed around a simple but powerful idea:

```text
Do not ask a single model to solve everything.
```

Instead:

```text
Normalize the data.
       ↓
Find plausible candidates.
       ↓
Measure probabilistic evidence.
       ↓
Rank using structured ML features.
       ↓
Use deep semantic reasoning only when necessary.
       ↓
Use graph structure to enforce global consistency.
       ↓
Produce final entity groups.
```

This creates a modular architecture where each stage has a clearly defined responsibility.

The result is a system designed to balance:

```text
⚡ Scalability
🎯 Precision
🧠 Semantic understanding
🔍 Interpretability
🌍 Multilingual robustness
🕸️ Global consistency
🔧 Modularity
🚀 Practical deployment efficiency
```

**Business Entity Resolution Engine — Amazon ML Challenge 2026** 🚀