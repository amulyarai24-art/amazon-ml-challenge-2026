# ⚡ Business Entity Resolution Engine 🏢🔍
> **Amazon ML Challenge 2026** | A high-throughput, multi-stage Machine Learning framework for large-scale record linkage, company entity matching, and cross-source record disambiguation. 🚀✨

---

## 🧭 System Architecture & Design 🛠️

In massive real-world datasets, checking every single company record against every other record creates an $O(N^2)$ computational bottleneck 🐢. Our pipeline solves this scalability challenge by using a **4-stage cascading architecture** 🌊. It filters out obvious non-matches in early high-speed stages while reserving powerful deep learning models for hard edge cases.

```text
 ┌──────────────────────────────────────────────────────────┐
 │               🏢 Raw Business Records Dataset             │
 └─────────────────────────────┬────────────────────────────┘
                               │
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 🎯 Stage 1: Deterministic & Multilingual Blocking        │
 │ • Filters candidate pairs by postal buckets & name keys   │
 │ • Reduces search space dramatically from O(N²) to O(N)   │
 └─────────────────────────────┬────────────────────────────┘
                               │  (Candidate Pairs 📑)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 📊 Stage 2: Probabilistic Fellegi-Sunter Weighting       │
 │ • Field-level agreement vs. disagreement log-likelihoods  │
 │ • Assigns initial baseline match scores                  │
 └─────────────────────────────┬────────────────────────────┘
                               │  (Filtered Candidates ⚖️)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 🌲 Stage 3: LightGBM Cascade Ranker                      │
 │ • Computes Jaro-Winkler, Levenshtein, & RapidFuzz ratios │
 │ • Gradient Boosted Decision Trees for fast classification│
 └─────────────────────────────┬────────────────────────────┘
                               │  (Hard / Unresolved Pairs 🤔)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 🤖 Stage 4: Ditto Transformer Cross-Encoder              │
 │ • Deep BERT fine-tuning with full sequence cross-attention│
 │ • Asymmetric Macro-F0.5 Focal Loss for precision balance │
 └─────────────────────────────┬────────────────────────────┘
                               │  (High-Confidence Links 🔗)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 🕸️ Stage 5: Tripartite Graph Consensus                   │
 │ • Connected component analysis & equivalence clustering  │
 └─────────────────────────────┬────────────────────────────┘
                               │
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │             🏆 Final Consolidated Match Output            │
 └─────────────────────────────┬────────────────────────────┘
🛠️ Core Tech Stack & Libraries 💻
Module Category	Tools & Technologies Included	Purpose & Application
Deep Learning Engine 🤖	PyTorch, HuggingFace Transformers, BERT / DistilBERT	Deep semantic cross-attention matching for ambiguous entity pairs
Gradient Boosting 🌲	LightGBM, Scikit-Learn	Ultra-fast tabular ranking over string distance feature vectors
String Metrics & Wrangling ⚡	RapidFuzz, Polars, Pandas	High-performance fuzzy matching, Levenshtein, & token-sorting
Graph Clustering 🕸️	NetworkX	Global transitive closure and connected component identification
Pipeline & Tools ⚙️	PyYAML, Custom Python CLI	Orchestration, configuration parsing, and artifact saving
⚙️ Exhaustive Technical Module Breakdown & Working Principles 📖
1️⃣ Multilingual Text Normalization & Cleaning (multilingual_normalization.py) 🧹
Purpose: Standardizes raw, noisy text across diverse character sets, languages, and formatting variations.

Working Principle:

Performs Unicode normalization (NFKD) to convert accented characters (e.g., é → e).

Normalizes legal business entity suffixes across global contexts (e.g., GmbH, Inc, LLC, Ltd, Corp → standardized tokens).

Strips special punctuation while preserving crucial alphanumeric codes (such as tax IDs or registration numbers).

2️⃣ Candidate Pair Blocking Engine (blocking.py) 🎯
Purpose: Solves the O(N 
2
 ) candidate explosion by pruning non-matching pairs before feature extraction.

Working Principle:

Exact Token Block: Hashes normalized core brand tokens to group similar business names into the same candidate buckets.

Multilingual Postal & Geolocation Bucket: Groups records sharing identical or regional postal prefixes.

Token Overlap Search: Uses sparse inverted indices to rapidly retrieve pairs sharing at least k overlapping words.

3️⃣ Fellegi-Sunter Probabilistic Linkage (fellegi_sunter_engine.py) 📊
Purpose: Applies classical probabilistic decision theory to compute baseline log-likelihood log-odds.

Mathematical Formulation:

Let m 
i
​
 =P(γ 
i
​
 ∣Match) be the probability that field i agrees given the records represent the same entity.

Let u 
i
​
 =P(γ 
i
​
 ∣Non-Match) be the probability that field i agrees purely by chance.

The weight ratio vector is calculated as:

Match Weight= 
i
∑
​
 log 
2
​
 ( 
u 
i
​
 
m 
i
​
 
​
 )
Pairs with weights above a dynamic threshold pass directly to Stage 3, while low-weight pairs are pruned immediately.

4️⃣ Stage 1: LightGBM Cascade Ranker (cascade_ranker.py) 🌲
Purpose: Evaluates tabular string similarity vectors using fast Gradient Boosted Decision Trees (GBDT).

Extracted Feature Vector:

Fuzzy Ratios: RapidFuzz Ratio, Partial Ratio, Token Sort Ratio, Token Set Ratio.

Distance Metrics: Normalized Levenshtein distance and Jaro-Winkler similarity.

Structural Matches: Binary agreement flags on registration numbers, postal codes, and city names.

Decision Cascade Thresholds:

Score≥T 
high
​
  (e.g., 0.85) → Accepted as Match.

Score<T 
low
​
  (e.g., 0.20) → Rejected as Non-Match.

T 
low
​
 ≤Score<T 
high
​
  → Escalated to Stage 2 Deep Learning Cross-Encoder.

5️⃣ Stage 2: Ditto Transformer Cross-Encoder (ditto_cross_encoder.py) 🤖
Purpose: Resolves complex, fine-grained semantic ambiguities where fuzzy string metrics fail.

Working Principle:

Formats candidate records into a serialized sequence:
[CLS] name: Acquired Tech | addr: 100 Main St [SEP] name: Acquired Tech Inc | addr: 100 Main Street [SEP]

Passes the concatenated sequence through a pre-trained BERT/DistilBERT architecture to allow deep cross-attention across all token pairs.

Outputs a continuous probability score P(match∣pair).

6️⃣ Custom Asymmetric Macro-F 
0.5
​
  Focal Loss (contrastive_focal_loss.py) ⚖️
Purpose: Addresses severe class imbalance while heavily prioritizing Precision (F 
0.5
​
 ) to prevent incorrect entity mergers.

Mathematical Formulation:

L 
Focal
​
 =−α 
t
​
 (1−p 
t
​
 ) 
γ
 log(p 
t
​
 )
Asymmetric Weighting: Applies higher penalty multipliers α 
t
​
  on False Positives, ensuring the model prefers missing an ambiguous link over creating a wrong entity merge.

Macro-F 
0.5
​
  Optimization: Directly optimizes model gradients toward high precision (β=0.5 in F 
β
​
 ), giving precision twice the weight of recall.

7️⃣ Tripartite Graph Consensus & Clustering (tripartite_graph_consensus.py) 🕸️
Purpose: Merges pairwise match predictions across multiple input sources into unified global entity clusters.

Working Principle:

Constructs an undirected graph G=(V,E) where nodes V are individual records and edges E are high-confidence match predictions.

Applies Connected Component Analysis and Transitive Graph Closure to resolve cross-source equivalence (A=B and B=C⟹A=C).

Resolves conflicting cluster overlaps using graph density constraints.

📁 Complete Repository Layout 📂
Plaintext
amazon-ml-challenge-2026/
├── 📑 configs/
│   └── config.yaml                   # Global hyperparameters, thresholds, & file paths
├── 📂 dataset/
│   ├── train/                        # Raw training datasets (TSV format)
│   └── test/                         # Input evaluation datasets
├── 💾 models/                           # Saved weights, GBDT models, & calibration params
├── 📊 output/                           # Candidate pairs, prediction logs, & final outputs
├── 🐍 src/
│   ├── blocking.py                   # Candidate pair generator
│   ├── cascade_ranker.py             # Feature extractor & LightGBM runner
│   ├── contrastive_focal_loss.py     # Custom asymmetric focal loss function
│   ├── ditto_cross_encoder.py        # BERT transformer model architecture
│   ├── fellegi_sunter_engine.py      # Probabilistic scoring engine
│   ├── multilingual_normalization.py # Text cleaner & script normalizer
│   ├── pipeline.py                   # End-to-end inference runner
│   ├── train.py                      # Master training pipeline runner
│   └── tripartite_graph_consensus.py # Graph clustering engine
├── 🛠️ utils/
│   └── validate_submission.py          # Submission validator script
├── 📄 README.md                      # Comprehensive project documentation
└── 📦 requirements.txt               # Dependencies list
🚀 Quickstart Guide & Terminal Commands 🏁
1. Install Required Dependencies 📦
Bash
pip install -r requirements.txt
2. Run Master Model Training 🏋️‍♂️
Executes full model training. If local TSV files are missing in dataset/train/, it automatically executes an end-to-end synthetic verification run:

Bash
cd code/business_entity_resolution
python src/train.py --train_dir "../../dataset/train" --model_dir "../../models"
3. Run Inference & Predictions Pipeline 🔮
Executes candidate blocking, GBDT ranking, transformer reranking, and global graph clustering to generate final submission outputs:

Bash
python src/pipeline.py --config configs/config.yaml