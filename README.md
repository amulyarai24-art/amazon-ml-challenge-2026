# ⚡ Business Entity Resolution Engine
> **Amazon ML Challenge 2026** | Ultra-scalable multi-stage entity resolution, record linkage, and cross-source disambiguation pipeline.

---

## 🧭 System Architecture & Design

The engine solves large-scale entity resolution by cascading from fast probabilistic filtering down to deep semantic transformer classification.

```text
 ┌─────────────────────────────────────────────────────────┐
 │                   Raw Business Records                  │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 1: Deterministic & Multilingual Blocking          │
 │ • Exact name blocks & token overlap                     │
 │ • Geographic & postal code grouping                     │
 └────────────────────────────┬────────────────────────────┘
                              │  (Candidate Pairs)
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 2: Probabilistic Fellegi-Sunter Weighting         │
 │ • Log-likelihood ratio calculation on string fields     │
 │ • Initial candidate pair pruning                        │
 └────────────────────────────┬────────────────────────────┘
                              │  (Filtered Candidates)
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 3: LightGBM Cascade Ranker                       │
 │ • Extracts Jaro-Winkler, Levenshtein, & RapidFuzz ratios │
 │ • High-throughput Gradient Boosted Decision Tree ranking│
 └────────────────────────────┬────────────────────────────┘
                              │  (Hard / Unresolved Pairs)
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 4: Ditto Transformer Cross-Encoder                │
 │ • Fine-tuned BERT / DistilBERT cross-attention          │
 │ • Macro-F0.5 Asymmetric Focal Loss optimization          │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │ Stage 5: Tripartite Graph Consensus                    │
 │ • Connected components & global equivalence clustering  │
 └────────────────────────────┬────────────────────────────┘
                              │
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │                Final Submissions Output                 │
 └─────────────────────────────────────────────────────────┘