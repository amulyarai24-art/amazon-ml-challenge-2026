# Amazon ML Challenge 2026 — Business Entity Resolution

A machine learning solution for the **Amazon ML Challenge 2026** Business Entity Resolution problem.

The objective is to identify records from multiple data sources that refer to the same real-world business entity, despite differences in names, addresses, formatting, spelling, and other noisy attributes.

---

## 📌 Problem Statement

The challenge involves matching business records across multiple independent data sources.

The same business may appear differently across sources because of:

- Spelling variations
- Abbreviations
- Typos
- Different formatting
- Missing information
- Address variations
- Word-order differences
- Multilingual text
- Transliteration
- Noisy or incomplete business information

The goal is to determine which records correspond to the same underlying business entity.

---

## 🎯 Objective

Build a robust and scalable **Business Entity Resolution** pipeline that can:

1. Process and normalize noisy business records.
2. Generate likely candidate matches efficiently.
3. Extract useful similarity features.
4. Predict whether candidate records represent the same entity.
5. Produce the required output files in the challenge format.

The solution is designed with particular attention to **precision**, since the competition uses the **Macro F0.5** evaluation metric.

---

## 📊 Evaluation Metric

The primary evaluation metric is **Macro F0.5**.

F0.5 gives more importance to precision than recall:

\[
F_{0.5} =
\frac{1.25 \times Precision \times Recall}
{0.25 \times Precision + Recall}
\]

This makes precision especially important during the final matching stage.

The system therefore aims to:

- Maintain high recall during candidate generation.
- Maintain high precision during final matching.
- Avoid incorrect entity merges.
- Correctly handle records with no corresponding match.

---

## 🧠 Solution Pipeline

The overall pipeline follows a multi-stage entity-resolution approach:

```text
                 Raw Data
                    │
                    ▼
        ┌──────────────────────┐
        │ Data Preprocessing   │
        │ & Normalization      │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Candidate Generation │
        │ / Blocking           │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Similarity Features  │
        │ & Feature Engineering│
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Matching Model       │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Final Matching &     │
        │ Output Generation    │
        └──────────────────────┘