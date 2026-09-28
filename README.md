@'
# Business Entity Matching

A machine learning pipeline for matching business entities across multiple data sources. The project combines data normalization, blocking, feature engineering, candidate generation, and XGBoost-based classification to identify records that represent the same real-world business.

## Problem

The goal is to identify whether records from different business data sources refer to the same entity.

The system works with:

- **Source 1 (S1)** — reference entities
- **Source 2 (S2)** — candidate entities
- **Source 3 (S3)** — additional candidate entities

Because the datasets contain millions of records, comparing every possible pair is computationally infeasible. The pipeline therefore uses blocking to efficiently generate candidate pairs before applying machine learning.

## Pipeline

```text
Raw Data
   |
   v
Data Audit
   |
   v
Normalization
   |
   v
Blocking / Candidate Generation
   |
   v
Feature Engineering
   |
   v
XGBoost Model
   |
   v
Prediction & Calibration
   |
   v
Final Entity Matches
