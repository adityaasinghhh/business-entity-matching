import os
import pandas as pd
import numpy as np
from tqdm import tqdm

BASE = r"C:\Users\adity\hackathon"

BLOCK_FILE = os.path.join(
    BASE, "data", "processed",
    "blocking_pilot_s1_s3_v4.csv"
)

OUT_FILE = os.path.join(
    BASE, "outputs",
    "s1_s3_predictions.csv"
)

TOP_N = 12

print("=" * 70)
print("LIGHTWEIGHT S1 -> S3 SCORING")
print("=" * 70)

df = pd.read_csv(
    BLOCK_FILE,
    dtype={
        "source1_entity_id": "string",
        "candidate_entity_ids": "string"
    }
)

print(f"Blocking rows: {len(df):,}")

rows = []

for row in tqdm(
    df.itertuples(index=False),
    total=len(df),
    desc="Selecting S3 candidates"
):
    s1 = row.source1_entity_id
    raw = row.candidate_entity_ids

    if pd.isna(raw) or not raw:
        continue

    candidates = [
        x.strip()
        for x in str(raw).split("|")
        if x.strip().startswith("S3-")
    ]

    if not candidates:
        continue

    # V4 blocker already orders candidates by blocking strength.
    # Preserve that ordering and retain a recall-oriented top-N.
    selected = candidates[:TOP_N]

    for rank, s3 in enumerate(selected, start=1):
        rows.append(
            (s1, s3, rank)
        )

out = pd.DataFrame(
    rows,
    columns=[
        "source1_entity_id",
        "source3_entity_id",
        "blocking_rank"
    ]
)

# Keep one candidate pair only.
out = out.drop_duplicates(
    ["source1_entity_id", "source3_entity_id"]
)

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

out.to_csv(
    OUT_FILE,
    index=False
)

print()
print("=" * 70)
print("S3 SCORING COMPLETE")
print("=" * 70)
print(f"Prediction pairs : {len(out):,}")
print(f"S1 entities      : {out.source1_entity_id.nunique():,}")
print(f"Output           : {OUT_FILE}")
print("=" * 70)
