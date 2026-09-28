import os
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE = r"C:\Users\adity\hackathon"

GT_FILE = os.path.join(
    BASE,
    "data", "raw", "train", "train",
    "train_ground_truth.tsv"
)

FEATURE_FILE = os.path.join(
    BASE,
    "data", "processed",
    "features_pilot_s1_s2_v3.csv"
)

BLOCKING_FILE = os.path.join(
    BASE,
    "data", "processed",
    "blocking_pilot_s1_s2_v3.csv"
)

CHUNK_SIZE = 250_000


print("=" * 70)
print("FAST FEATURE-STAGE RECALL EVALUATION")
print("=" * 70)


# ============================================================
# PILOT S1 IDS
# ============================================================

print("\nLoading pilot Source-1 IDs...")

pilot_s1 = pd.read_csv(
    BLOCKING_FILE,
    usecols=["source1_entity_id"],
    dtype=str
)

pilot_s1_ids = set(
    pilot_s1["source1_entity_id"]
)

print(
    f"Pilot S1 entities: {len(pilot_s1_ids):,}"
)

del pilot_s1


# ============================================================
# GROUND TRUTH
# ============================================================

print("\nReading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str,
    usecols=[
        "source1_entity_id",
        "matched_entity_ids"
    ]
)

gt = gt[
    gt["source1_entity_id"].isin(pilot_s1_ids)
].copy()

print(
    f"Pilot ground-truth rows: {len(gt):,}"
)


# ============================================================
# EXPLODE S2 MATCHES
# ============================================================

print("\nExtracting S2 ground-truth pairs...")

gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)

gt["s2_matches"] = gt[
    "matched_entity_ids"
].str.split(",")

gt = gt[
    ["source1_entity_id", "s2_matches"]
].explode(
    "s2_matches"
)

gt["source2_entity_id"] = (
    gt["s2_matches"]
    .fillna("")
    .str.strip()
)

gt = gt[
    gt["source2_entity_id"].str.startswith(
        "S2-"
    )
].copy()

gt = gt[
    [
        "source1_entity_id",
        "source2_entity_id"
    ]
].drop_duplicates()

print(
    f"S2 ground-truth pairs: {len(gt):,}"
)

gt_s1 = set(
    gt["source1_entity_id"]
)


# ============================================================
# FEATURE FILE
# ============================================================

print("\nScanning feature candidates...")

recovered_pairs = 0

recovered_s1 = set()

feature_rows = 0

# Keep only IDs — don't load the 314 MB file at once
usecols = [
    "source1_entity_id",
    "source2_entity_id"
]

for chunk in pd.read_csv(
    FEATURE_FILE,
    dtype=str,
    usecols=usecols,
    chunksize=CHUNK_SIZE
):

    feature_rows += len(chunk)

    # --------------------------------------------------------
    # Vectorized inner join against ground truth
    # --------------------------------------------------------

    matches = chunk.merge(
        gt,
        on=[
            "source1_entity_id",
            "source2_entity_id"
        ],
        how="inner"
    )

    if not matches.empty:

        recovered_pairs += len(matches)

        recovered_s1.update(
            matches[
                "source1_entity_id"
            ].unique()
        )

    del chunk
    del matches


# ============================================================
# PAIR RECALL
# ============================================================

total_gt_pairs = len(gt)

pair_recall = (
    recovered_pairs / total_gt_pairs
    if total_gt_pairs
    else 0
)


# ============================================================
# ENTITY RECALL
# ============================================================

entity_recall = (
    len(recovered_s1) / len(gt_s1)
    if gt_s1
    else 0
)


# ============================================================
# ALL-MATCH RECOVERY
# ============================================================

print("\nCalculating all-match recovery...")

# Re-read feature IDs and keep only GT matches.
# This is still vectorized and memory controlled.

recovered = []

for chunk in pd.read_csv(
    FEATURE_FILE,
    dtype=str,
    usecols=usecols,
    chunksize=CHUNK_SIZE
):

    matches = chunk.merge(
        gt,
        on=[
            "source1_entity_id",
            "source2_entity_id"
        ],
        how="inner"
    )

    if not matches.empty:
        recovered.append(matches)

    del chunk
    del matches


if recovered:

    recovered_df = pd.concat(
        recovered,
        ignore_index=True
    )

else:

    recovered_df = pd.DataFrame(
        columns=[
            "source1_entity_id",
            "source2_entity_id"
        ]
    )


# ------------------------------------------------------------
# Count true vs recovered matches per S1
# ------------------------------------------------------------

gt_counts = (
    gt.groupby(
        "source1_entity_id"
    )
    .size()
    .rename("gt_count")
)

recovered_counts = (
    recovered_df.groupby(
        "source1_entity_id"
    )
    .size()
    .rename("recovered_count")
)

comparison = pd.concat(
    [
        gt_counts,
        recovered_counts
    ],
    axis=1
).fillna(0)

comparison["all_recovered"] = (
    comparison["recovered_count"]
    >=
    comparison["gt_count"]
)

all_match_count = int(
    comparison["all_recovered"].sum()
)

all_match_rate = (
    all_match_count / len(gt_counts)
    if len(gt_counts)
    else 0
)

partial_count = int(
    (
        (comparison["recovered_count"] > 0)
        &
        (
            comparison["recovered_count"]
            <
            comparison["gt_count"]
        )
    ).sum()
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FEATURE-STAGE RECALL RESULTS")
print("=" * 70)

print(
    f"Feature rows evaluated: "
    f"{feature_rows:,}"
)

print(
    f"S1 entities with S2 GT: "
    f"{len(gt_s1):,}"
)

print(
    f"Total S2 GT pairs: "
    f"{total_gt_pairs:,}"
)

print(
    f"Recovered GT pairs: "
    f"{recovered_pairs:,}"
)

print(
    f"Pair-level feature recall: "
    f"{pair_recall:.4%}"
)

print(
    f"S1 entities with >=1 recovered: "
    f"{len(recovered_s1):,}"
)

print(
    f"Entity-level feature recall: "
    f"{entity_recall:.4%}"
)

print(
    f"S1 entities with ALL matches recovered: "
    f"{all_match_count:,}"
)

print(
    f"All-match recovery rate: "
    f"{all_match_rate:.4%}"
)

print(
    f"Partial recovery: "
    f"{partial_count:,}"
)

print("=" * 70)