import os
import pandas as pd
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\adity\hackathon"

GT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train",
    "train",
    "train_ground_truth.tsv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

BLOCKING_FILE = os.path.join(
    PROCESSED_DIR,
    "blocking_pilot_s1_s3_v4.csv"
)


# ============================================================
# LOAD BLOCKING FILE
# ============================================================

print("=" * 70)
print("CORRECT PILOT BLOCKING EVALUATION")
print("=" * 70)

print("\nLoading blocking file...")

blocking = pd.read_csv(
    BLOCKING_FILE,
    dtype=str
).fillna("")

pilot_s1_ids = set(
    blocking["source1_entity_id"]
)

print(
    f"Pilot S1 entities: {len(pilot_s1_ids):,}"
)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\nLoading ground truth...")

gt = pd.read_csv(
    GT_FILE,
    sep="\t",
    dtype=str
).fillna("")

gt = gt[
    gt["source1_entity_id"].isin(pilot_s1_ids)
].copy()

print(
    f"Ground-truth rows for pilot: {len(gt):,}"
)


# ============================================================
# BUILD S1 -> S3 GROUND TRUTH
# ============================================================

print("\nBuilding S1 -> S3 ground-truth lookup...")

gt_lookup = {}

total_gt_pairs = 0
s1_with_s3_gt = 0

for row in gt.itertuples(index=False):

    s1_id = row.source1_entity_id
    raw_matches = row.matched_entity_ids

    # --------------------------------------------------------
    # Ground truth contains BOTH S2 and S3 IDs.
    # We only evaluate S2 here.
    # --------------------------------------------------------

    matches = {
        entity_id.strip()
        for entity_id in raw_matches.split(",")
        if entity_id.strip().startswith("S3-")
    }

    gt_lookup[s1_id] = matches

    if matches:
        s1_with_s3_gt += 1
        total_gt_pairs += len(matches)


# ============================================================
# EVALUATE BLOCKING
# ============================================================

print("\nEvaluating candidates...")

recovered_pairs = 0
total_candidate_pairs = 0

entities_with_recovery = 0
entities_with_all_matches = 0
entities_with_partial_recovery = 0

candidate_counts = []


for row in tqdm(
    blocking.itertuples(index=False),
    total=len(blocking),
    desc="Evaluating"
):

    s1_id = row.source1_entity_id

    # --------------------------------------------------------
    # Parse blocking candidates
    # --------------------------------------------------------

    candidate_string = row.candidate_entity_ids

    if candidate_string:

        candidates = {
            x.strip()
            for x in candidate_string.split("|")
            if x.strip()
        }

    else:
        candidates = set()

    candidate_count = len(candidates)

    total_candidate_pairs += candidate_count
    candidate_counts.append(candidate_count)

    # --------------------------------------------------------
    # Ground truth S2 matches
    # --------------------------------------------------------

    gt_matches = gt_lookup.get(
        s1_id,
        set()
    )

    if not gt_matches:
        continue

    # --------------------------------------------------------
    # Intersection
    # --------------------------------------------------------

    recovered = candidates.intersection(
        gt_matches
    )

    recovered_count = len(recovered)

    recovered_pairs += recovered_count

    if recovered_count > 0:
        entities_with_recovery += 1

    if recovered_count == len(gt_matches):
        entities_with_all_matches += 1

    elif recovered_count > 0:
        entities_with_partial_recovery += 1


# ============================================================
# METRICS
# ============================================================

pair_recall = (
    recovered_pairs / total_gt_pairs
    if total_gt_pairs
    else 0
)

entity_recall = (
    entities_with_recovery / s1_with_s3_gt
    if s1_with_s3_gt
    else 0
)

all_match_rate = (
    entities_with_all_matches / s1_with_s3_gt
    if s1_with_s3_gt
    else 0
)

avg_candidates = (
    total_candidate_pairs / len(blocking)
    if len(blocking)
    else 0
)

avg_gt_matches = (
    total_gt_pairs / s1_with_s3_gt
    if s1_with_s3_gt
    else 0
)

avg_recovered = (
    recovered_pairs / len(blocking)
    if len(blocking)
    else 0
)


# ============================================================
# OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("CORRECT PILOT BLOCKING RESULTS")
print("=" * 70)

print(
    f"\nPilot S1 entities                 : "
    f"{len(blocking):,}"
)

print(
    f"S1 entities with S3 GT matches   : "
    f"{s1_with_s3_gt:,}"
)

print(
    f"Total S3 GT pairs                : "
    f"{total_gt_pairs:,}"
)

print(
    f"Total candidate pairs            : "
    f"{total_candidate_pairs:,}"
)

print(
    f"Recovered S3 GT pairs            : "
    f"{recovered_pairs:,}"
)

print(
    f"\nPAIR-LEVEL BLOCKING RECALL       : "
    f"{pair_recall:.4%}"
)

print(
    f"\nS1 with >=1 recovered match      : "
    f"{entities_with_recovery:,}"
)

print(
    f"ENTITY-LEVEL RECALL              : "
    f"{entity_recall:.4%}"
)

print(
    f"\nS1 with ALL S2 matches recovered : "
    f"{entities_with_all_matches:,}"
)

print(
    f"ALL-MATCH RATE                   : "
    f"{all_match_rate:.4%}"
)

print(
    f"\nS1 with partial recovery         : "
    f"{entities_with_partial_recovery:,}"
)

print(
    f"Average candidates per S1        : "
    f"{avg_candidates:.2f}"
)

print(
    f"Average S3 GT matches per S1     : "
    f"{avg_gt_matches:.2f}"
)

print(
    f"Average recovered matches per S1 : "
    f"{avg_recovered:.2f}"
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

