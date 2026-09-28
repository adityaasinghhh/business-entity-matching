import os
import re
import csv
from collections import defaultdict
from heapq import nlargest

import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\adity\hackathon"

TEST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "train",
    "train",
    "test",
    "test",
)

S1_FILE = os.path.join(
    TEST_DIR,
    "test_source1.tsv",
)

S2_FILE = os.path.join(
    TEST_DIR,
    "test_source2.tsv",
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed",
)

# IMPORTANT:
# New filename so the old incompatible file is preserved.
OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "blocking_test_s1_s2_v6.tsv",
)

# ------------------------------------------------------------
# Memory / speed
# ------------------------------------------------------------

S2_CHUNK_SIZE = 100_000

# Maximum final candidates per S1.
MAX_CANDIDATES = 50

# Do not allow a single blocking bucket to explode.
MAX_BLOCK_SIZE = 500

# Number of output rows kept in RAM before flushing.
BUFFER_SIZE = 5_000

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()

    if not x:
        return ""

    # Remove URLs.
    x = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        x,
    )

    # Unicode-safe punctuation removal.
    x = re.sub(
        r"[^\w\s]",
        " ",
        x,
        flags=re.UNICODE,
    )

    # Collapse whitespace.
    x = re.sub(
        r"\s+",
        " ",
        x,
    ).strip()

    return x


def tokenize(x):
    if not x:
        return []

    return [
        token
        for token in x.split()
        if len(token) >= 2
    ]


def get_address_number(address):
    if not address:
        return ""

    match = re.search(
        r"\b\d{1,6}\b",
        address,
    )

    if match:
        return match.group(0)

    return ""


def get_prefix(name, n=5):
    if not name:
        return ""

    compact = re.sub(
        r"\s+",
        "",
        name,
    )

    if len(compact) < n:
        return ""

    return compact[:n]


# ============================================================
# FILE CHECKS
# ============================================================

print("=" * 70)
print("LIGHTWEIGHT TEST BLOCKING V6 : S1 -> S2")
print("=" * 70)

if not os.path.exists(S1_FILE):
    raise FileNotFoundError(S1_FILE)

if not os.path.exists(S2_FILE):
    raise FileNotFoundError(S2_FILE)

print(f"S1     : {S1_FILE}")
print(f"S2     : {S2_FILE}")
print(f"Output : {OUTPUT_FILE}")


# ============================================================
# LOAD S1
# ============================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False,
    na_filter=False,
)

required_columns = {
    "entity_id",
    "business_name",
    "business_address",
    "country",
}

missing = required_columns - set(s1.columns)

if missing:
    raise RuntimeError(
        f"S1 is missing columns: {sorted(missing)}"
    )

print(
    f"S1 rows: {len(s1):,}"
)


# ============================================================
# NORMALIZE S1
# ============================================================

print("\nNormalizing S1...")

s1["name_norm"] = (
    s1["business_name"]
    .map(normalize_text)
)

s1["address_norm"] = (
    s1["business_address"]
    .map(normalize_text)
)

s1["tokens"] = (
    s1["name_norm"]
    .map(tokenize)
)

s1["token1"] = s1["tokens"].map(
    lambda x: x[0] if len(x) >= 1 else ""
)

s1["token2"] = s1["tokens"].map(
    lambda x: x[1] if len(x) >= 2 else ""
)

s1["address_number"] = (
    s1["address_norm"]
    .map(get_address_number)
)

s1["prefix5"] = (
    s1["name_norm"]
    .map(get_prefix)
)


# ============================================================
# BUILD COMPACT S2 INDEXES
# ============================================================

print("\nBuilding lightweight S2 indexes...")

indexes = {
    "exact": defaultdict(list),
    "token12": defaultdict(list),
    "token_number": defaultdict(list),
    "prefix5": defaultdict(list),
}


def add(index, key, entity_id):
    if key:
        index[key].append(entity_id)


processed = 0

for chunk in tqdm(
    pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str,
        chunksize=S2_CHUNK_SIZE,
        keep_default_na=False,
        na_filter=False,
    ),
    desc="Indexing S2",
):

    # --------------------------------------------------------
    # Normalize only this chunk.
    # --------------------------------------------------------

    chunk["name_norm"] = (
        chunk["business_name"]
        .map(normalize_text)
    )

    chunk["address_norm"] = (
        chunk["business_address"]
        .map(normalize_text)
    )

    chunk["tokens"] = (
        chunk["name_norm"]
        .map(tokenize)
    )

    chunk["token1"] = chunk["tokens"].map(
        lambda x: x[0] if len(x) >= 1 else ""
    )

    chunk["token2"] = chunk["tokens"].map(
        lambda x: x[1] if len(x) >= 2 else ""
    )

    chunk["address_number"] = (
        chunk["address_norm"]
        .map(get_address_number)
    )

    chunk["prefix5"] = (
        chunk["name_norm"]
        .map(get_prefix)
    )

    # --------------------------------------------------------
    # Build indexes.
    # --------------------------------------------------------

    for row in chunk.itertuples(index=False):

        entity_id = str(row.entity_id)
        country = str(row.country)

        # Rule 1
        # Exact normalized name + country
        if row.name_norm:
            add(
                indexes["exact"],
                (
                    row.name_norm,
                    country,
                ),
                entity_id,
            )

        # Rule 2
        # First + second token + country
        if row.token1 and row.token2:
            add(
                indexes["token12"],
                (
                    row.token1,
                    row.token2,
                    country,
                ),
                entity_id,
            )

        # Rule 3
        # First token + address number + country
        if row.token1 and row.address_number:
            add(
                indexes["token_number"],
                (
                    row.token1,
                    row.address_number,
                    country,
                ),
                entity_id,
            )

        # Rule 4
        # Five-character name prefix + country
        if row.prefix5:
            add(
                indexes["prefix5"],
                (
                    row.prefix5,
                    country,
                ),
                entity_id,
            )

    processed += len(chunk)

    # Explicitly release temporary chunk.
    del chunk

print(
    f"\nS2 indexed: {processed:,}"
)


# ============================================================
# REMOVE HUGE BLOCKS
# ============================================================

print("\nFiltering large blocks...")

for name, index in indexes.items():

    oversized = [
        key
        for key, values in index.items()
        if len(values) > MAX_BLOCK_SIZE
    ]

    for key in oversized:
        del index[key]

    print(
        f"{name:15s}: "
        f"{len(index):,} blocks, "
        f"removed {len(oversized):,}"
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("\nGenerating ranked candidates...")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)


# ------------------------------------------------------------
# Open TSV directly.
# This avoids creating a large pandas output DataFrame.
# ------------------------------------------------------------

output_file = open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
    newline="",
)

writer = csv.writer(
    output_file,
    delimiter="\t",
    lineterminator="\n",
)

writer.writerow(
    [
        "source1_entity_id",
        "candidate_count",
        "candidate_entity_ids",
    ]
)


candidate_counts = []

total_candidate_pairs = 0

zero_candidates = 0

matched_rule_counts = {
    "exact": 0,
    "token12": 0,
    "token_number": 0,
    "prefix5": 0,
}


# ============================================================
# BLOCK S1
# ============================================================

for row in tqdm(
    s1.itertuples(index=False),
    total=len(s1),
    desc="Blocking S1",
):

    country = str(row.country)

    # --------------------------------------------------------
    # candidate_scores:
    #
    # entity_id -> blocking evidence score
    #
    # Higher score = stronger blocking evidence.
    # --------------------------------------------------------

    candidate_scores = {}

    # --------------------------------------------------------
    # Rule 1
    #
    # Exact normalized name + country
    #
    # Strongest signal.
    # --------------------------------------------------------

    if row.name_norm:

        values = indexes["exact"].get(
            (
                row.name_norm,
                country,
            ),
            [],
        )

        if values:
            matched_rule_counts["exact"] += 1

        for entity_id in values:

            old = candidate_scores.get(
                entity_id,
                0,
            )

            candidate_scores[entity_id] = max(
                old,
                4,
            )

    # --------------------------------------------------------
    # Rule 2
    #
    # First + second token + country
    # --------------------------------------------------------

    if row.token1 and row.token2:

        values = indexes["token12"].get(
            (
                row.token1,
                row.token2,
                country,
            ),
            [],
        )

        if values:
            matched_rule_counts["token12"] += 1

        for entity_id in values:

            old = candidate_scores.get(
                entity_id,
                0,
            )

            candidate_scores[entity_id] = max(
                old,
                3,
            )

    # --------------------------------------------------------
    # Rule 3
    #
    # First token + address number + country
    # --------------------------------------------------------

    if row.token1 and row.address_number:

        values = indexes["token_number"].get(
            (
                row.token1,
                row.address_number,
                country,
            ),
            [],
        )

        if values:
            matched_rule_counts["token_number"] += 1

        for entity_id in values:

            old = candidate_scores.get(
                entity_id,
                0,
            )

            candidate_scores[entity_id] = max(
                old,
                3,
            )

    # --------------------------------------------------------
    # Rule 4
    #
    # Five-character name prefix + country
    # --------------------------------------------------------

    if row.prefix5:

        values = indexes["prefix5"].get(
            (
                row.prefix5,
                country,
            ),
            [],
        )

        if values:
            matched_rule_counts["prefix5"] += 1

        for entity_id in values:

            old = candidate_scores.get(
                entity_id,
                0,
            )

            candidate_scores[entity_id] = max(
                old,
                1,
            )

    # --------------------------------------------------------
    # RANK CANDIDATES
    # --------------------------------------------------------
    #
    # Primary:
    #   blocking evidence
    #
    # Secondary:
    #   entity_id
    #
    # This makes output deterministic.
    # --------------------------------------------------------

    ranked = sorted(
        candidate_scores.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    )

    # Keep top 50.
    ranked = ranked[:MAX_CANDIDATES]

    candidate_ids = [
        entity_id
        for entity_id, score in ranked
    ]

    count = len(candidate_ids)

    candidate_counts.append(count)

    total_candidate_pairs += count

    if count == 0:
        zero_candidates += 1

    writer.writerow(
        [
            str(row.entity_id),
            count,
            ",".join(candidate_ids),
        ]
    )


# ============================================================
# CLOSE OUTPUT
# ============================================================

output_file.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("S2 V6 BLOCKING COMPLETE")
print("=" * 70)

print(
    f"S1 entities           : {len(s1):,}"
)

print(
    f"Average candidates/S1 : "
    f"{sum(candidate_counts) / len(candidate_counts):.2f}"
)

print(
    f"Maximum candidates/S1 : "
    f"{max(candidate_counts):,}"
)

print(
    f"Zero candidates       : "
    f"{zero_candidates:,}"
)

print(
    f"Total candidate pairs : "
    f"{total_candidate_pairs:,}"
)

print("\nBlocking rule hits:")

for rule, count in matched_rule_counts.items():
    print(
        f"  {rule:15s}: {count:,}"
    )

if os.path.exists(OUTPUT_FILE):

    size_mb = (
        os.path.getsize(OUTPUT_FILE)
        / (1024 * 1024)
    )

    print(
        f"\nOutput size           : "
        f"{size_mb:.2f} MB"
    )


print(
    f"\nOutput:\n{OUTPUT_FILE}"
)


# ============================================================
# BASIC OUTPUT CHECK
# ============================================================

print("\nRunning output sanity check...")

check = pd.read_csv(
    OUTPUT_FILE,
    sep="\t",
    dtype=str,
    nrows=5,
    keep_default_na=False,
    na_filter=False,
)

expected_columns = [
    "source1_entity_id",
    "candidate_count",
    "candidate_entity_ids",
]

if list(check.columns) != expected_columns:
    raise RuntimeError(
        "Output columns are incorrect."
    )

print("Output columns: OK")
print(check.to_string(index=False))

print("\nS2 BLOCKING FINISHED SUCCESSFULLY.")