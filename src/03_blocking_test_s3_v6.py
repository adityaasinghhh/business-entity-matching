import os
import re
import csv
import pandas as pd
from collections import defaultdict, Counter
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
    "test"
)

S1_FILE = os.path.join(
    TEST_DIR,
    "test_source1.tsv"
)

S3_FILE = os.path.join(
    TEST_DIR,
    "test_source3.tsv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

# NEW V6 OUTPUT
OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "blocking_test_s1_s3_v6.tsv"
)

S3_CHUNK_SIZE = 100_000

MAX_CANDIDATES = 50

BUFFER_SIZE = 10_000

MAX_BLOCK_SIZE = 500

os.makedirs(
    PROCESSED_DIR,
    exist_ok=True
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):
    """
    Lowercase, remove URLs/punctuation,
    preserve Unicode letters/numbers,
    and normalize whitespace.
    """

    if pd.isna(x):
        return ""

    x = str(x).lower().strip()

    # Remove URLs
    x = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        x
    )

    # Keep Unicode letters, numbers and whitespace
    x = re.sub(
        r"[^\w\s]",
        " ",
        x,
        flags=re.UNICODE
    )

    # Collapse whitespace
    x = re.sub(
        r"\s+",
        " ",
        x
    ).strip()

    return x


def tokenize(x):
    """
    Split normalized text into meaningful tokens.
    """

    if not x:
        return []

    return [
        token
        for token in x.split()
        if len(token) >= 2
    ]


def get_address_number(address):
    """
    Extract the first standalone 1-6 digit number
    from an address.
    """

    if not address:
        return ""

    match = re.search(
        r"\b\d{1,6}\b",
        address
    )

    if match:
        return match.group(0)

    return ""


def get_prefix(name, n=5):
    """
    Return first n characters after removing spaces.
    """

    if not name:
        return ""

    compact = re.sub(
        r"\s+",
        "",
        name
    )

    if len(compact) < n:
        return ""

    return compact[:n]


# ============================================================
# START
# ============================================================

print("=" * 70)
print("LIGHTWEIGHT TEST BLOCKING V6 : S1 -> S3")
print("=" * 70)

print("\nChecking input files...")

if not os.path.exists(S1_FILE):
    raise FileNotFoundError(
        f"S1 test file not found:\n{S1_FILE}"
    )

if not os.path.exists(S3_FILE):
    raise FileNotFoundError(
        f"S3 test file not found:\n{S3_FILE}"
    )

print(f"S1     : {S1_FILE}")
print(f"S3     : {S3_FILE}")
print(f"Output : {OUTPUT_FILE}")


# ============================================================
# LOAD S1
# ============================================================

print("\nLoading S1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
).fillna("")

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
# BUILD S3 INDEXES
# ============================================================

print("\nBuilding lightweight S3 indexes...")

indexes = {
    # Rule 1
    "exact": defaultdict(list),

    # Rule 2
    "token12": defaultdict(list),

    # Rule 3
    "token_number": defaultdict(list),

    # Rule 4
    "prefix5": defaultdict(list),
}


def add_to_index(index, key, entity_id):

    if not key:
        return

    index[key].append(entity_id)


# ============================================================
# INDEX S3
# ============================================================

processed_s3 = 0

for chunk in tqdm(
    pd.read_csv(
        S3_FILE,
        sep="\t",
        dtype=str,
        chunksize=S3_CHUNK_SIZE
    ),
    desc="Indexing S3"
):

    chunk = chunk.fillna("")

    # --------------------------------------------------------
    # Normalize
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
    # Add to indexes
    # --------------------------------------------------------

    for row in chunk.itertuples(index=False):

        entity_id = row.entity_id
        country = row.country

        # Rule 1:
        # Exact normalized name + country
        if row.name_norm:

            add_to_index(
                indexes["exact"],
                (
                    row.name_norm,
                    country
                ),
                entity_id
            )

        # Rule 2:
        # First + second token + country
        if row.token1 and row.token2:

            add_to_index(
                indexes["token12"],
                (
                    row.token1,
                    row.token2,
                    country
                ),
                entity_id
            )

        # Rule 3:
        # First token + address number + country
        if row.token1 and row.address_number:

            add_to_index(
                indexes["token_number"],
                (
                    row.token1,
                    row.address_number,
                    country
                ),
                entity_id
            )

        # Rule 4:
        # First 5 characters + country
        if row.prefix5:

            add_to_index(
                indexes["prefix5"],
                (
                    row.prefix5,
                    country
                ),
                entity_id
            )

    processed_s3 += len(chunk)


print(
    f"\nS3 indexed: {processed_s3:,}"
)


# ============================================================
# REMOVE OVERSIZED BLOCKS
# ============================================================

print("\nFiltering large blocks...")

for name, index in indexes.items():

    before = len(index)

    oversized_keys = [
        key
        for key, values in index.items()
        if len(values) > MAX_BLOCK_SIZE
    ]

    for key in oversized_keys:
        del index[key]

    print(
        f"{name:15s}: "
        f"{len(index):,} blocks, "
        f"removed {len(oversized_keys):,}"
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("\nGenerating ranked candidates...")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

candidate_counts = []

zero_candidates = 0

total_candidate_pairs = 0

rule_hits = {
    "exact": 0,
    "token12": 0,
    "token_number": 0,
    "prefix5": 0
}


# ============================================================
# STREAM OUTPUT
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as out:

    writer = csv.writer(
        out,
        delimiter="\t"
    )

    # Intermediate blocker format
    writer.writerow([
        "source1_entity_id",
        "candidate_count",
        "candidate_entity_ids"
    ])

    # ========================================================
    # PROCESS S1
    # ========================================================

    for row in tqdm(
        s1.itertuples(index=False),
        total=len(s1),
        desc="Blocking S1"
    ):

        country = row.country

        # ----------------------------------------------------
        # Counter allows multiple rule hits
        # ----------------------------------------------------

        candidate_scores = Counter()

        # ----------------------------------------------------
        # Rule 1: Exact name + country
        # Weight = 4
        # ----------------------------------------------------

        if row.name_norm:

            values = indexes["exact"].get(
                (
                    row.name_norm,
                    country
                ),
                []
            )

            if values:
                rule_hits["exact"] += 1

            for entity_id in values:
                candidate_scores[entity_id] += 4

        # ----------------------------------------------------
        # Rule 2: First + second token + country
        # Weight = 3
        # ----------------------------------------------------

        if row.token1 and row.token2:

            values = indexes["token12"].get(
                (
                    row.token1,
                    row.token2,
                    country
                ),
                []
            )

            if values:
                rule_hits["token12"] += 1

            for entity_id in values:
                candidate_scores[entity_id] += 3

        # ----------------------------------------------------
        # Rule 3: First token + address number + country
        # Weight = 3
        # ----------------------------------------------------

        if row.token1 and row.address_number:

            values = indexes["token_number"].get(
                (
                    row.token1,
                    row.address_number,
                    country
                ),
                []
            )

            if values:
                rule_hits["token_number"] += 1

            for entity_id in values:
                candidate_scores[entity_id] += 3

        # ----------------------------------------------------
        # Rule 4: Prefix5 + country
        # Weight = 1
        # ----------------------------------------------------

        if row.prefix5:

            values = indexes["prefix5"].get(
                (
                    row.prefix5,
                    country
                ),
                []
            )

            if values:
                rule_hits["prefix5"] += 1

            for entity_id in values:
                candidate_scores[entity_id] += 1

        # ----------------------------------------------------
        # Deterministic ranking
        #
        # Higher blocking evidence first.
        # Entity ID breaks ties deterministically.
        # ----------------------------------------------------

        ranked_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: (-x[1], x[0])
        )

        # Keep only top 50
        ranked_candidates = ranked_candidates[
            :MAX_CANDIDATES
        ]

        candidate_ids = [
            entity_id
            for entity_id, score in ranked_candidates
        ]

        candidate_count = len(candidate_ids)

        candidate_counts.append(
            candidate_count
        )

        total_candidate_pairs += candidate_count

        if candidate_count == 0:
            zero_candidates += 1

        # IMPORTANT:
        # Comma-separated IDs.
        # This matches the S2 V6 format and
        # the final challenge candidate_pairs format.
        writer.writerow([
            row.entity_id,
            candidate_count,
            ",".join(candidate_ids)
        ])


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("S3 V6 BLOCKING COMPLETE")
print("=" * 70)

total_s1 = len(s1)

average_candidates = (
    total_candidate_pairs / total_s1
    if total_s1
    else 0
)

maximum_candidates = (
    max(candidate_counts)
    if candidate_counts
    else 0
)

print(
    f"S1 entities           : {total_s1:,}"
)

print(
    f"S3 entities indexed   : {processed_s3:,}"
)

print(
    f"Average candidates/S1 : "
    f"{average_candidates:.2f}"
)

print(
    f"Maximum candidates/S1 : "
    f"{maximum_candidates:,}"
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

print(
    f"  exact          : "
    f"{rule_hits['exact']:,}"
)

print(
    f"  token12        : "
    f"{rule_hits['token12']:,}"
)

print(
    f"  token_number   : "
    f"{rule_hits['token_number']:,}"
)

print(
    f"  prefix5        : "
    f"{rule_hits['prefix5']:,}"
)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

if not os.path.exists(OUTPUT_FILE):
    raise RuntimeError(
        "Output file was not created."
    )

output_size_mb = (
    os.path.getsize(OUTPUT_FILE)
    / (1024 * 1024)
)

print(
    f"\nOutput size           : "
    f"{output_size_mb:.2f} MB"
)

print(
    f"Output file           : "
    f"{OUTPUT_FILE}"
)


# ============================================================
# HEADER CHECK
# ============================================================

print("\nRunning output sanity check...")

with open(
    OUTPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    header = f.readline().rstrip("\n")

expected_header = (
    "source1_entity_id\t"
    "candidate_count\t"
    "candidate_entity_ids"
)

if header != expected_header:

    raise RuntimeError(
        f"Unexpected output header:\n{header}"
    )

print("Output columns: OK")

print("\n" + "=" * 70)
print("S3 BLOCKING FINISHED SUCCESSFULLY")
print("=" * 70)