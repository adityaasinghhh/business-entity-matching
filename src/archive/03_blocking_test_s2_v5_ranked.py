import os
import re
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
    "test_source2.tsv"
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "blocking_test_s1_s3_v5.tsv"
)

# Read S3 in manageable chunks
S3_CHUNK_SIZE = 100_000

# Maximum candidates retained for each S1
MAX_CANDIDATES = 50

# Write output every 10,000 S1 records
BUFFER_SIZE = 10_000

# Remove very large blocking groups
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
# CHECK FILES
# ============================================================

print("=" * 70)
print("LIGHTWEIGHT TEST BLOCKING V5 : S1 -> S3")
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

print(f"S1 file : {S1_FILE}")
print(f"S3 file : {S3_FILE}")
print(f"Output  : {OUTPUT_FILE}")


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("\nLoading TEST Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str
).fillna("")

print(
    f"S1 rows: {len(s1):,}"
)


# ============================================================
# NORMALIZE SOURCE 1
# ============================================================

print("\nNormalizing Source 1...")

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
# BUILD S3 BLOCKING INDEXES
# ============================================================

print("\nBuilding lightweight S3 indexes...")

indexes = {
    # Exact normalized name + country
    "exact": defaultdict(list),

    # First + second name token + country
    "token12": defaultdict(list),

    # First name token + address number + country
    "token_number": defaultdict(list),

    # First 5 name characters + country
    "prefix5": defaultdict(list),
}


def add_to_index(index, key, entity_id):
    """
    Add an entity ID to a blocking index.
    """

    if not key:
        return

    index[key].append(entity_id)


# ============================================================
# INDEX SOURCE 3
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
    # Normalize S3
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
    # Add S3 records to indexes
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
        # 5-character prefix + country
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
    f"\nS3 rows indexed: {processed_s3:,}"
)


# ============================================================
# REMOVE OVERSIZED BLOCKS
# ============================================================

print("\nFiltering oversized blocks...")

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
        f"blocks={len(index):,} "
        f"removed={len(oversized_keys):,}"
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("\nGenerating S1 -> S3 candidates...")

# Delete previous output
if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

buffer = []

header_written = False

candidate_counts = []

zero_candidates = 0


# ============================================================
# PROCESS S1
# ============================================================

for row in tqdm(
    s1.itertuples(index=False),
    total=len(s1),
    desc="Blocking S1 -> S3"
):

    country = row.country

    # Counter instead of set:
    # this allows candidates matching multiple rules
    # to receive higher scores.
    candidate_scores = Counter()

    # --------------------------------------------------------
    # Rule 1: Exact name + country
    # --------------------------------------------------------

    if row.name_norm:

        values = indexes["exact"].get(
            (
                row.name_norm,
                country
            ),
            []
        )

        for entity_id in values:
            candidate_scores[entity_id] += 4

    # --------------------------------------------------------
    # Rule 2: First + second token + country
    # --------------------------------------------------------

    if row.token1 and row.token2:

        values = indexes["token12"].get(
            (
                row.token1,
                row.token2,
                country
            ),
            []
        )

        for entity_id in values:
            candidate_scores[entity_id] += 3

    # --------------------------------------------------------
    # Rule 3: First token + address number + country
    # --------------------------------------------------------

    if row.token1 and row.address_number:

        values = indexes["token_number"].get(
            (
                row.token1,
                row.address_number,
                country
            ),
            []
        )

        for entity_id in values:
            candidate_scores[entity_id] += 3

    # --------------------------------------------------------
    # Rule 4: Prefix5 + country
    # --------------------------------------------------------

    if row.prefix5:

        values = indexes["prefix5"].get(
            (
                row.prefix5,
                country
            ),
            []
        )

        for entity_id in values:
            candidate_scores[entity_id] += 1

    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    ranked_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: (-x[1], x[0])
    )

    # Keep only top candidates
    ranked_candidates = (
        ranked_candidates[:MAX_CANDIDATES]
    )

    candidate_ids = [
        entity_id
        for entity_id, score in ranked_candidates
    ]

    candidate_count = len(candidate_ids)

    candidate_counts.append(
        candidate_count
    )

    if candidate_count == 0:
        zero_candidates += 1

    # --------------------------------------------------------
    # Add to streaming buffer
    # --------------------------------------------------------

    buffer.append({
        "source1_entity_id": row.entity_id,
        "candidate_count": candidate_count,
        "candidate_entity_ids": "|".join(candidate_ids)
    })

    # --------------------------------------------------------
    # Write every BUFFER_SIZE rows
    # --------------------------------------------------------

    if len(buffer) >= BUFFER_SIZE:

        pd.DataFrame(buffer).to_csv(
            OUTPUT_FILE,
            sep="\t",
            mode="a",
            header=not header_written,
            index=False
        )

        header_written = True

        buffer.clear()


# ============================================================
# WRITE REMAINING BUFFER
# ============================================================

if buffer:

    pd.DataFrame(buffer).to_csv(
        OUTPUT_FILE,
        sep="\t",
        mode="a",
        header=not header_written,
        index=False
    )

    buffer.clear()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("LIGHTWEIGHT S1 -> S3 BLOCKING COMPLETE")
print("=" * 70)

total_s1 = len(s1)

total_candidates = sum(
    candidate_counts
)

average_candidates = (
    total_candidates / total_s1
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
    f"{total_candidates:,}"
)


# ============================================================
# OUTPUT VALIDATION
# ============================================================

print("\nValidating output...")

if not os.path.exists(OUTPUT_FILE):

    raise RuntimeError(
        "Output file was not created."
    )

output_size_mb = (
    os.path.getsize(OUTPUT_FILE)
    / (1024 * 1024)
)

print(
    f"Output size           : "
    f"{output_size_mb:.2f} MB"
)

print(
    f"Output file           : "
    f"{OUTPUT_FILE}"
)

print("\nChecking TSV header...")

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

print("TSV header: OK")

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)