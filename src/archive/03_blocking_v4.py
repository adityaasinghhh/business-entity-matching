import os
import re
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\adity\hackathon"

S1_FILE = os.path.join(
    BASE_DIR, "data", "raw", "train", "train", "train_source1.tsv"
)

S2_FILE = os.path.join(
    BASE_DIR, "data", "raw", "train", "train", "train_source2.tsv"
)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

OUTPUT_FILE = os.path.join(
    PROCESSED_DIR,
    "blocking_pilot_s1_s2_v4.csv"
)

S1_LIMIT = 100_000
S2_CHUNK_SIZE = 100_000

# Keep blocks above this size out.
MAX_BLOCK_SIZE = 1200

# More room than v2, but still bounded.
MAX_CANDIDATES = 800

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(x):
    if pd.isna(x):
        return ""

    x = str(x).lower().strip()

    # Remove URLs
    x = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        x
    )

    # Keep unicode letters/numbers
    x = re.sub(r"[^\w\s]", " ", x, flags=re.UNICODE)

    # Collapse whitespace
    x = re.sub(r"\s+", " ", x).strip()

    return x


def tokenize(x):
    if not x:
        return []

    return [
        token
        for token in x.split()
        if len(token) >= 2
    ]


def get_name_prefix(name, n):
    if not name:
        return ""

    compact = re.sub(r"\s+", "", name)

    if len(compact) < n:
        return ""

    return compact[:n]


def get_name_suffix(name, n):
    if not name:
        return ""
    compact = re.sub(r"\s+", "", name)
    if len(compact) < n:
        return ""
    return compact[-n:]


def get_address_number(address):
    if not address:
        return ""

    match = re.search(r"\b\d{1,6}\b", address)

    if match:
        return match.group(0)

    return ""


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("=" * 70)
print("BLOCKING V4 - HIGH RECALL")
print("=" * 70)

print("\nLoading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    nrows=S1_LIMIT
).fillna("")

print(f"Source 1 entities: {len(s1):,}")


# Normalize S1

print("\nNormalizing Source 1...")

s1["name_norm"] = s1["business_name"].map(normalize_text)
s1["address_norm"] = s1["business_address"].map(normalize_text)

s1["name_tokens"] = s1["name_norm"].map(tokenize)

s1["name_token_1"] = s1["name_tokens"].map(
    lambda x: x[0] if len(x) >= 1 else ""
)

s1["name_token_2"] = s1["name_tokens"].map(
    lambda x: x[1] if len(x) >= 2 else ""
)

s1["name_token_3"] = s1["name_tokens"].map(
    lambda x: x[2] if len(x) >= 3 else ""
)

s1["name_prefix_3"] = s1["name_norm"].map(
    lambda x: get_name_prefix(x, 3)
)

s1["name_prefix_5"] = s1["name_norm"].map(
    lambda x: get_name_prefix(x, 5)
)

s1["name_suffix_4"] = s1["name_norm"].map(
    lambda x: get_name_suffix(x, 4)
)

s1["address_number"] = s1["address_norm"].map(
    get_address_number
)


# ============================================================
# BLOCK INDEXES
# ============================================================

print("\nBuilding blocking indexes...")

indexes = {
    "exact_name_country": defaultdict(list),
    "token1_country": defaultdict(list),
    "token2_country": defaultdict(list),
    "prefix3_country": defaultdict(list),
    "prefix5_country": defaultdict(list),
    "address_number_country": defaultdict(list),
    "token_number_country": defaultdict(list),
    "token_address_country": defaultdict(list),
    "token3_country": defaultdict(list),
    "last_token_country": defaultdict(list),
    "suffix4_country": defaultdict(list),
    "token1_token2_country": defaultdict(list),
}


def add_to_index(index, key, entity_id):
    if not key:
        return

    index[key].append(entity_id)


# ============================================================
# FIRST PASS — BUILD BLOCK INDEXES
# ============================================================

processed_s2 = 0

for chunk in tqdm(
    pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str,
        chunksize=S2_CHUNK_SIZE
    ),
    desc="Indexing Source 2"
):

    chunk = chunk.fillna("")

    chunk["name_norm"] = chunk["business_name"].map(
        normalize_text
    )

    chunk["address_norm"] = chunk["business_address"].map(
        normalize_text
    )

    chunk["name_tokens"] = chunk["name_norm"].map(
        tokenize
    )

    chunk["name_token_1"] = chunk["name_tokens"].map(
        lambda x: x[0] if len(x) >= 1 else ""
    )

    chunk["name_token_2"] = chunk["name_tokens"].map(
        lambda x: x[1] if len(x) >= 2 else ""
    )

    chunk["name_token_3"] = chunk["name_tokens"].map(
        lambda x: x[2] if len(x) >= 3 else ""
    )

    chunk["name_prefix_3"] = chunk["name_norm"].map(
        lambda x: get_name_prefix(x, 3)
    )

    chunk["name_prefix_5"] = chunk["name_norm"].map(
        lambda x: get_name_prefix(x, 5)
    )

    chunk["name_suffix_4"] = chunk["name_norm"].map(
        lambda x: get_name_suffix(x, 4)
    )

    chunk["address_number"] = chunk["address_norm"].map(
        get_address_number
    )

    for row in chunk.itertuples(index=False):

        entity_id = row.entity_id
        country = row.country

        # ----------------------------------------------------
        # Rule 1: Exact name + country
        # ----------------------------------------------------

        if row.name_norm:
            add_to_index(
                indexes["exact_name_country"],
                (row.name_norm, country),
                entity_id
            )

        # ----------------------------------------------------
        # Rule 2: First token + country
        # ----------------------------------------------------

        if row.name_token_1:
            add_to_index(
                indexes["token1_country"],
                (row.name_token_1, country),
                entity_id
            )

        # ----------------------------------------------------
        # Rule 3: Second token + country
        # ----------------------------------------------------

        if row.name_token_2:
            add_to_index(
                indexes["token2_country"],
                (row.name_token_2, country),
                entity_id
            )

        # ----------------------------------------------------
        # Extra recall rules
        # ----------------------------------------------------

        if row.name_token_3:
            add_to_index(
                indexes["token3_country"],
                (row.name_token_3, country),
                entity_id
            )

        if row.name_tokens:
            add_to_index(
                indexes["last_token_country"],
                (row.name_tokens[-1], country),
                entity_id
            )

        if row.name_suffix_4:
            add_to_index(
                indexes["suffix4_country"],
                (row.name_suffix_4, country),
                entity_id
            )

        if row.name_token_1 and row.name_token_2:
            add_to_index(
                indexes["token1_token2_country"],
                (row.name_token_1, row.name_token_2, country),
                entity_id
            )

        # ----------------------------------------------------
        # Rule 8: 3-character prefix
        # ----------------------------------------------------

        if row.name_prefix_3:
            add_to_index(
                indexes["prefix3_country"],
                (row.name_prefix_3, country),
                entity_id
            )

        # ----------------------------------------------------
        # Rule 5: 5-character prefix
        # ----------------------------------------------------

        if row.name_prefix_5:
            add_to_index(
                indexes["prefix5_country"],
                (row.name_prefix_5, country),
                entity_id
            )

        # ----------------------------------------------------
        # Rule 6: Address number + country
        # ----------------------------------------------------

        if row.address_number:
            add_to_index(
                indexes["address_number_country"],
                (row.address_number, country),
                entity_id
            )

        # ----------------------------------------------------
        # Rule 7: token + address number + country
        # ----------------------------------------------------

        if row.name_token_1 and row.address_number:
            add_to_index(
                indexes["token_number_country"],
                (
                    row.name_token_1,
                    row.address_number,
                    country
                ),
                entity_id
            )

        # ----------------------------------------------------
        # Rule 8: token + address text
        #
        # Use first address token because full addresses are
        # often noisy.
        # ----------------------------------------------------

        address_tokens = tokenize(row.address_norm)

        if row.name_token_1 and address_tokens:

            # Try first two meaningful address tokens
            for addr_token in address_tokens[:2]:

                if len(addr_token) >= 3:

                    add_to_index(
                        indexes["token_address_country"],
                        (
                            row.name_token_1,
                            addr_token,
                            country
                        ),
                        entity_id
                    )

    processed_s2 += len(chunk)


print(
    f"\nSource 2 rows indexed: {processed_s2:,}"
)


# ============================================================
# REMOVE OVERSIZED BLOCKS
# ============================================================

print("\nFiltering oversized blocks...")

for name, index in indexes.items():

    before = len(index)

    oversized = {
        key
        for key, values in index.items()
        if len(values) > MAX_BLOCK_SIZE
    }

    for key in oversized:
        del index[key]

    print(
        f"{name:30s} "
        f"blocks={before:,} "
        f"removed={len(oversized):,}"
    )


# ============================================================
# GENERATE CANDIDATES
# ============================================================

print("\nGenerating candidates...")

rows = []

candidate_counts = []

rule_names = list(indexes.keys())

for row in tqdm(
    s1.itertuples(index=False),
    total=len(s1),
    desc="Blocking Source 1"
):

    country = row.country

    candidate_scores = Counter()

    # --------------------------------------------------------
    # Collect candidates from each rule
    # --------------------------------------------------------

    def collect(rule, key):

        if not key:
            return

        values = indexes[rule].get(key)

        if not values:
            return

        for entity_id in values:
            candidate_scores[entity_id] += 1

    # Exact name
    collect(
        "exact_name_country",
        (row.name_norm, country)
    )

    # Token 1
    collect(
        "token1_country",
        (row.name_token_1, country)
    )

    # Token 2
    collect(
        "token2_country",
        (row.name_token_2, country)
    )

    # Extra recall rules
    collect(
        "token3_country",
        (row.name_token_3, country)
    )

    collect(
        "last_token_country",
        (row.name_tokens[-1] if row.name_tokens else "", country)
    )

    collect(
        "suffix4_country",
        (row.name_suffix_4, country)
    )

    collect(
        "token1_token2_country",
        (row.name_token_1, row.name_token_2, country)
    )

    # Prefix 3
    collect(
        "prefix3_country",
        (row.name_prefix_3, country)
    )

    # Prefix 5
    collect(
        "prefix5_country",
        (row.name_prefix_5, country)
    )

    # Address number
    collect(
        "address_number_country",
        (row.address_number, country)
    )

    # Token + number
    collect(
        "token_number_country",
        (
            row.name_token_1,
            row.address_number,
            country
        )
    )

    # Token + address
    address_tokens = tokenize(row.address_norm)

    if row.name_token_1:

        for addr_token in address_tokens[:2]:

            if len(addr_token) >= 3:

                collect(
                    "token_address_country",
                    (
                        row.name_token_1,
                        addr_token,
                        country
                    )
                )

    # --------------------------------------------------------
    # Rank candidates by number of blocking rules matched
    # --------------------------------------------------------

    ranked = sorted(
        candidate_scores.items(),
        key=lambda x: (-x[1], x[0])
    )

    ranked = ranked[:MAX_CANDIDATES]

    candidate_ids = [
        entity_id
        for entity_id, score in ranked
    ]

    candidate_counts.append(len(candidate_ids))

    rows.append({
        "source1_entity_id": row.entity_id,
        "candidate_count": len(candidate_ids),
        "candidate_entity_ids": "|".join(candidate_ids)
    })


# ============================================================
# SAVE
# ============================================================

result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("BLOCKING V4 COMPLETE")
print("=" * 70)

print(
    f"S1 entities                : {len(result):,}"
)

print(
    f"Average candidates/S1       : "
    f"{sum(candidate_counts) / len(candidate_counts):.2f}"
)

print(
    f"Maximum candidates/S1       : "
    f"{max(candidate_counts):,}"
)

print(
    f"Zero candidates             : "
    f"{sum(x == 0 for x in candidate_counts):,}"
)

print(
    f"Total candidate pairs       : "
    f"{sum(candidate_counts):,}"
)

print(
    f"\nOutput:"
)

print(OUTPUT_FILE)