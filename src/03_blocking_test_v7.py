from pathlib import Path
from collections import defaultdict
import csv
import re
import sys
import time

ROOT = Path(r"C:\Users\adity\hackathon")

RAW = ROOT / "data" / "raw" / "train" / "train" / "test" / "test"
PROCESSED = ROOT / "data" / "processed"

S1_FILE = RAW / "test_source1.tsv"

SOURCE = sys.argv[1].lower() if len(sys.argv) > 1 else "s2"

if SOURCE not in {"s2", "s3"}:
    raise SystemExit("Usage: python 03_blocking_test_v7.py s2|s3")

SRC_FILE = RAW / f"test_source{SOURCE[-1]}.tsv"
OUTPUT_FILE = PROCESSED / f"blocking_test_s1_{SOURCE}_v7.tsv"

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# This is deliberately larger than V6 because recall is our
# first priority.
#
# We do NOT simply take the first N dictionary results.
# Candidates are scored and ranked first.
# ------------------------------------------------------------

MAX_CANDIDATES = 100

# Individual blocking buckets larger than this are ignored.
# They are too non-selective to be useful.
MAX_BLOCK_SIZE = 1000

# Candidates with very strong independent evidence are preserved
# before the normal top-K selection.
MAX_STRONG_CANDIDATES = 30

S1_CHUNK_SIZE = 100_000
SRC_CHUNK_SIZE = 100_000
WRITE_BUFFER = 20_000

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"\d{1,8}")


def normalize(x):
    if x is None:
        return ""

    x = str(x).lower().strip()
    x = URL_RE.sub(" ", x)
    x = NON_WORD_RE.sub(" ", x)
    x = SPACE_RE.sub(" ", x).strip()

    return x


def tokens(x):
    if not x:
        return []

    return [t for t in x.split() if len(t) >= 2]


def prefix(x, n):
    compact = x.replace(" ", "")
    return compact[:n] if len(compact) >= n else ""


def suffix(x, n):
    compact = x.replace(" ", "")
    return compact[-n:] if len(compact) >= n else ""


def numbers(x):
    return set(NUMBER_RE.findall(x))


def add(index, key, entity_id):
    if not key:
        return
    index[key].append(entity_id)


def prepare_record(entity_id, name, address, country):
    name = normalize(name)
    address = normalize(address)
    country = normalize(country)

    nt = tokens(name)
    at = tokens(address)
    nums = numbers(address)

    return {
        "id": str(entity_id),
        "name": name,
        "address": address,
        "country": country,
        "tokens": nt,
        "address_tokens": at,
        "numbers": nums,
        "token1": nt[0] if len(nt) >= 1 else "",
        "token2": nt[1] if len(nt) >= 2 else "",
        "last_token": nt[-1] if nt else "",
        "prefix5": prefix(name, 5),
        "suffix4": suffix(name, 4),
        "address_number": next(iter(nums), ""),
    }


def build_indexes():
    print("=" * 80)
    print(f"V7 HIGH-RECALL BLOCKER — S1 -> {SOURCE.upper()}")
    print("=" * 80)

    indexes = {
        "exact_name": defaultdict(list),
        "token12": defaultdict(list),
        "token_number": defaultdict(list),
        "prefix5": defaultdict(list),
        "suffix4": defaultdict(list),
        "last_token": defaultdict(list),
        "address_number": defaultdict(list),
        "token_address": defaultdict(list),
        "token2_address": defaultdict(list),
        "address_token": defaultdict(list),
    }

    print(f"\nIndexing {SRC_FILE}")
    start = time.time()
    count = 0

    with open(
        SRC_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            r = prepare_record(
                row["entity_id"],
                row["business_name"],
                row["business_address"],
                row["country"],
            )

            eid = r["id"]
            country = r["country"]

            # ------------------------------------------------
            # 1. Exact normalized name + country
            # ------------------------------------------------
            if r["name"]:
                add(
                    indexes["exact_name"],
                    (r["name"], country),
                    eid
                )

            # ------------------------------------------------
            # 2. First + second name token + country
            # ------------------------------------------------
            if r["token1"] and r["token2"]:
                add(
                    indexes["token12"],
                    (r["token1"], r["token2"], country),
                    eid
                )

            # ------------------------------------------------
            # 3. First name token + address number + country
            # ------------------------------------------------
            if r["token1"] and r["address_number"]:
                add(
                    indexes["token_number"],
                    (
                        r["token1"],
                        r["address_number"],
                        country
                    ),
                    eid
                )

            # ------------------------------------------------
            # 4. Name prefix + country
            # ------------------------------------------------
            if r["prefix5"]:
                add(
                    indexes["prefix5"],
                    (r["prefix5"], country),
                    eid
                )

            # ------------------------------------------------
            # 5. Name suffix + country
            # ------------------------------------------------
            if r["suffix4"]:
                add(
                    indexes["suffix4"],
                    (r["suffix4"], country),
                    eid
                )

            # ------------------------------------------------
            # 6. Last name token + country
            # ------------------------------------------------
            if r["last_token"]:
                add(
                    indexes["last_token"],
                    (r["last_token"], country),
                    eid
                )

            # ------------------------------------------------
            # 7. Address number + country
            # ------------------------------------------------
            if r["address_number"]:
                add(
                    indexes["address_number"],
                    (
                        r["address_number"],
                        country
                    ),
                    eid
                )

            # ------------------------------------------------
            # 8. First name token + address token + country
            # ------------------------------------------------
            if r["token1"]:
                for at in r["address_tokens"][:4]:
                    if len(at) >= 3:
                        add(
                            indexes["token_address"],
                            (
                                r["token1"],
                                at,
                                country
                            ),
                            eid
                        )

            # ------------------------------------------------
            # 9. Second name token + address token + country
            # ------------------------------------------------
            if r["token2"]:
                for at in r["address_tokens"][:4]:
                    if len(at) >= 3:
                        add(
                            indexes["token2_address"],
                            (
                                r["token2"],
                                at,
                                country
                            ),
                            eid
                        )

            # ------------------------------------------------
            # 10. Address token + country
            # ------------------------------------------------
            for at in r["address_tokens"][:4]:
                if len(at) >= 4:
                    add(
                        indexes["address_token"],
                        (at, country),
                        eid
                    )

            count += 1

            if count % 500_000 == 0:
                elapsed = time.time() - start
                print(
                    f"Indexed {count:,} {SOURCE.upper()} rows "
                    f"({count / max(elapsed, 1):,.0f}/sec)"
                )

    print(f"\nIndexed {count:,} {SOURCE.upper()} records.")

    # --------------------------------------------------------
    # Remove oversized buckets.
    # --------------------------------------------------------

    print("\nFiltering oversized blocking buckets:")

    for rule, index in indexes.items():
        before = len(index)

        oversized = [
            key
            for key, values in index.items()
            if len(values) > MAX_BLOCK_SIZE
        ]

        for key in oversized:
            del index[key]

        print(
            f"{rule:20s} "
            f"blocks={before:,} "
            f"removed={len(oversized):,}"
        )

    return indexes


def retrieve_candidates(r, indexes):
    """
    Return:
        candidate_id -> set(rule names)
    """

    hits = defaultdict(set)
    country = r["country"]

    def collect(rule, key):
        if not key:
            return

        values = indexes[rule].get(key)

        if values:
            for eid in values:
                hits[eid].add(rule)

    # Strong / precise rules first.
    if r["name"]:
        collect(
            "exact_name",
            (r["name"], country)
        )

    if r["token1"] and r["token2"]:
        collect(
            "token12",
            (
                r["token1"],
                r["token2"],
                country
            )
        )

    if r["token1"] and r["address_number"]:
        collect(
            "token_number",
            (
                r["token1"],
                r["address_number"],
                country
            )
        )

    if r["prefix5"]:
        collect(
            "prefix5",
            (
                r["prefix5"],
                country
            )
        )

    if r["suffix4"]:
        collect(
            "suffix4",
            (
                r["suffix4"],
                country
            )
        )

    if r["last_token"]:
        collect(
            "last_token",
            (
                r["last_token"],
                country
            )
        )

    # Address-derived retrieval.
    if r["address_number"]:
        collect(
            "address_number",
            (
                r["address_number"],
                country
            )
        )

    for at in r["address_tokens"][:4]:

        if len(at) < 3:
            continue

        if r["token1"]:
            collect(
                "token_address",
                (
                    r["token1"],
                    at,
                    country
                )
            )

        if r["token2"]:
            collect(
                "token2_address",
                (
                    r["token2"],
                    at,
                    country
                )
            )

        collect(
            "address_token",
            (at, country)
        )

    return hits


def cheap_score(query, candidate):
    """
    Deterministic blocker score.

    This is NOT the ML score.

    It exists only to rank candidates after high-recall
    retrieval.
    """

    qname = query["name"]
    cname = candidate["name"]

    qaddr = query["address"]
    caddr = candidate["address"]

    score = 0

    rules = candidate["rules"]

    # Rule evidence.
    weights = {
        "exact_name": 1000,
        "token12": 350,
        "token_number": 320,
        "token_address": 280,
        "token2_address": 260,
        "prefix5": 220,
        "suffix4": 180,
        "last_token": 150,
        "address_number": 140,
        "address_token": 110,
    }

    score += sum(
        weights.get(rule, 0)
        for rule in rules
    )

    # Exact normalized name.
    if qname and qname == cname:
        score += 1000

    # Name token overlap.
    qtokens = set(query["tokens"])
    ctokens = set(candidate["tokens"])

    shared_name = qtokens.intersection(ctokens)

    if shared_name:
        score += min(
            120,
            30 * len(shared_name)
        )

        if qtokens and shared_name == qtokens:
            score += 80

    # Address token overlap.
    qaddr_tokens = set(query["address_tokens"])
    caddr_tokens = set(candidate["address_tokens"])

    shared_addr = qaddr_tokens.intersection(caddr_tokens)

    if shared_addr:
        score += min(
            180,
            30 * len(shared_addr)
        )

    # Address number overlap.
    shared_numbers = query["numbers"].intersection(
        candidate["numbers"]
    )

    if shared_numbers:
        score += 300

    # Name + address combination.
    if shared_name and shared_addr:
        score += 250

    # Exact address.
    if qaddr and qaddr == caddr:
        score += 500

    return score


def load_candidate_record_cache():
    """
    Source records are loaded into a dictionary only for candidates
    actually retrieved.

    This avoids loading another huge DataFrame into memory.
    """

    cache = {}

    return cache


def rank_and_limit(query, hit_map, source_lookup):
    records = []

    eids = list(hit_map.keys())

    if not eids:
        return records

    placeholders = ",".join(["?"] * len(eids))

    rows = source_lookup.execute(
        f"""
        SELECT id, name, address, country
        FROM source_lookup
        WHERE id IN ({placeholders})
        """,
        eids
    ).fetchall()

    source_rows = {
        row[0]: {
            "id": row[0],
            "name": row[1],
            "address": row[2],
            "country": row[3],
        }
        for row in rows
    }

    for eid, rules in hit_map.items():

        rec = source_rows.get(eid)

        if rec is None:
            continue

        rec2 = {
            **rec,
            "rules": rules
        }

        score = cheap_score(
            query,
            rec2
        )

        records.append(
            (
                score,
                eid,
                rules
            )
        )

    if not records:
        return records

    records.sort(
        key=lambda x: (-x[0], x[1])
    )

    return records[:MAX_CANDIDATES]
def run(indexes):
    print("\nLoading S1...")

    s1_rows = []

    with open(
        S1_FILE,
        "r",
        encoding="utf-8",
        errors="replace",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in reader:
            s1_rows.append(
                prepare_record(
                    row["entity_id"],
                    row["business_name"],
                    row["business_address"],
                    row["country"]
                )
            )

    print(
        f"S1 records: {len(s1_rows):,}"
    )

    # --------------------------------------------------------
    # Source lookup.
    #
    # Only the source-side fields needed for cheap ranking are
    # retained. This is ~5M records, but much lighter than
    # keeping the entire source DataFrame.
    # --------------------------------------------------------

    print(
        f"\nLoading lightweight {SOURCE.upper()} lookup..."
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT use csv.DictReader for the second pass over the
    # 5M-row TSV. On Windows this can fail with:
    #
    # OSError: [Errno 22] Invalid argument
    #
    # Read in bounded pandas chunks instead.
    # --------------------------------------------------------

    import pandas as pd

    source_lookup = {}

    source_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    lookup_chunks = 0

    for chunk in pd.read_csv(
        SRC_FILE,
        sep="\\t",
        dtype=str,
        usecols=source_columns,
        chunksize=100_000,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
        on_bad_lines="error",
    ):

        for row in chunk.itertuples(index=False):

            r = prepare_record(
                row.entity_id,
                row.business_name,
                row.business_address,
                row.country,
            )

            source_lookup[r["id"]] = r

        lookup_chunks += 1

        if lookup_chunks % 10 == 0:
            print(
                f"Lookup loaded: "
                f"{len(source_lookup):,}"
            )

    print(
        f"Source lookup: {len(source_lookup):,}"
    )

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_candidates = 0
    zero_candidates = 0
    max_candidates = 0
    rows_written = 0

    rule_hits = defaultdict(int)

    start = time.time()

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as out:

        writer = csv.writer(
            out,
            delimiter="\t",
            lineterminator="\n"
        )

        writer.writerow(
            [
                "source1_entity_id",
                "source_entity_id",
                "block_rank",
                "block_score",
                "block_rules",
            ]
        )

        buffer = []

        for i, query in enumerate(
            s1_rows,
            start=1
        ):

            hits = retrieve_candidates(
                query,
                indexes
            )

            ranked = rank_and_limit(
                query,
                hits,
                source_lookup
            )

            count = len(ranked)

            total_candidates += count
            max_candidates = max(
                max_candidates,
                count
            )

            if count == 0:
                zero_candidates += 1

            for rank, (score, eid, rules) in enumerate(
                ranked,
                start=1
            ):

                rule_string = ",".join(
                    sorted(rules)
                )

                buffer.append(
                    [
                        query["id"],
                        eid,
                        rank,
                        score,
                        rule_string
                    ]
                )

                for rule in rules:
                    rule_hits[rule] += 1

            if len(buffer) >= WRITE_BUFFER:

                writer.writerows(buffer)
                rows_written += len(buffer)
                buffer.clear()

            if i % 100_000 == 0:

                elapsed = time.time() - start

                print(
                    f"S1 {i:,}/{len(s1_rows):,} | "
                    f"avg candidates="
                    f"{total_candidates / i:.2f} | "
                    f"zero={zero_candidates:,} | "
                    f"rate="
                    f"{i / max(elapsed, 1):.1f}/sec"
                )

        if buffer:
            writer.writerows(buffer)
            rows_written += len(buffer)

    print("\n" + "=" * 80)
    print("V7 BLOCKING COMPLETE")
    print("=" * 80)

    print(
        f"\nS1 records             : {len(s1_rows):,}"
    )

    print(
        f"Candidate rows         : {rows_written:,}"
    )

    print(
        f"Average candidates/S1  : "
        f"{total_candidates / max(len(s1_rows), 1):.2f}"
    )

    print(
        f"Maximum candidates/S1  : {max_candidates:,}"
    )

    print(
        f"Zero-candidate S1      : {zero_candidates:,}"
    )

    print("\nRule hits:")

    for rule, count in sorted(
        rule_hits.items(),
        key=lambda x: -x[1]
    ):
        print(
            f"{rule:20s}: {count:,}"
        )

    print(
        f"\nOutput:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":

    indexes = build_indexes()
    run(indexes)

