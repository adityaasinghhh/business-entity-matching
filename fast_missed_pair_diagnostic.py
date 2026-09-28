"""
FAST MISSED-PAIR DIAGNOSTIC - V2 (FIXED)

Run from:
    C:/Users/adity/hackathon

Purpose:
    Measure V6 blocking recall on a 100K S1 training sample and
    identify which additional blocking rules recover missed GT pairs.

Designed for ~8 GB RAM.
Does NOT build DuckDB indexes.
Does NOT load all S2/S3 into RAM.

FIXES vs V2 original:
    - S2/S3 column names are now resolved dynamically with get_col(),
      same as S1. The original hardcoded usecols=["entity_id",
      "business_name", "business_address", "country"] and would crash
      with a ValueError if S2/S3 used different column names (e.g.
      "source2_entity_id").
    - s1.iterrows() was being run twice to build exact_index/
      prefix_index/number_index and then s1_info separately; merged
      into one pass.
    - Added `matching_results`: a per-pair table recording which rule
      FIRST would have recovered each missed GT pair (rule priority
      order), plus a printed summary and a saved TSV.
"""

from pathlib import Path
import pandas as pd
import re
import time
import sys


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
PROC = DATA / "processed"
TRAIN = DATA / "raw" / "train" / "train"

GT = TRAIN / "train_ground_truth.tsv"
S1 = TRAIN / "train_source1.tsv"
S2 = TRAIN / "train_source2.tsv"
S3 = TRAIN / "train_source3.tsv"

N_S1 = 100_000
CHUNK = 100_000

MATCHING_RESULTS_OUT = PROC / "missed_pair_matching_results.tsv"


# ============================================================
# HELPERS
# ============================================================

def norm(x):
    """Lightweight normalization."""
    if pd.isna(x):
        return ""

    x = str(x).lower()

    x = re.sub(
        r"[^0-9a-z\u0080-\uffff]+",
        " ",
        x
    )

    return re.sub(r"\s+", " ", x).strip()


def get_col(df, names):
    """Find a column using multiple possible names."""
    low = {c.lower(): c for c in df.columns}

    for name in names:
        if name.lower() in low:
            return low[name.lower()]

    return None


def resolve_source_columns(path):
    """
    Peek at a source file's header and resolve the id/name/address/
    country columns using the same flexible name matching as S1.
    Returns (id_col, name_col, addr_col, country_col).
    """

    header = pd.read_csv(path, sep="\t", dtype=str, nrows=0)

    id_col = get_col(
        header,
        ["entity_id", "source2_entity_id", "source3_entity_id", "id"]
    )

    name_col = get_col(
        header,
        ["business_name", "name", "entity_name"]
    )

    addr_col = get_col(
        header,
        ["business_address", "address", "address_full", "full_address"]
    )

    country_col = get_col(
        header,
        ["country", "country_code"]
    )

    missing = [
        label for label, col in [
            ("entity id", id_col),
            ("business name", name_col),
            ("address", addr_col),
            ("country", country_col),
        ] if col is None
    ]

    if missing:
        raise RuntimeError(
            f"Could not resolve columns {missing} in {path.name}. "
            f"Columns available: {header.columns.tolist()}"
        )

    return id_col, name_col, addr_col, country_col


def parse_candidate_ids(raw):
    """
    Parse V6 candidate_entity_ids.

    Handles formats such as:
        ['abc', 'def']
        ["abc", "def"]
        abc,def
        abc;def
    """

    if pd.isna(raw):
        return []

    text = str(raw)

    # Entity IDs are normally alphanumeric with _ / -
    return re.findall(r"[A-Za-z0-9_-]+", text)


def split_gt_ids(value):
    """
    Parse matched_entity_ids from ground truth.
    """

    if pd.isna(value):
        return []

    text = str(value)

    return [
        x for x in re.split(
            r"[,;|\\s]+",
            text
        )
        if x
    ]


# ============================================================
# PRINT PATHS
# ============================================================

print("=" * 70)
print("FAST MISSED-PAIR DIAGNOSTIC V2 (FIXED)")
print("=" * 70)

print("ROOT:", ROOT)
print("GT:", GT)
print("S1:", S1)
print("S2:", S2)
print("S3:", S3)

for p in [GT, S1, S2, S3]:
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")


# ============================================================
# LOAD 100K S1
# ============================================================

print("\n[1/6] Loading first 100K S1 records...")

t0 = time.time()

s1 = pd.read_csv(
    S1,
    sep="\t",
    dtype=str,
    nrows=N_S1
)

print(
    f"S1 loaded: {len(s1):,} rows "
    f"in {time.time() - t0:.1f}s"
)

s1_id = get_col(s1, ["entity_id", "source1_entity_id"])
name1 = get_col(s1, ["business_name", "name", "entity_name"])
addr1 = get_col(s1, ["business_address", "address", "address_full", "full_address"])
country1 = get_col(s1, ["country", "country_code"])

if not s1_id:
    raise RuntimeError(f"Could not find S1 ID column. Columns: {s1.columns.tolist()}")
if not name1:
    raise RuntimeError(f"Could not find S1 business name column. Columns: {s1.columns.tolist()}")
if not addr1:
    raise RuntimeError(f"Could not find S1 address column. Columns: {s1.columns.tolist()}")
if not country1:
    raise RuntimeError(f"Could not find S1 country column. Columns: {s1.columns.tolist()}")

sample_ids = set(s1[s1_id].dropna().astype(str))

print(f"100K S1 IDs: {len(sample_ids):,}")


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

print("\n[2/6] Loading ground truth...")

t0 = time.time()

gt = pd.read_csv(GT, sep="\t", dtype=str)

gt_s1 = get_col(gt, ["source1_entity_id"])
gt_match = get_col(gt, ["matched_entity_ids"])

if not gt_s1 or not gt_match:
    raise RuntimeError(f"Ground truth columns not found. Columns: {gt.columns.tolist()}")

gt = gt[gt[gt_s1].isin(sample_ids)].copy()

truth = {}

for _, row in gt.iterrows():
    sid = str(row[gt_s1])
    matches = split_gt_ids(row[gt_match])
    truth[sid] = set(matches)

total_truth = sum(len(v) for v in truth.values())

print(f"GT S1 rows: {len(truth):,}")
print(f"GT pairs: {total_truth:,}")
print(f"GT load time: {time.time() - t0:.1f}s")


# ============================================================
# LOAD EXISTING V6 CANDIDATES
# ============================================================

print("\n[3/6] Reading existing V6 candidate files...")

existing = set()

v6_files = [
    PROC / "blocking_test_s1_s2_v6.tsv",
    PROC / "blocking_test_s1_s3_v6.tsv"
]

for path in v6_files:

    if not path.exists():
        print("Missing:", path)
        continue

    print("\nReading:", path.name)

    file_start = time.time()
    chunk_number = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=["source1_entity_id", "candidate_entity_ids"],
        chunksize=CHUNK
    ):

        chunk_number += 1

        chunk = chunk[chunk["source1_entity_id"].isin(sample_ids)]

        for sid, raw in zip(chunk["source1_entity_id"], chunk["candidate_entity_ids"]):

            if pd.isna(raw):
                continue

            sid = str(sid)
            candidates = parse_candidate_ids(raw)

            for candidate in candidates:
                existing.add((sid, candidate))

        if chunk_number % 5 == 0:
            print(
                f"  chunks={chunk_number:,} "
                f"pairs={len(existing):,} "
                f"time={time.time() - file_start:.1f}s",
                flush=True
            )

print(f"\nExisting V6 candidate pairs: {len(existing):,}")


# ============================================================
# BUILD MISSED GROUND TRUTH
# ============================================================

print("\n[4/6] Calculating missed ground-truth pairs...")

missed = []
covered_count = 0

for sid, matches in truth.items():
    for target in matches:
        pair = (sid, target)
        if pair in existing:
            covered_count += 1
        else:
            missed.append(pair)

missed_count = len(missed)
baseline_recall = covered_count / max(total_truth, 1)

print(f"GT pairs:          {total_truth:,}")
print(f"Covered by V6:     {covered_count:,}")
print(f"Missed by V6:      {missed_count:,}")
print(f"V6 pair recall:    {baseline_recall:.4%}")

if not missed:
    print("\nNo missed pairs. V6 already covers all sampled GT pairs.")
    sys.exit(0)

miss_set = set(missed)
miss_s1 = {sid for sid, _ in missed}


# ============================================================
# PREPARE S1 FEATURES + BUILD INDEXES (single pass)
# ============================================================

print("\n[5/6] Building S1 blocking indexes...")

s1 = s1[s1[s1_id].isin(miss_s1)].copy()

s1["tmp_id"] = s1[s1_id].astype(str)
s1["tmp_name"] = s1[name1].map(norm)
s1["tmp_addr"] = s1[addr1].map(norm)
s1["tmp_country"] = s1[country1].map(norm)
s1["tmp_prefix3"] = s1["tmp_name"].str.replace(" ", "", regex=False).str[:3]
s1["tmp_tokens"] = s1["tmp_name"].str.split()
s1["tmp_num"] = s1["tmp_addr"].str.extract(r"(\d+)", expand=False).fillna("")

exact_index = {}
prefix_index = {}
number_index = {}
s1_info = {}

for row in s1.itertuples(index=False):

    sid = str(row.tmp_id)
    nm = row.tmp_name
    country = row.tmp_country
    prefix = row.tmp_prefix3
    number = row.tmp_num
    tokens = set(row.tmp_tokens) if row.tmp_tokens else set()

    exact_index.setdefault((nm, country), []).append(sid)
    prefix_index.setdefault((prefix, country), []).append(sid)

    if number:
        number_index.setdefault((number, country), []).append(sid)

    s1_info[sid] = {
        "name": nm,
        "tokens": tokens,
        "country": country,
        "number": number
    }

print(f"S1 records indexed: {len(s1_info):,}")
print(f"Exact-name keys: {len(exact_index):,}")
print(f"Prefix keys:     {len(prefix_index):,}")
print(f"Number keys:     {len(number_index):,}")


# ============================================================
# RULE HIT STORAGE
# ============================================================

RULE_ORDER = [
    "exact_name_country",
    "prefix3_country",
    "name_token_overlap",
    "address_number_country",
    "name_and_address_token",
]

rule_hits = {rule: set() for rule in RULE_ORDER}


# ============================================================
# PROCESS ONE SOURCE
# ============================================================

def process_source(path):

    print(f"\nScanning {path.name}")

    id_col, name_col, addr_col, country_col = resolve_source_columns(path)

    print(
        f"  Resolved columns -> id={id_col!r} name={name_col!r} "
        f"address={addr_col!r} country={country_col!r}"
    )

    start = time.time()
    chunk_no = 0

    use_columns = [id_col, name_col, addr_col, country_col]

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=use_columns,
        chunksize=CHUNK
    ):

        chunk_no += 1

        chunk["tmp_name"] = chunk[name_col].map(norm)
        chunk["tmp_addr"] = chunk[addr_col].map(norm)
        chunk["tmp_country"] = chunk[country_col].map(norm)
        chunk["tmp_prefix3"] = chunk["tmp_name"].str.replace(" ", "", regex=False).str[:3]
        chunk["tmp_num"] = chunk["tmp_addr"].str.extract(r"(\d+)", expand=False).fillna("")

        for row in chunk.itertuples(index=False):

            target_id = str(getattr(row, id_col))
            name = row.tmp_name
            country = row.tmp_country
            prefix = row.tmp_prefix3
            number = row.tmp_num
            target_tokens = set(name.split()) if name else set()

            # RULE 1: exact normalized name + country
            for sid in exact_index.get((name, country), []):
                pair = (sid, target_id)
                if pair in miss_set:
                    rule_hits["exact_name_country"].add(pair)

            # RULE 2: prefix3 + country
            prefix_candidates = prefix_index.get((prefix, country), [])

            for sid in prefix_candidates:

                pair = (sid, target_id)

                if pair not in miss_set:
                    continue

                rule_hits["prefix3_country"].add(pair)

                info = s1_info[sid]
                source_tokens = info["tokens"]

                # RULE 3: name token overlap >= 50%
                if target_tokens and source_tokens:
                    overlap = (
                        len(target_tokens & source_tokens)
                        / min(len(target_tokens), len(source_tokens))
                    )
                    if overlap >= 0.50:
                        rule_hits["name_token_overlap"].add(pair)

                # RULE 5: name token + same address number
                source_number = info["number"]
                if number and source_number and number == source_number:
                    if len(target_tokens & source_tokens) >= 1:
                        rule_hits["name_and_address_token"].add(pair)

            # RULE 4: address number + country
            if number:
                for sid in number_index.get((number, country), []):
                    pair = (sid, target_id)
                    if pair in miss_set:
                        rule_hits["address_number_country"].add(pair)

        if chunk_no % 5 == 0:
            elapsed = time.time() - start
            print(
                f"  chunks={chunk_no:,} "
                f"rows={chunk_no * CHUNK:,} "
                f"elapsed={elapsed:.1f}s",
                flush=True
            )

    print(f"Finished {path.name} in {time.time() - start:.1f}s")


# ============================================================
# SCAN S2 + S3
# ============================================================

process_source(S2)
process_source(S3)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("RESULTS")
print("=" * 70)

covered = set()
matching_results = []  # one row per missed pair, first rule that recovers it

for rule in RULE_ORDER:

    hits = rule_hits[rule]
    incremental = hits - covered

    for sid, target_id in incremental:
        matching_results.append({
            "source1_entity_id": sid,
            "matched_entity_id": target_id,
            "recovered_by_rule": rule,
        })

    covered |= hits

    print(
        f"{rule:28s} "
        f"hits={len(hits):10,d} "
        f"incremental={len(incremental):10,d}"
    )

new_recall = (covered_count + len(covered)) / max(total_truth, 1)

print("-" * 70)
print(f"V6 baseline recall: {baseline_recall:.4%}")
print(f"Missed GT pairs:    {missed_count:,}")
print(f"Recovered by new rules: {len(covered):,}")
print(f"Potential recall: {new_recall:.4%}")
print("=" * 70)


# ============================================================
# MATCHING RESULTS
# ============================================================

matching_results_df = pd.DataFrame(
    matching_results,
    columns=["source1_entity_id", "matched_entity_id", "recovered_by_rule"]
)

print("\n")
print("=" * 70)
print("MATCHING RESULTS")
print("=" * 70)

print(f"Total recovered pairs in matching_results: {len(matching_results_df):,}")

if not matching_results_df.empty:

    print("\nBreakdown by recovering rule:")
    print(
        matching_results_df["recovered_by_rule"]
        .value_counts()
        .rename_axis("rule")
        .reset_index(name="count")
        .to_string(index=False)
    )

    print("\nSample rows:")
    print(matching_results_df.head(20).to_string(index=False))

    still_missed = missed_count - len(matching_results_df)
    print(f"\nStill missed after new rules: {still_missed:,}")

    try:
        PROC.mkdir(parents=True, exist_ok=True)
        matching_results_df.to_csv(MATCHING_RESULTS_OUT, sep="\t", index=False)
        print(f"\nmatching_results saved to: {MATCHING_RESULTS_OUT}")
    except OSError as e:
        print(f"\nCould not save matching_results TSV: {e}")

else:
    print("No pairs were recovered by the new rules.")

print("=" * 70)